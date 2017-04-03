#TODO: comments/doc

import numpy as np
from filterpy.kalman import KalmanFilter
import scipy
from scipy import ndimage
from scipy import signal
from scipy.linalg import block_diag,inv
from filterpy.common import Q_discrete_white_noise
from filterpy.stats import plot_covariance_ellipse
import matplotlib.pyplot as plt
from os.path import join as pjoin

import lib
import lbtoolbox.plotting as lbplt

# all_bs for bbox regression
all_bs = np.array([[256.3190, -0.0207, 136.6533, 0.1978],
                    [212.9634, 0.0055, 126.0157, 0.2036],
                    [277.3869, -0.0154, 5.2019, 0.4442],
                    [-296.1867, 0.3356, 54.3528, 0.3093],
                    [258.1709, -0.0258, 144.2437, 0.2030],
                    [152.2878, 0.0296, -271.9162, 0.6985],
                    [208.9894, 0.0349, -298.6897, 0.7266],
                    [170.6156, 0.0128, 81.8043, 0.1659]])

HOT_CMAP = lib.get_transparent_colormap()


class Track(object):

    """ Implements a track (not a tracker, a track).
    With KalmanFilter and some other stuff like status for track management

    Attributes
    ----------
    TODO: Move to time using dt

    """

    def __init__(self, embed_crops_fn, curr_frame, init_pose, image,
                 state_shape, state_pad, output_shape, track_id=-1,
                 person_matching_threshold=0.5, debug_out_dir=None):
        self.embed_crops_fn = embed_crops_fn
        self.person_matching_threshold = person_matching_threshold
        self.debug_out_dir = debug_out_dir

        init_x = [0.0, 0.0]
        self.init_P_scale = 200.0

        self.VEL_MEAS_CERT_THRESH = 0.015

        self.track_id = track_id
        self.color = np.random.rand(3)
        self.hm_colormap = lbplt.linear_map((1,1,1), self.color)
        self.hm_colormap = lib.get_transparent_colormap(self.hm_colormap)
        self.xs=[init_x]
        self.Ps=[self.init_P_scale*np.eye(2)]

        self.KF = KalmanFilter(dim_x=2, dim_z=2)
        self.KF.F = np.array([[1, 0],
                              [0, 1]], dtype=np.float64)
        #q = Q_discrete_white_noise(dim=2, dt=dt, var=200.)
        #self.KF.Q = block_diag(q, q)  # TODO: matrix design for all the filters
        #self.KF.Q = q  # heatmap v only
        self.KF.Q = 0.02*np.eye(2)  # Higher: more prediction uncertainty.
        self.KF.H = np.array([[1, 0],
                              [0, 1]], dtype=np.float64)
        self.KF.R = 150.0*np.eye(2)  # Lower: jump more to measurement
        self.KF.x = init_x
        self.KF.P = self.init_P_scale*np.eye(2)

        self.missed_for = 0
        self.deleted_at = 0
        self.last_matched_at = curr_frame
        self.created_at = curr_frame
        self.n_exits = 0

        self.status = 'matched' # matched, missed, deleted
        self.age = 1 #age in frames

        #missed for [delete_thresh] times? delete!
        self.delete_thresh = 90  # 1.5s

        self.state_shape = state_shape
        self.state_pad = state_pad
        self.output_shape = output_shape

        pad_y, pad_x = state_pad[0][0], state_pad[1][0]
        self.poses=[np.array([init_pose[0]+pad_x, init_pose[1]+pad_y])]

        self.embedding = None
        self.update_embedding(self.get_embedding_at_current_pos(image, curr_frame))

    def init_heatmap(self, heatmap):
        self.pos_heatmap = self.resize_map_to_state(heatmap)
        self.old_heatmap = None
        self.id_heatmap = np.full_like(heatmap, 1/np.prod(self.pos_heatmap.shape))

    # ==Heatmap stuff==
    def resize_map_to_state(self, heatmap):
        assert heatmap.shape == self.state_shape, "Lying Lucas giving me a heatmap that's not state-shaped!"
        return np.pad(heatmap, self.state_pad, mode='constant')
        #return lib.resize_map(heatmap, self.state_shape, interp='bicubic')

    def unpad_state_map(self, statemap):
        return statemap[self.state_pad[0][0]:-self.state_pad[0][1],
                        self.state_pad[1][0]:-self.state_pad[1][1]]

    def get_crop_at_pos(self,pos,image):
        # TODO: fix bb: 128x48
        x, y = pos
        box_c = lib.box_centered(x, y, 128, 48, bounds=(0,0,image.shape[1],image.shape[0]))
        crop = lib.cutout_abs_hwc(image, box_c)
        return crop

    def get_embedding_at_current_pos(self, image, debug_curr_frame):
        crop = self.get_crop_at_pos(
            self.state_to_output(*self.poses[-1], output_shape=(image.shape[0], image.shape[1])),
            image
        )
        if self.debug_out_dir is not None:
            lib.imwrite(pjoin(self.debug_out_dir, 'crops', '{}-{}.jpg'.format(self.track_id, debug_curr_frame)), crop)
        return self.embed_crops_fn(crop[None], fake_id=self.track_id)[0]

    def update_embedding(self, new_embedding):
        if self.embedding is None:
            self.embedding = new_embedding
            self.n_embs_seen = 1
        else:
            return  # For this paper, we ignore new embeddings as the first is almost perfect.
            #self.embedding = self.embedding*self.n_embs_seen + new_embedding
            #self.n_embs_seen += 1
            #self.embedding /= self.n_embs_seen

    # ==Track state==
    def state_to_output(self, x, y, output_shape=None, ignore_padding=False):
        """
        The optional `output_shape` is in (H,W) format.
        """
        if output_shape is None:
            output_shape = self.output_shape

        if not ignore_padding:
            x = x - self.state_pad[1][0]
            y = y - self.state_pad[0][0]

        return [x/self.state_shape[1]*output_shape[1],
                y/self.state_shape[0]*output_shape[0]]


    def states_to_outputs(self, xy, output_shape, ignore_padding=False):
        # xy is of shape (N,2)
        if output_shape is None:
            output_shape = self.output_shape

        if not ignore_padding:
            xy = xy - np.array([[self.state_pad[1][0], self.state_pad[0][0]]])

        factors = [output_shape[1]/self.state_shape[1],
                   output_shape[0]/self.state_shape[0]]
        return xy*factors

    def get_velocity_estimate(self, old_heatmap, pos_heatmap):
        old_peak = lib.argmax2d_xy(old_heatmap)
        new_peak = lib.argmax2d_xy(pos_heatmap)
        return new_peak - old_peak

    def track_predict(self):
        # standard KF
        self.KF.predict()

        vx, vy = self.KF.x
        self.pred_heatmap = scipy.ndimage.shift(self.pos_heatmap, [vy, vx])
        gaussian = lib.gauss2d_xy(np.clip(self.KF.P, 1e-5, self.init_P_scale))
        self.pred_heatmap = lib.convolve_edge_same(self.pred_heatmap, gaussian)
        self.pred_heatmap /= np.sum(self.pred_heatmap) # Re-normalize to probabilities

    def track_update(self, id_heatmap, curr_frame, image_getter):
        prev_id_heatmap_ent = lib.entropy_score_avg(self.id_heatmap)
        self.id_heatmap = self.resize_map_to_state(id_heatmap)
        self.this_id_heatmap_ent = lib.entropy_score_avg(self.id_heatmap)

        self.old_heatmap = self.pos_heatmap

        # TODO: Maybe 0.09, but definitely in [0.05, 0.12].
        ID_MAP_THRESH = 0.1
        if ID_MAP_THRESH < self.this_id_heatmap_ent:
            self.pos_heatmap = self.pred_heatmap*self.id_heatmap
            self.pos_heatmap /= np.sum(self.pos_heatmap)  # Re-normalize to probabilities
        else:
            self.pos_heatmap = self.pred_heatmap

        # Compute a velocity measurement from previous and current peaks in heatmap.
        # The certainty of the velocity measurement is a function of the certainties of
        # both position "measurements", i.e. how peaky both heatmaps are.
        #self.vel_meas_certainty = lib.entropy_score_avg(self.old_heatmap)*lib.entropy_score_avg(self.pos_heatmap)
        self.vel_meas_certainty = prev_id_heatmap_ent*self.this_id_heatmap_ent
        if self.VEL_MEAS_CERT_THRESH < self.vel_meas_certainty:
            vel_measurement = self.get_velocity_estimate(self.old_heatmap, self.pos_heatmap)
            #self.KF.R = ... 
            self.KF.update(vel_measurement)

        if ID_MAP_THRESH < self.this_id_heatmap_ent:
            self.track_is_matched(curr_frame)

            # update embedding. Needs to happen after the above, as that updates current_pos.
            self.update_embedding(self.get_embedding_at_current_pos(image_getter(), curr_frame))
        else:
            self.track_is_missed(curr_frame)


    # ==Track status management==
    def track_is_missed(self,curr_frame):
        self.missed_for += 1
        self.status = 'missed'
        if self.missed_for >= self.delete_thresh or self.n_exits > 10:
            self.track_is_deleted(curr_frame)
        else:
            self.age += 1
            self.xs.append(self.KF.x)
            self.Ps.append(self.KF.P)
            xy = lib.argmax2d_xy(self.pos_heatmap)

            # TODO: Such "exit zones" are a workaround, a larger-than-image map would be better.
            x, y = xy
            vx, vy = self.xs[-1]
            if (x == 0 and vx < 0) or \
               (x == self.pos_heatmap.shape[1]-1 and 0 < vx) or \
               (y == 0 and vy < 0) or \
               (y == self.pos_heatmap.shape[0]-1 and 0 < vy):
                self.n_exits += 1
            self.poses.append(xy)

    def track_is_matched(self,curr_frame):
        self.last_matched_at = curr_frame
        self.status = 'matched'
        self.missed_for = 0
        self.n_exits = 0
        self.age += 1
        self.xs.append(self.KF.x)
        self.Ps.append(self.KF.P)
        self.poses.append(lib.argmax2d_xy(self.pos_heatmap))

    def track_is_deleted(self,curr_frame):
        self.deleted_at = curr_frame
        self.status = 'deleted'

    # ==Evaluation==
    def get_track_eval_line(self, cid, frame):
        #dukeMTMC format
        #[cam, ID, frame, left, top, width, height, worldX, worldY]
        cX,cY = self.state_to_output(*self.poses[-1])
        h = int(((all_bs[cid-1][0]+all_bs[cid-1][1]*cX) + (all_bs[cid-1][2]+all_bs[cid-1][3]*cY))/2)
        w = int(0.4*h)
        l = int(cX-w/2)
        t = int(cY-h/2)
        # id-shift-quick-hack for multi-cam eval.
        return [cid, self.track_id+cid*100000, lib.glob2loc(frame, cid), l, t, w, h, -1, -1]



    # ==Visualization==
    def plot_track(self, ax, plot_past_trajectory=False, output_shape=None, time_scale=1):
        if output_shape is None:
            output_shape = self.output_shape

        if self.status == 'deleted':
            return

        #plot_covariance_ellipse((self.KF.x[0], self.KF.x[2]), self.KF.P, fc=self.color, alpha=0.4, std=[1,2,3])
        #print(self.poses)
        cX, cY = self.state_to_output(*self.poses[-1], output_shape=output_shape)
        vX, vY = self.state_to_output(*self.xs[-1], output_shape=output_shape, ignore_padding=True)
        #print('vX: {}, vY: {}'.format(vX,vY))
        ax.plot(cX, cY, color=self.color, marker='o')
        ax.arrow(cX, cY, vX*time_scale, vY*time_scale, head_width=20, head_length=7, fc=self.color, ec=self.color, linestyle='--')
        plot_covariance_ellipse((cX+vX*10, cY+vY*10), self.Ps[-1], fc=self.color, alpha=0.5, std=[1, 2, 3])
        #plt.text(*self.state_to_output(*self.poses[-1], output_shape=output_shape), s='{}'.format(self.embedding))
        if plot_past_trajectory and len(self.poses)>1:
            outputs_xy = self.states_to_outputs(np.array(self.poses), output_shape)
            ax.plot(*outputs_xy.T, linewidth=2.0, color=self.color)


    def _plot_heatmap(self, ax, hm, output_shape=None):
        if self.status == 'deleted':
            return

        if output_shape is None:
            output_shape = self.output_shape

        return ax.imshow(self.unpad_state_map(hm), interpolation='none', cmap=self.hm_colormap,
                         #clim=(0, lib.ramp(lib.entropy_score(hm), 0.2, 1, 0.8, np.max(hm))), #alpha=0.5,
                         extent=[0, output_shape[1], output_shape[0], 0])

    def plot_pos_heatmap(self, ax, output_shape=None):
        hm = self._plot_heatmap(ax, self.pos_heatmap, output_shape)
        vX, vY = self.state_to_output(*self.xs[-1], output_shape=output_shape, ignore_padding=True)
        ax.text(*self.state_to_output(*self.poses[-1], output_shape=output_shape), s='{:.2f} ({:.2f}, {:.2f})'.format(np.sqrt(vX*vX + vY*vY), vX, vY))
        return hm

    def plot_pred_heatmap(self, ax, output_shape=None):
        hm = self._plot_heatmap(ax, self.pred_heatmap, output_shape)
        if hasattr(self, 'vel_meas_certainty'):
            ax.text(*self.state_to_output(*self.poses[-1], output_shape=output_shape), s='{:.8f}'.format(self.vel_meas_certainty))
        return hm

    def plot_id_heatmap(self, ax, output_shape=None):
        hm = self._plot_heatmap(ax, self.id_heatmap, output_shape)
        if hasattr(self, 'this_id_heatmap_ent'):
            ax.text(*self.state_to_output(*self.poses[-1], output_shape=output_shape), s='{:.8f}'.format(self.this_id_heatmap_ent))
        return hm
