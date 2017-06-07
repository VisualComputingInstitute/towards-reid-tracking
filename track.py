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
                 dist_thresh=7, entropy_thresh=0.10,
                 unmiss_thresh=2, delete_thresh=90,
                 tp_hack=None, maxlife=None,
                 debug_out_dir=None):
        self.embed_crops_fn = embed_crops_fn
        self.debug_out_dir = debug_out_dir

        init_x = [0.0, 0.0]
        #self.init_P_scale = 200.0
        #self.init_P_scale = 5.0
        self.init_P_scale = 5.0**2

        self.DIST_THRESH = dist_thresh
        self.ENT_THRESH = entropy_thresh
        #self.VEL_MEAS_CERT_THRESH = 0.015

        self.KF = KalmanFilter(dim_x=2, dim_z=2)
        self.KF.F = np.array([[1, 0],
                              [0, 1]], dtype=np.float64)
        #q = Q_discrete_white_noise(dim=2, dt=dt, var=200.)
        #self.KF.Q = block_diag(q, q)  # TODO: matrix design for all the filters
        #self.KF.Q = q  # heatmap v only
        # 0.02
        #self.KF.Q = 0.02*np.eye(2)  # Process noise. Always added to prediction. Higher = uncertainty grows faster when no measurement
        self.KF.Q = 0.3**2*np.eye(2)  # Process noise. Always added to prediction. Higher = uncertainty grows faster when no measurement
        self.KF.H = np.array([[1, 0],
                              [0, 1]], dtype=np.float64)
        #self.KF.R = 100.0*np.eye(2)  # Measurement variance. Lower: jump more to measurement
        self.KF.R = 20.0**2*np.eye(2)  # Lower: jump more to measurement
        self.KF.x = init_x
        self.KF.P = self.init_P_scale*np.eye(2)

        self.track_id = track_id
        self.color = np.random.rand(3)
        self.hm_colormap = lbplt.linear_map((1,1,1), self.color)
        self.hm_colormap = lib.get_transparent_colormap(self.hm_colormap)
        self.xs=[self.KF.x]
        self.Ps=[self.KF.P]

        self.missed_for = 0
        self.missed_sightings = 0
        self.deleted_at = 0
        self.last_matched_at = curr_frame
        self.created_at = curr_frame
        self.n_exits = 0

        self.status = 'matched' # matched, missed, deleted
        self.age = 1 #age in frames
        self.MAXLIFE = maxlife
        self.TP_HACK = tp_hack

        #missed for [delete_thresh] times? delete!
        #self.DELETE_THRESH = 300 #90  # 1.5s
        self.DELETE_THRESH = delete_thresh  # 1.5s

        # How many times do I need to see him while he's missing to un-miss him?
        self.UNMISS_THRESH = unmiss_thresh

        self.state_shape = state_shape
        self.state_pad = state_pad
        self.output_shape = output_shape

        pad_y, pad_x = state_pad[0][0], state_pad[1][0]
        self.poses=[np.array([init_pose[0]+pad_x, init_pose[1]+pad_y])]

        self.embedding = None
        self.update_embedding(self.get_embedding_at_current_pos(image, curr_frame))

    def init_heatmap(self, heatmap):
        #self.pos_heatmap = self.resize_map_to_state(np.full_like(heatmap, 1/np.prod(heatmap.shape)))
        self.pos_heatmap = self.resize_map_to_state(heatmap)
        self.old_heatmap = None
        #self.id_heatmap = np.full_like(heatmap, 1/np.prod(self.pos_heatmap.shape))
        self.id_heatmap = self.resize_map_to_state(np.full_like(heatmap, 1/np.prod(heatmap.shape)))

        self.idmap_ent = 0.0 #lib.entropy_score_avg(self.id_heatmap)
        self.idmap_score = 9999  # np.min(id_distmap)
        self.this_map_good = False #self.idmap_score < self.DIST_THRESH and self.ENT_THRESH < self.idmap_ent

    # ==Heatmap stuff==
    def resize_map_to_state(self, heatmap, keep_sum=True):
        assert heatmap.shape == self.state_shape, "Lying Lucas giving me a heatmap that's not state-shaped!"
        #hm = np.pad(heatmap, self.state_pad, mode='constant', constant_values=1/np.prod(heatmap.shape))
        hm = np.pad(heatmap, self.state_pad, mode='edge')
        if keep_sum:
            hm /= np.sum(hm)*np.sum(heatmap)
        return hm
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

        return np.array([
            x/self.state_shape[1]*output_shape[1],
            y/self.state_shape[0]*output_shape[0]
        ])


    def states_to_outputs(self, xy, output_shape, ignore_padding=False):
        # xy is of shape (N,2)
        if output_shape is None:
            output_shape = self.output_shape

        if not ignore_padding:
            xy = xy - np.array([[self.state_pad[1][0], self.state_pad[0][0]]])

        factors = [output_shape[1]/self.state_shape[1],
                   output_shape[0]/self.state_shape[0]]
        return xy*factors

    def estimate_peak_xy(self, heatmap):
        #return lib.argmax2d_xy(heatmap)
        return lib.expected_xy(heatmap, magic_thresh=2)

    def get_velocity_estimate(self, old_heatmap, pos_heatmap):
        old_peak = self.estimate_peak_xy(old_heatmap)
        new_peak = self.estimate_peak_xy(pos_heatmap)
        return new_peak - old_peak

    def track_predict(self):
        vx, vy = self.KF.x
        #self.pred_heatmap = scipy.ndimage.shift(self.pos_heatmap, [vy, vx])
        gaussian = lib.gauss2d_xy(np.clip(self.KF.P, 1e-5, self.init_P_scale), nstd=2, mean=[-vx, -vy])
        self.pred_heatmap = lib.convolve_edge_same(self.pos_heatmap, gaussian)
        self.pred_heatmap /= np.sum(self.pred_heatmap)  # Re-normalize to probabilities

        # standard KF
        self.KF.predict()

    def track_update(self, id_heatmap, id_distmap, curr_frame, image_getter):
        self.age += 1

        # Hard rule for pathological cases.
        if self.MAXLIFE is not None and self.MAXLIFE < self.age:
            print("WARNING: Killing one of age.")
            return self.track_is_deleted(curr_frame)

        self.old_heatmap = self.pos_heatmap
        self.old_map_good = self.this_map_good

        self.id_heatmap = self.resize_map_to_state(id_heatmap)

        self.idmap_ent = lib.entropy_score_avg(self.id_heatmap)
        self.idmap_score = np.min(id_distmap)
        self.this_map_good = self.idmap_score < self.DIST_THRESH and self.ENT_THRESH < self.idmap_ent

        if self.this_map_good:
            self.pos_heatmap = self.pred_heatmap*self.id_heatmap
            self.pos_heatmap /= np.sum(self.pos_heatmap)  # Re-normalize to probabilities

            # Discard impossible jumps. TODO: It's a hack
            if self.TP_HACK is not None:
                xy = self.estimate_peak_xy(self.pos_heatmap)
                tpdist = np.sqrt(np.sum((self.poses[-1] - xy)**2))
                if tpdist > self.TP_HACK:
                    self.pos_heatmap = self.pred_heatmap
                    self.this_map_good = False
        else:
            self.pos_heatmap = self.pred_heatmap
            #self.pos_heatmap = self.pred_heatmap*lib.softmax(self.id_heatmap, T=10)
            #self.pos_heatmap /= np.sum(self.pos_heatmap)  # Re-normalize to probabilities
        #self.pos_heatmap = self.pred_heatmap*self.id_heatmap
        #self.pos_heatmap /= np.sum(self.pos_heatmap)  # Re-normalize to probabilities

        # Compute a velocity measurement from previous and current peaks in heatmap.
        # The certainty of the velocity measurement is a function of the certainties of
        # both position "measurements", i.e. how peaky both heatmaps are.
        #self.vel_meas_certainty = lib.entropy_score_avg(self.old_heatmap)*lib.entropy_score_avg(self.pos_heatmap)
        #self.vel_meas_certainty = prev_id_heatmap_ent*this_id_heatmap_ent
        #if self.VEL_MEAS_CERT_THRESH < self.vel_meas_certainty:
        if self.old_map_good and self.this_map_good:
            vel_measurement = self.get_velocity_estimate(self.old_heatmap, self.pos_heatmap)
            #self.KF.R = ... 
            self.KF.update(vel_measurement)

        self.xs.append(self.KF.x)
        self.Ps.append(self.KF.P)
        self.poses.append(self.estimate_peak_xy(self.pos_heatmap))

        if self.this_map_good:
            self.track_is_matched(curr_frame)

            # update embedding. Needs to happen after the above, as that updates current_pos.
            # TODO: Future work. Currently we only keep initial one.
            #self.update_embedding(self.get_embedding_at_current_pos(image_getter(), curr_frame))
        else:
            self.track_is_missed(curr_frame)

    # ==Track status management==
    def track_is_missed(self, curr_frame):
        self.missed_for += 1
        self.status = 'missed'
        if self.missed_for >= self.DELETE_THRESH: # or self.n_exits > 10:
            self.track_is_deleted(curr_frame)
        else:
            pass
            # TODO: Such "exit zones" are a workaround, a larger-than-image map would be better.
            #x, y = self.poses[-1]
            #vx, vy = self.xs[-1]
            #if (x == 0 and vx < 0) or \
            #   (x == self.pos_heatmap.shape[1]-1 and 0 < vx) or \
            #   (y == 0 and vy < 0) or \
            #   (y == self.pos_heatmap.shape[0]-1 and 0 < vy):
            #    self.n_exits += 1

    def track_is_matched(self, curr_frame):
        if 0 < self.missed_for:
            # Been missing until now, but...
            self.missed_sightings += 1

            # ...Only revive if seen enough times!
            if self.missed_sightings < self.UNMISS_THRESH:
                return

        self.last_matched_at = curr_frame
        self.status = 'matched'
        self.missed_for = 0
        self.missed_sightings = 0
        self.n_exits = 0

    def track_is_deleted(self,curr_frame):
        self.deleted_at = curr_frame
        self.status = 'deleted'

    # ==Evaluation==
    def get_track_eval_line(self, cid, frame):
        #dukeMTMC format
        #[cam, ID, frame, left, top, width, height, worldX, worldY]
        cX, cY = self.state_to_output(*self.poses[-1])
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
        vX, vY = self.state_to_output(*self.xs[-1], output_shape=output_shape, ignore_padding=True)*time_scale
        #print('vX: {}, vY: {}'.format(vX,vY))
        ax.plot(cX, cY, color=self.color, marker='o')
        ax.arrow(cX, cY, vX, vY, head_width=20, head_length=7, fc=self.color, ec=self.color, linestyle='--')
        # TODO: The cov is not in output space!
        #plot_covariance_ellipse((cX+vX, cY+vY), self.Ps[-1], fc=self.color, alpha=0.5, std=[1, 2, 3])
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
        if hasattr(self, 'idmap_score'):
            ax.text(*self.state_to_output(*self.poses[-1], output_shape=output_shape), s='{:.2f} | {:.3f}'.format(self.idmap_score, self.idmap_ent))
        return hm
