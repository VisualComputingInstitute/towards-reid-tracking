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
all_bs = np.array([[255.9004, -0.0205, 138.4768, 0.1939],
                   [213.1062, 0.0039, 124.5369, 0.2044],
                   [279.7756, -0.0181, 12.4071, 0.4278],
                   [-384.7724, 0.3990, 54.3856, 0.3107],
                   [259.4900, -0.0271, 146.8988, 0.1994],
                   [147.8093, 0.0344, -275.1930, 0.7032],
                   [209.2198, 0.0354, -296.9731, 0.7247],
                   [175.5730, 0.0077, 76.2798, 0.1737]])

HOT_CMAP = lib.get_transparent_colormap()


class Track(object):

    """ Implements a track (not a tracker, a track).
    With KalmanFilter and some other stuff like status for track management

    Attributes
    ----------
    TODO

    """

    def __init__(self, embed_crop_fn, dt, curr_frame, init_heatmap, image,
                 state_shape, output_shape, track_dim=2, det_dim=2, track_id=-1,
                 person_matching_threshold=0.5, debug_out_dir=None):
        self.embed_crop_fn = embed_crop_fn
        self.person_matching_threshold = person_matching_threshold
        self.debug_out_dir = debug_out_dir

        init_x = [0.0, 0.0]
        init_P = [[20.0, 0], [0, 20.0]]

        self.track_id = track_id
        self.color = np.random.rand(3)
        self.hm_colormap = lbplt.linear_map(self.color,(1,1,1))
        self.hm_colormap = lib.get_transparent_colormap(self.hm_colormap)
        self.xs=[init_x]
        self.Ps=[init_P]

        self.KF = KalmanFilter(dim_x=track_dim, dim_z=det_dim)
        self.KF.F = np.array([[1, 0],
                              [0, 1]], dtype=np.float64)
        q = Q_discrete_white_noise(dim=2, dt=dt, var=10.)
        #self.KF.Q = block_diag(q, q)  # TODO: matrix design for all the filters
        self.KF.Q = q  # heatmap v only
        self.KF.H = np.array([[1, 0],
                              [0, 1]], dtype=np.float64)
        self.KF.R = np.array([[10.0, 0],
                              [0, 10.0]], dtype=np.float64)
        self.KF.x = init_x
        self.KF.P = init_P

        self.missed_for = 0
        self.deleted_at = 0
        self.last_matched_at = curr_frame
        self.created_at = curr_frame

        self.status = 'matched' # matched, missed, deleted
        self.age = 1 #age in frames

        #missed for [delete_thresh] times? delete!
        self.delete_thresh = 1

        self.state_shape = state_shape
        self.output_shape = output_shape

        self.pos_heatmap = self.resize_map_to_state(init_heatmap)
        self.old_heatmap = self.resize_map_to_state(init_heatmap)

        self.poses=[self.get_peak_in_heatmap(self.pos_heatmap)]

        self.embedding = None
        # TODO: Make TID real
        self.update_embedding(self.embed_crop_fn(self.get_crop_at_current_pos(image), fake_id=track_id))

    # ==Heatmap stuff==
    def resize_map_to_state(self, heatmap):
        assert heatmap.shape == self.state_shape, "Lying Lucas giving me a heatmap that's not state-shaped!"
        return np.array(heatmap)
        #return lib.resize_map(heatmap, self.state_shape, interp='bicubic')

    def get_crop_at_pos(self,pos,image):
        # TODO: fix bb: 128x48
        x, y = pos
        box_c = lib.box_centered(x, y, 128*2, 48*2, bounds=(0,0,image.shape[1],image.shape[0]))
        crop = lib.cutout_abs_hwc(image, box_c)
        return crop

    def get_crop_at_current_pos(self, image):
        return self.get_crop_at_pos(
            self.state_to_output(*self.poses[-1], output_shape=(image.shape[0], image.shape[1])),
            image
        )

    def get_peak_in_heatmap(self,heatmap):
        idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
        return [idx[1],idx[0]]

    def get_regressed_bbox_hw(self,pos):
        return[w,h]

    def update_embedding(self, new_embedding):
        if self.embedding is None:
            self.embedding = new_embedding
            self.n_embs_seen = 1
        else:
            self.embedding = self.embedding*self.n_embs_seen + new_embedding
            self.n_embs_seen += 1
            self.embedding /= self.n_embs_seen

    # ==Track state==
    def state_to_output(self, x, y, output_shape=None):
        """
        The optional `output_shape` is in (H,W) format.
        """
        if output_shape is None:
            output_shape = self.output_shape

        return [x/self.state_shape[1]*output_shape[1],
                y/self.state_shape[0]*output_shape[0]]

    def states_to_outputs(self, xy, output_shape):
        # xy is of shape (N,2)
        if output_shape is None:
            output_shape = self.output_shape

        factors = [output_shape[1]/self.state_shape[1],
                   output_shape[0]/self.state_shape[0]]
        return xy*factors

    def get_velocity_estimate(self,old_heatmap, pos_heatmap):
        old_peak = self.get_peak_in_heatmap(old_heatmap)
        new_peak = self.get_peak_in_heatmap(pos_heatmap)
        return np.subtract(new_peak,old_peak)

    def track_predict(self):
        # standard KF
        self.KF.predict()
        # heatmap
        self.old_heatmap = self.pos_heatmap
        self.pos_heatmap = scipy.ndimage.shift(self.pos_heatmap,self.KF.x)
        self.pos_heatmap = lib.convolve_edge_same(self.pos_heatmap, lib.gauss2d(self.KF.P))
        self.pos_heatmap /= np.sum(self.pos_heatmap) # Re-normalize to probabilities
        self.pred_heatmap = self.pos_heatmap

    def track_update(self, id_heatmap, curr_frame, image):
        self.id_heatmap = self.resize_map_to_state(id_heatmap)

        # heatmap
        self.pos_heatmap = self.pos_heatmap*self.id_heatmap
        self.pos_heatmap /= np.sum(self.pos_heatmap)  # Re-normalize to probabilities
        # standard KF
        vel_measurement = self.get_velocity_estimate(self.old_heatmap, self.pos_heatmap)
        self.KF.update(vel_measurement)

        if self.debug_out_dir is not None:
            with open(pjoin(self.debug_out_dir, 'pos_heatmaps.txt'), 'a') as f:
                vals = list(np.percentile(self.pos_heatmap, [0,90,95,99]))
                vals.extend(np.sort(self.pos_heatmap.flatten())[-5:])
                f.write(('{},{},' + ','.join(['{}']*len(vals)) + '\n').format(self.track_id, curr_frame, *vals))

        uniform = 1/np.prod(self.state_shape)
        T = uniform + self.person_matching_threshold*(1.0 - uniform)
        if T < self.pos_heatmap.max():
            self.track_is_matched(curr_frame)

            # update embedding. Needs to happen after the above, as that updates current_pos.
            crop = self.get_crop_at_current_pos(image)
            if self.debug_out_dir is not None:
                lib.imwrite(pjoin(self.debug_out_dir, 'crops', '{}-{}.jpg'.format(self.track_id, curr_frame)), crop)
            self.update_embedding(self.embed_crop_fn(crop, fake_id=self.track_id))  # TODO: Make real
        else:
            self.track_is_missed(curr_frame)


    # ==Track status management==
    def track_is_missed(self,curr_frame):
        self.missed_for += 1
        self.status = 'missed'
        if self.missed_for >= self.delete_thresh:
            self.track_is_deleted(curr_frame)
        else:
            self.age += 1
            self.xs.append(self.KF.x)
            self.Ps.append(self.KF.P)
            self.poses.append(self.get_peak_in_heatmap(self.pos_heatmap))

    def track_is_matched(self,curr_frame):
        self.last_matched_at = curr_frame
        self.status = 'matched'
        self.missed_for = 0
        self.age += 1
        self.xs.append(self.KF.x)
        self.Ps.append(self.KF.P)
        self.poses.append(self.get_peak_in_heatmap(self.pos_heatmap))

    def track_is_deleted(self,curr_frame):
        self.deleted_at = curr_frame
        self.status = 'deleted'

    # ==Evaluation==
    def get_track_eval_line(self,cid=1,frame=0):
        #pymot format
        #[height,width,id,y,x,z]
        #return {"height": 0, "width": 0, "id": self.track_id, "y": self.KF.x[2], "x": self.KF.x[0], "z": 0}
        #motchallenge format
        #TODO
        #dukeMTMC format
        #[cam, ID, frame, left, top, width, height, worldX, worldY]
        cX,cY = self.state_to_output(*self.poses[-1])
        h = int(((all_bs[cid-1][0]+all_bs[cid-1][1]*cX) + (all_bs[cid-1][2]+all_bs[cid-1][3]*cY))/2)
        w = int(0.4*h)
        l = int(cX-w/2)
        t = int(cY-h/2)
        return [cid, self.track_id, frame, l, t, w, h, -1, -1]


    # ==Visualization==
    def plot_track(self, ax, plot_past_trajectory=False, output_shape=None):
        if output_shape is None:
            output_shape = self.output_shape

        if self.status == 'deleted':
            return

        #plot_covariance_ellipse((self.KF.x[0], self.KF.x[2]), self.KF.P, fc=self.color, alpha=0.4, std=[1,2,3])
        #print(self.poses)
        cX, cY = self.state_to_output(*self.poses[-1], output_shape=output_shape)
        vX, vY = self.state_to_output(*self.xs[-1], output_shape=output_shape)
        #print('vX: {}, vY: {}'.format(vX,vY))
        ax.plot(cX, cY, color=self.color, marker='o')
        ax.arrow(cX, cY, vX*10, vY*10, head_width=15, head_length=5, fc=self.color, ec=self.color)
        plot_covariance_ellipse((cX+vX*10, cY+vY*10), self.Ps[-1], fc=self.color, alpha=0.5, std=[1, 2, 3])
        #plt.text(*self.state_to_output(*self.poses[-1], output_shape=output_shape), s='{}'.format(self.embedding))
        if plot_past_trajectory and len(self.poses)>1:
            outputs_xy = self.states_to_outputs(np.array(self.poses), output_shape)
            ax.plot(*outputs_xy.T, linewidth=2.0, color=self.color)


    def _plot_heatmap(self, ax, hm, output_shape=None):
        if output_shape is None:
            output_shape = self.output_shape

        return ax.imshow(hm, interpolation='none', cmap=self.hm_colormap, clim=(0, 1), #alpha=0.5,
                         extent=[0, output_shape[1], output_shape[0], 0])

    def plot_pos_heatmap(self, ax, output_shape=None):
        return self._plot_heatmap(ax, self.pos_heatmap, output_shape)

    def plot_pred_heatmap(self, ax, output_shape=None):
        return self._plot_heatmap(ax, self.pred_heatmap, output_shape)

    def plot_id_heatmap(self, ax, output_shape=None):
        return self._plot_heatmap(ax, self.id_heatmap, output_shape)
