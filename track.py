#TODO: comments/doc

import numpy as np
from filterpy.kalman import KalmanFilter
import scipy
from scipy import ndimage
from scipy.linalg import block_diag,inv
from filterpy.common import Q_discrete_white_noise
from filterpy.stats import plot_covariance_ellipse
import matplotlib.pyplot as plt
from os.path import join as pjoin

# all_bs for bbox regression
all_bs = np.array([[255.9004, -0.0205, 138.4768, 0.1939],
                   [213.1062, 0.0039, 124.5369, 0.2044],
                   [279.7756, -0.0181, 12.4071, 0.4278],
                   [-384.7724, 0.3990, 54.3856, 0.3107],
                   [259.4900, -0.0271, 146.8988, 0.1994],
                   [147.8093, 0.0344, -275.1930, 0.7032],
                   [209.2198, 0.0354, -296.9731, 0.7247],
                   [175.5730, 0.0077, 76.2798, 0.1737]])


def stick_to_bounds(box, bounds=(0,0,1,1)):
    # bo is l t w h
    l, t, w, h = box
    bl, bt, bw, bh = bounds

    l += max(bl - l, 0)
    l -= max((l+w) - (bl+bw), 0)

    t += max(bt - t, 0)
    t -= max((t+h) - (bt+bh), 0)

    return l, t, w, h


def box_centered(cx, cy, h, w, bounds=(0, 0, 1, 1)):
    # box is l t w h
    return stick_to_bounds((cx - w / 2, cy - h / 2, w, h), bounds)


def cutout(img, box):
    # box is l, t, w, h
    l, t, w, h = map(int, box)
    return img[t:t+h,l:l+w]


def embed_crop(crop, fake_id):
    return fake_id


class Track(object):

    """ Implements a track (not a tracker, a track).
    With KalmanFilter and some other stuff like status for track management

    Attributes
    ----------
    TODO

    """

    def __init__(self, dt, curr_frame, init_heatmap, image,
                 state_shape, output_shape, track_dim=2, det_dim=2, track_id=-1):
        init_x = [0.0, 0.0]
        init_P = [[10.0, 0], [0, 10.0]]

        self.track_id = track_id
        self.color = np.random.rand(3)
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
        self.KF.R = np.array([[10, 0],
                              [0, 10]], dtype=np.float64)
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

        self.pos_heatmap = self.resize_to_state(init_heatmap)
        self.old_heatmap = self.resize_to_state(init_heatmap)

        self.poses=[self.get_peak_in_heatmap(self.pos_heatmap)]

        self.embedding = None
        crop = self.get_crop_at_pos(self.poses[0],image)
        self.update_embedding(embed_crop(crop, fake_id=track_id))  # TODO: Make real

    # ==Heatmap stuff==
    def resize_to_state(self, heatmap):
        return scipy.misc.imresize(heatmap, self.state_shape, interp='bicubic', mode='F')

    def get_crop_at_pos(self,pos,image):
        #fix bb: 128x48
        x,y = pos
        box_c = box_centered(x,y,128*2,48*2,bounds=(0,0,image.shape[0],image.shape[1]))
        crop = cutout(image,box_c)
        return crop

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
    def state_to_output(self, x, y):
        return [x/self.state_shape[1]*self.output_shape[1],
                y/self.state_shape[0]*self.output_shape[0]]

    def states_to_outputs(self, xy):
        # xy is of shape (N,2)
        factors = [self.output_shape[1]/self.state_shape[1],
                   self.output_shape[0]/self.state_shape[0]]
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
        self.pos_heatmap = scipy.ndimage.shift(self.pos_heatmap,self.KF.x) #TODO noise of v_cov
        self.pos_heatmap = scipy.ndimage.filters.gaussian_filter(self.pos_heatmap, (self.KF.P[0,0],self.KF.P[0,0]))  # TODO: non-diag cov

    def track_update(self, id_heatmap, curr_frame, image):
        id_heatmap = self.resize_to_state(id_heatmap)

        # heatmap
        normalizer = 1
        self.pos_heatmap = id_heatmap #(self.pos_heatmap*id_heatmap) / normalizer
        # standard KF
        vel_measurement = self.get_velocity_estimate(self.old_heatmap, self.pos_heatmap)
        self.KF.update(vel_measurement)

        if self.pos_heatmap.max() > 2.0: # TODO: magic threshold
            self.track_is_matched(curr_frame)
            # update embedding
            crop = self.get_crop_at_pos(self.poses[0], image)
            self.update_embedding(embed_crop(crop, fake_id=self.track_id))  # TODO: Make real
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
    def get_track_state_dict(self):
        #pymot format
        # TODO: don't forget to scale up from state shape to output shape.
        return {"height": 0, "width": 0, "id": self.track_id, "y": self.KF.x[2], "x": self.KF.x[0], "z": 0}
        #motchallenge format
        #TODO
        #dukeMTMC format
        #TODO

    # ==Visualization==
    def plot_track(self, plot_past_trajectory=False, plot_heatmap=False):
        #plot_covariance_ellipse((self.KF.x[0], self.KF.x[2]), self.KF.P, fc=self.color, alpha=0.4, std=[1,2,3])
        #print(self.poses)
        plt.plot(*self.state_to_output(*self.poses[-1]), color=self.color, marker='o')
        plt.text(*self.state_to_output(*self.poses[-1]),s='{}'.format(self.embedding))
        if plot_past_trajectory and len(self.poses)>1:
            outputs_xy = self.states_to_outputs(np.array(self.poses))
            plt.plot(*outputs_xy.T, linewidth=2.0, color=self.color)
        if plot_heatmap:
            plt.imshow(self.pos_heatmap, alpha=0.5, interpolation='none', cmap='hot',
                       extent=[0, self.output_shape[1], self.output_shape[0], 0], clim=(0, 10))
