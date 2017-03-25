#TODO: comments/doc

import numpy as np
from filterpy.kalman import KalmanFilter
import scipy
from scipy import ndimage
from scipy.linalg import block_diag,inv
from filterpy.common import Q_discrete_white_noise
from filterpy.stats import plot_covariance_ellipse
import matplotlib.pyplot as plt

class Track(object):

    """ Implements a track (not a tracker, a track).
    With KalmanFilter and some other stuff like status for track management

    Attributes
    ----------
    TODO

    """

    def __init__(self, init_x, init_P, dt, curr_frame, start_pose, init_heatmap, track_dim=2, det_dim=2, track_id=-1):
        self.track_id = track_id
        self.color = np.random.rand(3)
        self.xs=[init_x]
        self.Ps=[init_P]

        self.poses=[start_pose]

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

        self.pos_heatmap = init_heatmap
        self.old_heatmap = init_heatmap

    # ==Heatmap stuff==
    def get_peak_in_heatmap(self,heatmap):
        idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
        return [idx[1],idx[0]]

    # ==Track state==
    def get_velocity_estimate(self,old_heatmap,pos_heatmap):
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

    def track_update(self, id_heatmap):
        # heatmap
        normalizer = 1
        self.pos_heatmap = id_heatmap #(self.pos_heatmap*id_heatmap) / normalizer
        # standard KF
        vel_measurement = self.get_velocity_estimate(self.old_heatmap, self.pos_heatmap)
        self.KF.update(vel_measurement)


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
        return {"height": 0, "width": 0, "id": self.track_id, "y": self.KF.x[2], "x": self.KF.x[0], "z": 0}
        #motchallenge format
        #TODO
        #dukeMTMC format
        #TODO

    # ==Visualization==
    def plot_track(self, plot_past_trajectory=False, plot_heatmap=False):
        #plot_covariance_ellipse((self.KF.x[0], self.KF.x[2]), self.KF.P, fc=self.color, alpha=0.4, std=[1,2,3])
        #print(self.poses)
        plt.plot(self.poses[len(self.poses)-1][0],self.poses[len(self.poses)-1][1],linewidth=2.0, color=self.color)
        if plot_past_trajectory and len(self.poses)>1:
            for i in range(len(self.poses)-1):
                #TODO: incorporate camera info if available
                plt.plot([self.poses[i][0], self.poses[i+1][0]], [self.poses[i][1], self.poses[i+1][1]],
                         linewidth=2.0, color=self.color)
        if plot_heatmap:
            plt.imshow(self.pos_heatmap, alpha=0.5, interpolation='none', cmap='hot',
                       extent=[0, 1920, 1080, 0], clim=(0, 10)) #TODO: shape

#test = scipy.misc.imresize(image,(1920,1080),interp='bicubic',mode='F')
#fix bb: 128x48
