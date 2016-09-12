#TODO: comments/doc

import numpy as np
from filterpy.kalman import KalmanFilter
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

    def __init__(self, init_x, init_P, dt, curr_frame, track_dim=4, det_dim=2, track_id=-1):
        self.track_id = track_id
        self.color = np.random.rand(3)
        self.xs=[init_x]
        self.Ps=[init_P]
        self.KF = KalmanFilter(dim_x=track_dim, dim_z=det_dim)
        self.KF.F = np.array([[1, dt, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, dt],
                              [0, 0, 0, 1]], dtype=np.float64)
        q = Q_discrete_white_noise(dim=2, dt=dt, var=5000.)
        self.KF.Q = block_diag(q, q)  # TODO: matrix design for all the filters
        self.KF.H = np.array([[1, 0, 0, 0],
                              [0, 0, 1, 0]], dtype=np.float64)
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
        self.delete_thresh = 5

    def plot_track(self, plot_past_trajectory=False):
        plot_covariance_ellipse((self.KF.x[0], self.KF.x[2]), self.KF.P, fc=self.color, alpha=0.4, std=[1,2,3])
        if plot_past_trajectory and len(self.xs)>1:
            for i in range(len(self.xs)-1):
                #TODO: incorporate camera info if available
                plt.plot([self.xs[i][0], self.xs[i+1][0]], [self.xs[i][2], self.xs[i+1][2]],
                         linewidth=2.0, color=self.color)

    def track_is_missed(self,curr_frame):
        self.missed_for += 1
        self.status = 'missed'
        if self.missed_for >= self.delete_thresh:
            self.track_is_deleted(curr_frame)
        else:
            self.age += 1
            self.xs.append(self.KF.x)
            self.Ps.append(self.KF.P)

    def track_is_matched(self,curr_frame):
        self.last_matched_at = curr_frame
        self.status = 'matched'
        self.missed_for = 0
        self.age += 1
        self.xs.append(self.KF.x)
        self.Ps.append(self.KF.P)

    def track_is_deleted(self,curr_frame):
        self.deleted_at = curr_frame
        self.status = 'deleted'
