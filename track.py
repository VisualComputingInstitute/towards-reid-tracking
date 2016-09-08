#TODO: comments/doc

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag,inv
from filterpy.common import Q_discrete_white_noise

class Track(object):

    """ Implements a track (not a tracker, a track).
    With KalmanFilter and some other stuff like status for track management

    Attributes
    ----------
    TODO

    """

    def __init__(self, init_x, init_P, dt, curr_frame, track_dim=4, det_dim=2, track_id=-1):
        self.track_id = track_id
        self.color = np.random.rand(1, 3).tolist()[0]
        self.xs=[]
        self.Ps=[]
        self.KF = KalmanFilter(dim_x=track_dim, dim_z=det_dim)
        self.KF.F = np.array([[1, dt, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, dt],
                              [0, 0, 0, 1]])
        q = Q_discrete_white_noise(dim=2, dt=dt, var=5000.)
        self.KF.Q = block_diag(q, q)  # TODO: matrix design for all the filters
        self.KF.H = np.array([[1, 0, 0, 0],
                              [0, 0, 1, 0]])
        self.KF.R = np.array([[10., 0],
                              [0, 10.]])
        self.KF.x = init_x
        self.KF.P = init_P

        self.missed_for = 0;
        self.deleted_at = 0;
        self.last_matched_at = curr_frame
        self.created_at = curr_frame

        self.status = 'matched' # matched, missed, deleted

        #missed for [delete_thresh] times? delete!
        self.delete_thresh = 5

    #def plot_track:

    def track_is_missed(self,curr_frame):
        self.missed_for += 1
        self.status = 'missed'
        if(self.missed_for>=self.delete_thresh):
            self.track_is_deleted(curr_frame)

    def track_is_matched(self,curr_frame):
        self.last_matched_at = curr_frame
        self.status = 'matched'
        self.missed_for = 0

    def track_is_deleted(self,curr_frame):
        self.deleted_at = curr_frame
        self.status = 'deleted'