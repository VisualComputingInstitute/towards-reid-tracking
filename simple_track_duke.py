# -*- coding: utf-8 -*-
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
    TODO

    """

    def __init__(self, dt, curr_frame, init_pose, track_dim=4, det_dim=2, track_id=-1,
                 embedding=None, debug_out_dir=None, init_thresh=3, delete_thresh=5,):
        self.debug_out_dir = debug_out_dir

        init_x = [init_pose[0], 0.0, init_pose[1], 0.0]
        init_P = [[200.0, 0, 0, 0], [0, 100.0, 0, 0], [0, 0, 200.0, 0], [0, 0, 0, 100.0]]

        self.track_id = track_id
        self.color = np.random.rand(3)
        self.xs=[init_x]
        self.Ps=[init_P]

        self.KF = KalmanFilter(dim_x=track_dim, dim_z=det_dim)
        self.KF.F = np.array([[1, dt, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, dt],
                              [0, 0, 0, 1]], dtype=np.float64)
        q = Q_discrete_white_noise(dim=2, dt=dt, var=50.)
        self.KF.Q = block_diag(q, q)
        self.KF.H = np.array([[1, 0, 0, 0],
                              [0, 0, 1, 0]], dtype=np.float64)
        self.KF.R = np.array([[50.0, 0],
                              [0, 50.0]], dtype=np.float64)
        self.KF.x = init_x
        self.KF.P = init_P

        self.missed_for = 0
        self.deleted_at = 0
        self.last_matched_at = curr_frame
        self.created_at = curr_frame

        self.age = 1 #age in frames

        #missed for [delete_thresh] times? delete!
        self.delete_thresh = delete_thresh #240=4 seconds ("occluded by car"-scenario in cam1)
        self.init_thresh = init_thresh  #of consecutive detection responses before reporting this track
        # set status: {init, matched, missed, deleted}
        if self.init_thresh == 1:
            self.status='matched'
        else:
            self.status='init'

        self.poses=[init_pose]

        #only if ReID is used for DA
        self.embedding = embedding

    # ==Track state==
    def track_predict(self):
        # standard KF
        self.KF.predict()

    def track_update(self, z):
        self.KF.update(z)

    # ==Track status management==
    def track_is_missed(self,curr_frame):
        self.missed_for += 1
        self.status = 'missed'
        if (self.missed_for >= self.delete_thresh) or (self.status=='init'):
            self.track_is_deleted(curr_frame)
        else:
            self.age += 1
            self.xs.append(self.KF.x)
            self.Ps.append(self.KF.P)
            self.poses.append([self.KF.x[0],self.KF.x[2]])

    def track_is_matched(self,curr_frame):
        self.last_matched_at = curr_frame
        self.missed_for = 0
        self.age += 1
        self.xs.append(self.KF.x)
        self.Ps.append(self.KF.P)
        self.poses.append([self.KF.x[0],self.KF.x[2]])
        if ((self.status=='init') and (curr_frame-self.created_at+1 < self.init_thresh)):
            pass # stay in init as long as threshold not exceeded
        else:
            self.status = 'matched' # in all other cases, go to matched

    def track_is_deleted(self,curr_frame):
        self.deleted_at = curr_frame
        self.status = 'deleted'

    # ==Evaluation==
    def get_track_eval_line(self,cid=1,frame=0):
        if (self.status == 'deleted' or self.status == 'init'):
            return None

        #pymot format
        #[height,width,id,y,x,z]
        #return {"height": 0, "width": 0, "id": self.track_id, "y": self.KF.x[2], "x": self.KF.x[0], "z": 0}
        #motchallenge format
        #TODO
        #dukeMTMC format
        #[cam, ID, frame, left, top, width, height, worldX, worldY]
        cX,cY = self.poses[-1]
        h = int(((all_bs[cid-1][0]+all_bs[cid-1][1]*cX) + (all_bs[cid-1][2]+all_bs[cid-1][3]*cY))/2)
        w = int(0.4*h)
        l = int(cX-w/2)
        t = int(cY-h/2)
        return [cid, self.track_id, lib.glob2loc(frame,cid), l, t, w, h, -1, -1]


    # ==Visualization==
    def plot_track(self, ax, plot_past_trajectory=False, output_shape=None):
        if (self.status == 'deleted' or self.status == 'init'):
            return

        #plot_covariance_ellipse((self.KF.x[0], self.KF.x[2]), self.KF.P, fc=self.color, alpha=0.4, std=[1,2,3])
        #print(self.poses)
        cX, vX, cY, vY = self.xs[-1]
        #print('vX: {}, vY: {}'.format(vX,vY))
        ax.plot(cX, cY, color=self.color, marker='o')
        ax.arrow(cX, cY, vX, vY, head_width=50, head_length=20, fc=self.color, ec=self.color)
        plot_covariance_ellipse((cX+vX, cY+vY), self.KF.P[1::2,1::2], fc=self.color, alpha=0.5, std=[3])
        plot_covariance_ellipse((cX, cY), self.KF.P[::2,::2], fc=self.color, alpha=0.5, std=[1, 2, 3])
        #plt.text(*self.state_to_output(*self.poses[-1], output_shape=output_shape), s='{}'.format(self.track_id))
        if plot_past_trajectory and len(self.poses)>1:
            outputs_xy = np.array(self.poses)
            ax.plot(*outputs_xy.T, linewidth=2.0, color=self.color)
