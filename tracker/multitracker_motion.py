import numpy as np
from numba import jit
from collections import deque
import itertools
import os
import os.path as osp
import time
import torch
import torch.nn.functional as F
import sys

from utils.utils import *
from utils.log import logger
from utils.kalman_filter import KalmanFilter
from models import *
from tracker import matching
from .basetrack import BaseTrack, TrackState


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, temp_feat, buffer_size=30):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        """ when an strack is initialized for the first time, 
        set strack.stage = TrackState.tracked, strack.is_activated = False """
        self.is_activated = False  # BaseTrack __init__, state = TrackState.New
        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9
    
    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat 
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            #  We maintain a single feature vector for a tracklet by moving-averaging the features in each frame, with a momentum \alpha.
            self.smooth_feat = self.alpha *self.smooth_feat + (1-self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            """ Q: Why in predict(), if state is not Tracked, you zero one of the state variables (velocity of h)?
            A: For the first question, we find that setting the velocity to zero decreases the ID switches. 
            Keeping predicting the states of lost tracklets indeed brings more ID recall, 
            but in more cases, the tracklets tend to drift, which introduces many wrong assignments. 
            Therefore the overall IDS increases. 
            如果一直预测丢失的轨迹，虽然会提升recall，但是会带来轨迹漂移，从而导致错误分配，从而导致IDs上升
            We are looking for a better association algorithm to address these problems. 
            As for zeroing the velocity, we have not investigated whether it is good to zero the whole velocity vector (vx, vy, va, vh). This part of code is adapted from longcw/MOTDT."""
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i,st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        """ strack.is_activated will be set to True when this strack is associated by another observation (via IOU distance) in the consecutive frames. """
        self.is_activated = True 
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    #@jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    #@jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    #@jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    #@jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    #@jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        self.model = Darknet(opt.cfg)
        # load_darknet_weights(self.model, opt.weights)
        self.model.load_state_dict(torch.load(opt.weights, map_location='cpu')['model'], strict=False)
        self.model.cuda().eval()

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size

        self.kalman_filter = KalmanFilter()


    def update(self, im_blob, img0):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        t1 = time.time()
        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)  # 包括跟踪到的和丢失的轨迹，不包括的未确认的？
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)    # kalman滤波估计mean std, multi_mean, multi_covariance = STrack.shared_kalman.
        print("# strack_pool", len(strack_pool))
        sys.stdout.flush()
            
        ''' Step 1: Network forward, get detections & embeddings'''
        self.opt.conf_thres = 0.3
        self.opt.nms_thres = 0.8
        with torch.no_grad():
            pred = self.model(im_blob)  # im_blob: torch.Size([1, 3, 480, 864]), pred: torch.Size([1, 34020, 518])
            print("# real dets:", len(pred)) 
            sys.stdout.flush()

        pred = pred[pred[:, :, 4] > self.opt.conf_thres]  # 0.5  #TODO, 一般的置信度是多少？还是要删掉置信度太低的  # torch.Size([68, 518])
        print("# 1-pass filter dets:", len(pred)) 
        sys.stdout.flush()

        if len(pred) > 0:
            # dets = non_max_suppression(pred.unsqueeze(0), 0.3, 0.8)[0]   # conf_thres: 0.5->0.3, nms_thres: 0.4->0.8
            dets = pred
            motion_dists = matching.iou_motion(strack_pool, dets)  # 已有的轨迹的预测结果叫做strack_pool
            '''cost_matrix[row] = lambda_ * cost_matrix[row] + (1-lambda_)* gating_distance, lambda_=0.98'''
            # alpha = 2.0
            # motion_dists = torch.squeeze(motion_dists, 0)  # argument 'input' (position 1) must be Tensor, not numpy.ndarray
            # print(torch.from_numpy(motion_dists).dtype)
            # print(dets[:, 4].dtype)
            # print("motion_dists", motion_dists.shape)
            # print("dets", dets.shape)
            # dets[:, 4] = alpha * dets[:, 4] + (1 - alpha) * torch.from_numpy(motion_dists).float().cuda()
            # dets[:, 4] = alpha * dets[:, 4] + (1 - alpha) * torch.from_numpy(motion_dists).cuda()
            # dets[:, 4] = dets[:, 4] + alpha * torch.from_numpy(motion_dists).cuda()
            dets[:, 4] = dets[:, 4] + 2.0 * torch.from_numpy(motion_dists).cuda()
            dets = non_max_suppression(dets.unsqueeze(0), self.opt.conf_thres, 
                                        self.opt.nms_thres)[0]
            scale_coords(self.opt.img_size, dets[:, :4], img0.shape).round()
            dets, embs = dets[:, :5].cpu().numpy(), dets[:, 6:].cpu().numpy()
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30) for
                          (tlbrs, f) in zip(dets, embs)]
        else:
            detections = []
        
        '''cost_matrix[row] = lambda_ * cost_matrix[row] + (1-lambda_)* gating_distance, lambda_=0.98'''
        
        dists = matching.embedding_distance(strack_pool, detections)  
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)  
        # dists = matching.iou_distance(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        ''' 对于上次没有关联上的量测，以下代码基本没有改动 '''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state==TrackState.Tracked ]  # 如果以前是关联上的，但是今天没有关联
        dists = matching.iou_distance(r_tracked_stracks, detections)
        # dists = matching.embedding_distance(r_tracked_stracks, detections)  
        # dists = matching.fuse_motion(self.kalman_filter, dists, r_tracked_stracks, detections)  
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)
        
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))
        return output_stracks

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):  # 如果该tid不在集合中，则加入集合
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist<0.15)
    dupa, dupb = list(), list()
    for p,q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)  # 保留比较长的轨迹，pq是轨迹的index
        else:
            dupa.append(p)
    resa = [t for i,t in enumerate(stracksa) if not i in dupa]  # 将对应index的删掉
    resb = [t for i,t in enumerate(stracksb) if not i in dupb]
    return resa, resb
            

