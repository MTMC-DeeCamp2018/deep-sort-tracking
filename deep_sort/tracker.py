# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, index=0,camera_num=1,max_iou_distance=0.7, max_age=30, n_init=0):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.camera_num = camera_num
        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self.track_dic = {}
        self.track_global_dic = {}
        self._next_id = 0
        self.index = index

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections,tracker_dic,global_id,global_track):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections,matches_c = \
            self._match(detections,tracker_dic,self.camera_num,global_track, global_id)

        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for global_track_idx, detection_idx in matches_c:
            mean, covariance = self.kf.initiate(detections[detection_idx].to_xyah())
            track = Track(mean, covariance,self._next_id,self.n_init, self.max_age,global_track_idx,self.index,
            detections[detection_idx].feature)
            track.update(self.kf,detections[detection_idx], True)
            self.track_global_dic[global_track_idx] = track            
            self.tracks.append(track)
            self.track_dic[self._next_id] = track
            self._next_id += 1
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx],global_track,global_id)
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed() ]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            # print("the num of feature is {}, feature dimension is {}".format(len(features),len(features[0]))
            targets += [track.track_id for _ in track.features]
            # track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets,self.index)


    def _match(self, detections,tracker_dic,camera_num,global_track,global_id):

        def gated_metric(tracks, dets, track_indices, detection_indices, cross_camera = False):
            # print ("detection_indices is {}".format(detection_indices))
            features = np.array([dets[i].feature for i in detection_indices])
            if cross_camera == True:
                targets = np.array([tracks[i].global_id for i in track_indices])
            else:
                targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets,self.index,cross_camera,global_track,global_id)
            if cross_camera == False:
                cost_matrix = linear_assignment.gate_cost_matrix(
                    self.kf, cost_matrix, tracks, dets, track_indices,
                    detection_indices)
            print ("cost_matrix is {}".format(cost_matrix))
            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections,confirmed_tracks, None,tracker_dic,self.index,camera_num,global_track)
        # print ("initial matches are {} unmatched tracks_a are {}, unmatched_detections are {}".format(matches_a,unmatched_tracks_a,unmatched_detections))
        print ("initial matches are {}".format(matches_a))
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b = []
        unmatched_tracks_b = []
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        matches_c = []
        print ("second matches are {}".format(matches_b))
        if camera_num > 1 and len(unmatched_detections) > 0:
            matches_c, unmatched_detections_c = linear_assignment.cross_camera_matching(gated_metric, self.metric.matching_threshold, global_track, detections,global_id,unmatched_detections)
            unmatched_detections = list(set(unmatched_detections)-set([i[1] for i in matches_c]))

            matches_detections = [i[1] for i in matches]
            matches_c = [i for i in matches_c if i[1] not in matches_detections]
            print ("cross_camera match is {}".format(matches_c))
            for i in range(len(matches_c)):
                print ("the distance between matches_c is {}".format(1. - np.dot(global_track[matches_c[0][0]].features[-1], detections[matches_c[0][1]].feature.T)))
        return matches, unmatched_tracks, unmatched_detections, matches_c

    def _initiate_track(self, detection,global_track,global_id):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        track = Track(
            mean, covariance, self._next_id, self.n_init, self.max_age, global_id[0],self.index,
            detection.feature)
        self.tracks.append(track)
        self.track_global_dic[global_id[0]] = track
        self.track_dic[self._next_id] = track
        self._next_id += 1
        global_track.append(track)
        print ("global count is {}".format(global_id[0]))
        global_id[0] = global_id[0]+1

