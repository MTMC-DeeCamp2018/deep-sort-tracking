# vim: expandtab:ts=4:sw=4
import numpy as np
from . import epfl_calib


def _pdist(a, b,data_is_normalized=False):
    """Compute pair-wise squared distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _nn_euclidean_distance(x, y):
    """ Helper function for nearest neighbor distance metric (Euclidean).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.

    """
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y):
    """ Helper function for nearest neighbor distance metric (cosine).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest cosine distance to a sample in `x`.

    """
    distances = _cosine_distance(x, y, False)
    return distances.min(axis=0)


class NearestNeighborDistanceMetric(object):
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.

    Parameters
    ----------
    metric : str
        Either "euclidean" or "cosine".
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.

    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.

    """

    def __init__(self, metric, matching_threshold, budget=None,world_viewer_threshold=None):


        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.world_viewer_threshold = world_viewer_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets,index):
        """Update the distance metric with new data.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : ndarray
            An integer array of associated target identities.
        active_targets : List[int]
            A list of targets that are currently present in the scene.

        """
        # print ("current active targets are {}".format(active_targets))
        # print ("current features is {}".format(features))
        for feature, target in zip(features, targets):
            # print ("size of feature is {}, feature is {}".format(len(feature),feature))
            if index not in self.samples.keys():
                self.samples[index] = {}
            self.samples[index].setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[index][target] = self.samples[index][target][-self.budget:]
        self.samples[index] = {k: self.samples[index][k] for k in active_targets}
        # print ("current samples are {}".format(self.samples[index]))

    def distance(self, features, coordinates,targets,index,cross_camera,global_track,global_id, tracker_dic):
        """Compute distance between features and targets.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : List[int]
            A list of targets to match the given `features` against.

        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.

        """
        if cross_camera == False:
            cost_matrix = np.zeros((len(targets), len(features)))
            for i, target in enumerate(targets):
                cost_matrix[i, :] = self._metric(self.samples[index][target], features)
        else:
            cost_matrix = np.zeros((global_id[0], len(features)))
            for i in range(global_id[0]):
                if global_track[i].camera_index == index or i in tracker_dic[index].track_global_dic and tracker_dic[index].track_global_dic[i].is_confirmed():
                    cost_matrix[i,:] = 1e+5
                else:
                    cost_matrix[i,:] = self._metric(global_track[i].features, features)
                    # print ("the cost matrix is {}".format(cost_matrix[i,:]))
                    if self.world_viewer_threshold:
                        track_coordinate = global_track[i].trace[-1]
                        # track_coordinate = np.array(track_coordinate, dtype=np.float32)
                        # track_coordinate = epfl_calib.img_to_world(track_coordinate, epfl_calib.terrace_H()[global_track[i].camera_index])
                        for j in range(len(coordinates)):
                            detection_coordinate = [coordinates[j][0]+coordinates[j][2]/2, coordinates[j][1]+coordinates[j][3]]
                            # print ("the detection coordinate is {}".format(detection_coordinate))
                            detection_coordinate = np.array(detection_coordinate, dtype=np.float32)
                            detection_coordinate = epfl_calib.img_to_world(detection_coordinate, epfl_calib.terrace_H()[index])
                            print ("track_coordinate is {}, detection_coordinate is {}".format(track_coordinate,detection_coordinate))
                            squared_distance = (detection_coordinate[0]-track_coordinate[0]) ** 2 + (detection_coordinate[1]-track_coordinate[1]) ** 2
                            print ("camera index is {},track id is {}, detection id is {},potential match squared distance is {}, the cost matrix is {}, total is {}".format(index,i,j,squared_distance,cost_matrix[i][j],cost_matrix[i][j] + squared_distance / 10000)) 
                            if cost_matrix[i][j] > self.matching_threshold or squared_distance / 10000 > self.world_viewer_threshold:
                                cost_matrix[i][j] = 1e+5
                            else:
                                # print("cost matrix entry is {}".format(cost_matrix[i][j]))
                                cost_matrix[i][j] = cost_matrix[i][j] + squared_distance / 10000
                                # cost_matrix[i][j] = squared_distance / 10000
                                # print (cost_matrix[i][j])

            # print (cost_matrix)
        return cost_matrix
