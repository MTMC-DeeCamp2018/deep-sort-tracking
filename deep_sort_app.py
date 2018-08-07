# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np
from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker



def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info

def gather_video_info(video_dir,detection_file):
    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    min_frame_idx = int(detections[:, 0].min())
    max_frame_idx = int(detections[:, 0].max())
    video = cv2.VideoCapture(video_dir)
    feature_dim = detections.shape[1] - 6 if detections is not None else 0
    seq_info = {
        "detections": detections,
        "video": video,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
    }
    return seq_info

def create_detections(detection_mat, frame_idx, camera_index, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx
    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature, camera_index))
    return detection_list



def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display,video_dir=None,max_age=None):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    sequence_dir_list = []
    video_dir_list = []
    if video_dir:
        video_dir_list = video_dir.split(",")
    else:
        sequence_dir_list = sequence_dir.split(",")
    detection_file_list = detection_file.split(",")
    camera_num = len(detection_file_list)
    seq_info_dic = {}
    tracker_dic = {}
    global_track = []
    global_id = [0]
    for i in range(camera_num):
        if video_dir:
            seq_info_dic[i] = gather_video_info(video_dir_list[i],detection_file_list[i])
        else:
            seq_info_dic[i] = gather_sequence_info(sequence_dir_list[i], detection_file_list[i])
    # print ("initial seq_info_dic is {}".format(seq_info_dic))
    for i in range(camera_num):
        tracker_dic[i] = Tracker(nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget),i,camera_num,max_age=max_age)
    results = []

    def video_frame_processing(vis,frame_idx, index, image):
        print("Processing frame %05d" % frame_idx)
        frame_indices = seq_info_dic[index]["detections"][:, 0].astype(np.int)
        mask = frame_indices == frame_idx
        detection_list = []
        for row in seq_info_dic[index]["detections"][mask]:
            bbox, confidence, feature = row[1:5], row[5], row[6:]
            # if bbox[3] < min_height:
            #     continue
            detection_list.append(Detection(bbox, confidence, feature, index))
        detections = [d for d in detection_list if d.confidence >= min_confidence]
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        tracker_dic[index].predict()
        tracker_dic[index].update(detections,tracker_dic,global_id,global_track)
        if display:
            if index == 0:
                vis.reset_image()
            vis.append_image(image.copy())
            vis.draw_trackers(tracker_dic[index].tracks,index)
        # Store results.
        # for i in range(camera_num):
        #     for track in tracker_dic[i].tracks:
        #         if not track.is_confirmed() or track.time_since_update > 1:
        #             continue
        #         bbox = track.to_tlwh()
        #         results.append([
        #             frame_idx, track.global_id, bbox[0], bbox[1], bbox[2], bbox[3]])


    def frame_callback(vis, frame_idx,index):
        print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(
            seq_info_dic[index]["detections"], frame_idx,index, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        # print ("the camera_index is {}, the coordiantes of detections is {}".format(index,[detection.tlwh for detection in detections]))
        # Update tracker.
        tracker_dic[index].predict()
        tracker_dic[index].update(detections,tracker_dic,global_id,global_track)

        # Update visualization.
        if display:
            if index == 0:
                vis.reset_image()
            image = cv2.imread(
                seq_info_dic[index]["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.append_image(image.copy())
            vis.draw_trackers(tracker_dic[index].tracks,index)
        # Store results.
        # for i in range(camera_num):
        #     for track in tracker_dic[i].tracks:
        #         if not track.is_confirmed() or track.time_since_update > 1:
        #             continue
        #         bbox = track.to_tlwh()
        #         results.append([
        #             frame_idx, track.global_id, bbox[0], bbox[1], bbox[2], bbox[3]])
    last_frame = {}
    max_frame_idx = 0
    for i in range(camera_num):
            last_frame[i] = seq_info_dic[i]["max_frame_idx"]
            if last_frame[i] > max_frame_idx:
                max_frame_idx = last_frame[i]
    if video_dir == None:
        visualizer = visualization.Visualization(seq_info_dic,global_id, camera_num,5,last_frame)
        visualizer.run(frame_callback,tracker_dic,camera_num,max_frame_idx)
    else:
        # print ("seq_info_dic is {}".format(seq_info_dic))
        visualizer = visualization.Visualization(seq_info_dic,global_id, camera_num,5,last_frame,video_dir_list)
        visualizer.run(video_frame_processing,tracker_dic,camera_num,max_frame_idx)
    # Store results.
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None, required=True)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=100, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.3)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool)
    parser.add_argument(
        "--video_dir", help="Path to camera video.", default=None)
    parser.add_argument(
        "--max_age", help="Maximum time interval for a pedestrian disappears.", default=500,type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display,args.video_dir, args.max_age)
