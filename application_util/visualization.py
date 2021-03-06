# vim: expandtab:ts=4:sw=4
import numpy as np
import colorsys
import cv2
from .image_viewer import ImageViewer
from . import epfl_calib

def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)


class NoVisualization(object):
    """
    A dummy visualization object that loops through all frames in a given
    sequence to update the tracker without performing any visualization.
    """

    def __init__(self, seq_info):
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]

    def set_image(self, image):
        pass

    def draw_groundtruth(self, track_ids, boxes):
        pass

    def draw_detections(self, detections):
        pass

    def draw_trackers(self, trackers):
        pass

    def run(self, frame_callback):
        while self.frame_idx <= self.last_idx:
            frame_callback(self, self.frame_idx)
            self.frame_idx += 1


class Visualization(object):
    """
    This class shows tracking output in an OpenCV image viewer.
    """

    def __init__(self, seq_info_list, global_id,camera_num,update_ms, last_frame,video_dir_list = None):
        # print ("seq_info_keys is {}".format(seq_info_list.keys()))
        seq_info = seq_info_list[0]
        if video_dir_list == None:
            image_shape = seq_info["image_size"][::-1]
            aspect_ratio = float(image_shape[1]) / image_shape[0]
            image_shape = 1024, int(aspect_ratio * 1024)
        if video_dir_list:
            self.viewer = ImageViewer(update_ms, camera_num = camera_num)
        else:
            self.viewer = ImageViewer(
                update_ms, image_shape, "Figure %s" % seq_info["sequence_name"],camera_num)
        self.viewer.thickness = 2
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = last_frame
        self.global_id = global_id
        self.video_dir_list = video_dir_list

    def run(self,frame_callback,tracker_dic,camera_num, max_frame_idx = None):
        if self.video_dir_list:
            frame_idx = 1
            video_player = []
            for i in range(camera_num):
                video_player.append(cv2.VideoCapture(self.video_dir_list[i]))
            while frame_idx <= max_frame_idx:
                for i in range(camera_num):
                    if frame_idx > self.last_idx[i]:
                        continue
                    (grabbed, frame) = video_player[i].read()
                    image = frame.copy()
                    frame_callback(self,frame_idx,i,image)
                self.viewer.run(self,tracker_dic)
                frame_idx += 1
        else:
            self.viewer.run(self,tracker_dic,lambda: self._update_fun(frame_callback,camera_num))

    def _update_fun(self, frame_callback,camera_num):
        # if self.frame_idx > self.last_idx:
        #     return False  # Terminate
        for i in range(camera_num):
            if self.frame_idx > self.last_idx[i]:
                continue
            # frame_callback(self, self.frame_idx,i)
            if i == 1:
                frame_callback(self, self.frame_idx+100,i)
                # if self.frame_idx > 100:
                #     frame_callback(self, self.frame_idx-100,i,self.global_id)
            else:
                frame_callback(self, self.frame_idx,i)
        self.frame_idx += 1
        return True

    def set_image(self, image):
        self.viewer.image = image

    def reset_image(self):
        self.viewer.image = []

    def append_image(self,image):
        self.viewer.image.append(image)

    def draw_groundtruth(self, track_ids, boxes,index):
        self.viewer.thickness = 2
        for track_id, box in zip(track_ids, boxes):
            self.viewer.color = create_unique_color_uchar(track_id)
            self.viewer.rectangle(*box.astype(np.int), index,label=str(track_id))

    def draw_detections(self, detections,index):
        self.viewer.thickness = 2
        self.viewer.color = 0, 0, 255
        for i, detection in enumerate(detections):
            self.viewer.rectangle(*detection.tlwh,index)
            # print(*detection.tlwh) 

    def draw_trackers(self, tracks,index,frame_idx,file=None):
        self.viewer.thickness = 2
        for track in tracks:
            # print ("the feature of current track is {}".format(track.features))
            # print ("the coordinates of current track is {}".format(track.to_tlwh()))
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            if len(track.trace) > 10:
                track.trace.pop(0)
            bbox = track.to_tlwh()
            coordinate = [bbox[0]+bbox[2]/2, bbox[1]+bbox[3]]
            world_coordinate = np.array(coordinate, dtype=np.float32)
            world_coordinate = epfl_calib.img_to_world(world_coordinate, epfl_calib.terrace_H()[index])
            if file:
                line = ("{} {} {} {}\n".format(frame_idx,track.global_id,world_coordinate[0],world_coordinate[1]))
                line = line.replace("[","")
                line = line.replace("]","")
                file.write(line)
            # if len(track.trace) >= 1:
                # world_coordinate_prev = np.array(track.trace[-1], dtype=np.float32)
                # world_coordinate_prev = epfl_calib.img_to_world(world_coordinate_prev, epfl_calib.terrace_H()[index])

                # world_coordinate_prev = track.trace[-1]
                # squared_distance = (world_coordinate[0]-world_coordinate_prev[0]) ** 2 + (world_coordinate[1]-world_coordinate_prev[1]) ** 2
                # print ("the coordinate diff is {}".format(squared_distance))

            # print ("camera_index is {}, global_track_id is {}, world_coordinate is {}".format(index,track.global_id,world_coordinate))
            # track.trace.append(coordinate)
            track.trace.append(world_coordinate)
            # if len(track.features) > 0:
            #     a = np.asarray(track.features[-1]) / np.linalg.norm(track.features[-1], axis=1, keepdims=True)
            #     b = np.asarray(track.features[-2]) / np.linalg.norm(track.features[-2], axis=1, keepdims=True)
            # print("the feature diff for the same person is {}".format(1. - np.dot(a, b.T)))
            self.viewer.color = create_unique_color_uchar(track.global_id)
            # for trace in track.trace:
                # self.viewer.circle(index, trace[0], trace[1], 2)
            self.viewer.rectangle(
                *track.to_tlwh().astype(np.int), index,label=str(track.global_id))
        for i, track in enumerate(tracks):
            if len(track.trace) > 0:
                print ("camera index is {}, global_id is {}, track is {}".format(index,i,track.trace[-1]))


#           
