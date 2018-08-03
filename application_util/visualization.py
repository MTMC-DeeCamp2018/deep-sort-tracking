# vim: expandtab:ts=4:sw=4
import numpy as np
import colorsys
from .image_viewer import ImageViewer


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

    def __init__(self, seq_info_list, global_id,camera_num,update_ms):
        seq_info = seq_info_list[0]
        image_shape = seq_info["image_size"][::-1]
        aspect_ratio = float(image_shape[1]) / image_shape[0]
        image_shape = 1024, int(aspect_ratio * 1024)
        self.viewer = ImageViewer(
            update_ms, image_shape, "Figure %s" % seq_info["sequence_name"],camera_num)
        self.viewer.thickness = 2
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]
        self.global_id = global_id

    def run(self,frame_callback,tracker_dic,camera_num):
        self.viewer.run(self,tracker_dic,lambda: self._update_fun(frame_callback,camera_num))

    def _update_fun(self, frame_callback,camera_num):
        if self.frame_idx > self.last_idx:
            return False  # Terminate
        for i in range(camera_num):
            frame_callback(self, self.frame_idx,i,self.global_id)
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

    def draw_trackers(self, tracks,index):
        self.viewer.thickness = 2
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            self.viewer.color = create_unique_color_uchar(track.track_id)
            self.viewer.rectangle(
                *track.to_tlwh().astype(np.int), index,label=str(track.track_id))
            # if index > 0:
            #     self.viewer.rectangle(
            #     track.to_tlwh().astype(np.int)[0]+10,track.to_tlwh().astype(np.int)[1]+10,track.to_tlwh().astype(np.int)[2], track.to_tlwh().astype(np.int)[3],index,label=str(track.track_id))
            # print(*track.to_tlwh().astype(np.int))
            # for trace in track.trace:
            #     # print (trace[0])
            #     # print (trace[1])
            #     self.viewer.circle(trace[0],trace[1],2)
            # self.viewer.colored_points(track.trace)
            # self.viewer.gaussian(track.mean[:2], track.covariance[:2, :2],
            #                      label="%d" % track.track_id)
#           
