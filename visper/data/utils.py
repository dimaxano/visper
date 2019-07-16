import sys

import cv2
import numpy as np
import torchvision.transforms.functional as functional
import torchvision.transforms as transforms
import torch
from torch.nn import functional as F
import gin

from .transforms.statefultransforms import StatefulRandomCrop, StatefulRandomHorizontalFlip, GaussianSmoothing

def get_video_frames(path, num_channels=3):
    """
        Return list of frames (numpy arrays)
    """
    reader = cv2.VideoCapture(path)

    frames = []
    while True:
        ret, frame = reader.read()
        if not ret:
            break
        else:
            if num_channels == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
            else:
                frames.append(frame)

    return frames


def shorten2length(video, lenght=29):
    current_lenght = len(video)

    diff = current_lenght - lenght

    frame_counter = 0
    diff_count = 0
    while diff_count < diff:
        if frame_counter >= len(video):
            frame_counter = 0

        if frame_counter % 2 == 0:
            video.pop(frame_counter)
            diff_count += 1
        
        frame_counter += 1


    return video


def extend2length(video, lenght=29):
    """
        Extends video to be of `lenght`

        Args:
            `video`:list of frames (numpy arrays)
    """

    def_len = len(video)

    
    diff = lenght - def_len

    indexes = {a: a for a in range(def_len)}

    j = 0
    # = len(video) - 1
    for _ in range(diff):
        if j == def_len:
            j = 0

        try:
            current_frame = video[indexes[j]].copy()
            video.insert(indexes[j] + 1, current_frame)
        except:
            print("broken video")
            return None

        # after each insert, we should update indexes
        for k in indexes:
            if k != 0 and k > j:
                indexes[k] += 1

        j += 1

    return video


def read_video(filename, video_len=29):
    """
    Reads video. If video length != `video_len`, video will extended or shortened to video_len

    Args:
        filename (str): The path to the file to load.
            Should be a format that ffmpeg can handle.

    Returns:
        List[FloatTensor]: the frames of the video as a list of 3D tensors
            (channels, width, height)"""

    images = get_video_frames(filename, num_channels=3)

    if len(images) > video_len:
        images = shorten2length(images, lenght=29)
    elif len(images) < video_len:
        images = extend2length(images, lenght=29)
    
    if not images:
        return None

    frames = [functional.to_tensor(image) for image in images]
    return frames

@gin.configurable
def preprocess(
    vidframes,
    augmentation=False,
    augmentations_list = [],
    src_size=(112, 112),
    dst_size=(88, 88),
    video_len=29):
    """
    Preprocess input video

    Args:
        vidframes (List[FloatTensor]):  The frames of the video as a list of
            3D tensors (channels, width, height)
        augmentation (Bool): whether to perform augmentations or not
        augmentations_list: [List[torchvision.Transforms]] - list of augmentations to be apllied
        src_size (Tuple): w, h of input video
        dst_size (Tuple): w, h of video after preprocessing (size of model input)
        video_len (Int): # frames in input video

    Returns:
        FloatTensor: preprocessed video as a temporal volume, represented as a 5D tensor
            (batch, channel, time, width, height)"""

    src_w, src_h = src_size
    dst_w, dst_h = dst_size


    temporalvolume = torch.FloatTensor(1, video_len, dst_h, dst_w)

    croptransform = transforms.CenterCrop((dst_h, dst_w))

    if augmentation:
        croptransform = transforms.Compose(augmentations_list)

    for i in range(0, 29):
        result = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Resize(112), # if comes images of size 100,100 - resize operation won't be performed
            croptransform,
            transforms.Grayscale(num_output_channels=1),
        ])(vidframes[i])

        numpy_arr = np.array(result)
        torch_tensor = torch.from_numpy(numpy_arr)


        temporalvolume[0][i] = torch_tensor #torch_tensor

    return temporalvolume
