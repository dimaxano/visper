import sys

import cv2
import numpy as np
import torchvision.transforms.functional as functional
import torchvision.transforms as transforms
import torch
from torch.nn import functional as F

from .statefultransforms import StatefulRandomCrop, StatefulRandomHorizontalFlip, GaussianSmoothing


def get_video_frames(path, num_channels=3):
    """
        returns list of RGB, jpg-encoded images
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
        Args:
            `video`:list of frames
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


def load_video(filename):
    """Loads the specified video using ffmpeg.

    Args:
        filename (str): The path to the file to load.
            Should be a format that ffmpeg can handle.

    Returns:
        List[FloatTensor]: the frames of the video as a list of 3D tensors
            (channels, width, height)"""

    images = get_video_frames(filename, num_channels=3)

    if len(images) > 29:
        images = shorten2length(images, lenght=29)
    elif len(images) < 29:
        images = extend2length(images, lenght=29)
    
    if not images:
        return None

    frames = []
    for image in images:
        image = functional.to_tensor(image)
        frames.append(image)
    return frames


def bbc(vidframes, augmentation=True):
    """Preprocesses the specified list of frames by center cropping.
    This will only work correctly on videos that are already centered on the
    mouth region, such as LRITW.

    Args:
        vidframes (List[FloatTensor]):  The frames of the video as a list of
            3D tensors (channels, width, height)

    Returns:
        FloatTensor: The video as a temporal volume, represented as a 5D tensor
            (batch, channel, time, width, height)"""

    temporalvolume = torch.FloatTensor(1,29,88,88)

    croptransform = transforms.CenterCrop((88, 88))

    if augmentation:
        crop = StatefulRandomCrop((112,112), (88, 88))
        flip = StatefulRandomHorizontalFlip(0.5)

        croptransform = transforms.Compose([
            crop,
            flip
        ])

    for i in range(0, 29):
        """ inp = vidframes[i].unsqueeze(0)
        inp = F.pad(inp, (2, 2, 2, 2), mode='reflect')
        vidframes[i] = smoothing(inp)[0] """

        result = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(112), # if comes images of size 100,100 - resize operation won't be performed
            croptransform,
            transforms.Grayscale(num_output_channels=1),
            #transforms.ToTensor(),
            #transforms.Normalize([0.4161,], [0.1688,]),
        ])(vidframes[i])

        numpy_arr = np.array(result)
        torch_tensor = torch.from_numpy(numpy_arr)


        temporalvolume[0][i] = torch_tensor #torch_tensor

    return temporalvolume
