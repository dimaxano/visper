from pathlib import Path
from typing import Tuple

import dlib
import cv2
import numpy as np

from .matlab_cp2tform import get_similarity_transform_for_cv2


REFERENCE_POINTS = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ] # for image of size 96x112

def make_video(frames, save_path, w, h, is_color, fps=25):
    """
        :param frames: list of numpy arrays
    """

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path, fourcc, fps, (w,h), is_color)

    for i, frame in enumerate(frames):
        #print(frame)
        out.write(frame)
    
    out.release()


def alignment_orig(src_img, src_pts, ncols=96, nrows=112):
    """
    Original alignment function for MTCNN
    :param src_img: input image
    :param src_pts: landmark points
    :return:
    """
    
    REFERENCE_POINTS = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
    ref_pts = REFERENCE_POINTS
    if nrows == 128 and ncols == 128:
        for row in ref_pts:
            row[1] += 16.0
            row[0] += 16.0
    
    if ncols == 112:
        for row in ref_pts:
            row[0] += 10.0

    crop_size = (ncols, nrows)
    src_pts = np.array(src_pts).reshape(5, 2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img


def plot_points(img, points, color=(0,0,255), thickness=4, show_image=True):
    """
        Args:
            img - image where point will be displayed
            points[numpy.ndarray] - array of points of size Nx2 (N - number of points)
            show_image[Bool] -      If True, image will be displayed (with OpenCV), else image with points will be returned
    """

    img = img.copy()

    for point in points:
        point = tuple(map(int, point))
        cv2.line(img, point, point, (0,0,255), 4)
    
    if show_image:
        cv2.imshow("image", img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        return img


def dlib2mtcnn_lm(lm_coordinates):
    """
        Transform 68 dlib landmarks to 5 mtcnn-format landmarks
    """

    def get_center_point(point1, point2):
        """
            Args:
                point1[list] - x,y coordinates
                point2[list] - x,y coordinates
        """
        x1, y1 = point1
        x2, y2 = point2

        center_x = x2 + (x1 - x2) / 2
        center_x = int(center_x)

        center_y = y2 + (y1 - y2) / 2
        center_y = int(center_y)

        return (center_x, center_y)


    left_eye_center = get_center_point(lm_coordinates[37], lm_coordinates[40])
    right_eye_center = get_center_point(lm_coordinates[43], lm_coordinates[46])
    nose = lm_coordinates[30]
    left_mouth = lm_coordinates[48]
    right_mouth = lm_coordinates[54]

    return [left_eye_center, right_eye_center, nose, left_mouth, right_mouth]


def img_lm_transform(image, points, dest_size=(112, 96)):
    """
        Resize image to dest_size and transforms corresponding image keypoints

        Return
            resized face, transformed keypoints
    """

    dest_rows, dest_cols = dest_size
    src_rows, src_cols, _ = image.shape
    delta_rows, delta_cols = src_rows / dest_rows, src_cols / dest_cols

    # transform image
    resized_image = cv2.resize(image, (dest_cols, dest_rows), interpolation=cv2.INTER_CUBIC)

    # transfrom points
    transformed_points = []
    for point in points:
        src_x, src_y = point
        dest_x, dest_y = src_x//delta_cols, src_y//delta_rows

        transformed_points.append((dest_x, dest_y))
    
    return resized_image, transformed_points
    

def align_face(image, detector, predictor, crop_size: Tuple):
    """
        Args:
            crop_size [Tuple] - h,w
    """
    frame = image

    dets = detector(frame, 1)
    if not dets:
        print("face detector failed")
        return None
    elif len(dets) > 1:
        print("More than two faces detected")
        return None

    d = dets[0]
    shape = predictor(frame, d)

    facial_points = []
    for part in shape.parts():
        facial_points.append((part.x, part.y))

    mtcnn_lm = dlib2mtcnn_lm(facial_points)
    
    h, w = crop_size
    align_face = alignment_orig(frame, mtcnn_lm, nrows=h, ncols=w)

    return align_face


def align_video(frames, detector, predictor, crop_size: Tuple):
    """
        Args:
            frames[List] - list of ndarrays
            crop_size [Tuple] - h,w
    """

    aligned_frames = []
    for frame in frames:
        align_frame = align_face(frame, detector, predictor, crop_size)
        if align_frame is None:
            return None

        aligned_frames.append(align_frame)
    
    return aligned_frames


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


if __name__ == "__main__":
    dataset_path = Path("/home/dmitry.klimenkov/Documents/datasets/lipreading_datasets/FACE/lrs2lrw_frontal")
    predictor_path = "/home/dmitry.klimenkov/Documents/projects/3DResNet/LM/shape_predictor_68_face_landmarks.dat"

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    k = 0
    for video_path in dataset_path.glob("*/*.mp4"):
        frames = get_video_frames(str(video_path))
        aligned_frames = align_video(frames, False, detector, predictor, 112, 112)

        if aligned_frames:
            make_video(aligned_frames, f"/home/dmitry.klimenkov/Documents/projects/3DResNet/contrib/face_alignment_research/files/aligned_video_{k}.mp4", 100, 100, True)
            if k == 5:
                break
            k+=1
    
        