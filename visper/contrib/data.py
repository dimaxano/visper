"""
Here you'll find set of (very supplementary) methods 
for different operations on data, especially on lipreading datasets:
    MV-LRS, LRW, LRS3-TED, Grid Corpus(:but_why:)

"""

import subprocess
from os.path import isfile
import os
from pathlib import Path
import time

import cv2

from ..data.utils import get_video_frames
from .face_alignment import align_video, make_video
import dlib


def clear_text_lrs2(text, strict=False):
  """
      Takes all text from file and returns only label text

      Args:
        text[str] - raw text from label-file
        strict - if True text labels with digits and '-symbol will be removed
  """
  def hasNumbers(text):
    return any(char.isdigit() for char in text)

  first_line = text.split("\n")[0]
  old_label = first_line.split(":")[1].strip().lower()

  if strict and (hasNumbers(old_label) or "'" in old_label):
      return None
  else:
      return old_label


def get_words_with_timesteps(text, idx):
  lines = text.split("\n")
  
  words = []
  if lines[3].startswith("WORD"):
    lines = lines[4:]
    for i in idx:
      word_and_ts = lines[i].split(" ")[:-1]
      words.append(word_and_ts)

    return words
  else:
    return None


def words_in_sentence(words, sentence):
  """
    Return idx for `words` in sentence

    Args:
      words[list] - list of lowercase words
      
  """
  sentence_words = sentence.split(" ")
  words = [word.lower() for word in words]

  idx = []
  for word in words:
    try:
      i = sentence_words.index(word)
      idx.append(i)
    except:
      pass
  
  return idx


def clip(src_video_path, start, end, dst_video_path):
    """
        Clip givem `src_video_path` according given `start`, `end` timesteps.
        Clipped video saved at `dst_video_path`
    """
    subprocess.run(["ffmpeg",  #Calls ffmpeg program
        "-ss",str(start),        #Begining of recording, must be string
        "-i", src_video_path,    #Inputs command line argument 1
        "-t", str(end),
        "-c", "copy",
        "-loglevel", "quiet",
        "-y",
        dst_video_path])


def mvlrs2words(
    dataset_path: Path,
    save_dir: Path,
    words,
    word_video_length=1.16,
    shape_predictor_path=None):
    """
        Make sentence level mv-lrs dataset (pretrain part)
        to be a word level dataset.
        
        Args:
            dataset_path [Path] - path to the mv-lrs/pretrained
            save_dir [Path] - new word level dataset path
            words [List[String]] - what words include in dataset.
            word_video_length [Float] - in seconds
            save_lips [Bool] - If True, also lips video will be saved. 
                BE AWARE: save_dir for saving lip dataset will be created automatically
            shape_predictor_path [String] - 
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)

    annotations_files = dataset_path.glob("*/*.txt")
    for i, annotation_file in enumerate(annotations_files):
        print(f"{i}", end="\r")
        raw_text = open(str(annotation_file), "r").read()

        res = clear_text_lrs2(raw_text)
        if res:
            idx = words_in_sentence(words, res)

            if idx:
                words_with_ts = get_words_with_timesteps(raw_text, idx)
                
                for word_with_ts in words_with_ts:
                    word, start, end = word_with_ts
                    start, end = float(start), float(end)

                    video_len = end - start
                    if video_len < word_video_length:
                        start -= (word_video_length - video_len)/2
                        end += (word_video_length - video_len)/2

                    if start < 0:
                        continue
                    
                    word_save_dir = save_dir / word
                    word_save_dir.mkdir(parents=True, exist_ok=True)

                    video_save_path = word_save_dir / f"{i}.mp4"
                    mp4_filename = annotation_file.parent / (annotation_file.stem + ".mp4")
                    clip(str(mp4_filename), start, word_video_length, str(video_save_path))

                    frames = get_video_frames(str(video_save_path))
                    aligned_frames = align_video(
                        frames,
                        detector,
                        predictor,
                        (112, 112))

                    if aligned_frames is None:
                        video_save_path.unlink() # remove clip if video couldn't be aligned correctly
                    else:
                        make_video(aligned_frames, str(video_save_path), 112, 112, True)


if __name__ == "__main__":
    mvlrs_pretrained = Path(("/home/dmitry.klimenkov/Documents/datasets"
                        "/lipreading_datasets/FACE/mvlrs_v1/pretrain"))
    words = ["SOMETHING", "YEARS", "GREAT", "QUITE", "NEVER", \
            "TODAY", "ALWAYS", "BEFORE", "MIGHT", "HOUSE", \
            "PLACE", "WORLD", "THREE", "PROBABLY", "ANYTHING" \
            "FAMILY", "WOULD", "WHERE"]
    save_dir = Path(("/home/dmitry.klimenkov/Documents/datasets"
                "/lipreading_datasets/FACE/LRS2LRW"))
    shape_predictor_path = ("/home/dmitry.klimenkov/Documents/misc/"
                "dlib_predictor/shape_predictor_68_face_landmarks.dat")

    mvlrs2words(
        dataset_path=mvlrs_pretrained,
        save_dir=save_dir,
        words=words,
        shape_predictor_path=shape_predictor_path
        )
    

