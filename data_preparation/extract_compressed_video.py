"""
Extracts images from (compressed) videos, used for the FaceForensics++ dataset

Usage: see -h or https://github.com/ondyari/FaceForensics

Author: Andreas Roessler
Date: 25.01.2019
"""
import os
import cv2
from facenet_pytorch.models.mtcnn import MTCNN
import numpy as np
from os.path import join
import argparse
import subprocess
from tqdm import tqdm
import face_mtcnn



DATASET_PATHS = {
    'original': 'original_sequences/youtube',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'FaceShifter': 'manipulated_sequences/FaceShifter',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures'
}
DEEPFEAKES_DETECTION_DATASET = {
    'DeepFakeDetection_original': 'original_sequences/actors',
    'DeepFakeDetection': 'manipulated_sequences/DeepFakeDetection',
}
COMPRESSION = ['c0', 'c23', 'c40']


def get_saving_frames_durations(cap, saving_fps, fps):
    """A function that returns the list of durations where to save the frames"""
    s = []
    # get the clip duration by dividing number of frames by the number of frames per second
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    # use np.arange() to make floating-point steps
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s

def extract_frames(data_path, output_path, SAVING_FRAMES_PER_SECOND = 5, method='cv2'):
    """Method to extract frames, either with ffmpeg or opencv. FFmpeg won't
    start from 0 so we would have to rename if we want to keep the filenames
    coherent."""
    os.makedirs(output_path, exist_ok=True)
    face_detector = face_mtcnn.FaceDetector(image_size=320)

    if method == 'ffmpeg':
        subprocess.check_output(
            'ffmpeg -i {} {}'.format(
                data_path, join(output_path, '%04d.png')),
            shell=True, stderr=subprocess.STDOUT)

    elif method == 'cv2':
        reader = cv2.VideoCapture(data_path)
        # get the FPS of the video
        fps = reader.get(cv2.CAP_PROP_FPS)
        # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
        saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
        # get the list of duration spots to save
        saving_frames_durations = get_saving_frames_durations(reader, saving_frames_per_second, fps)
        frame_num = 0
        while reader.isOpened():
            success, image = reader.read()
            if not success:
                break
            try:
                # get the earliest duration to save
                closest_duration = saving_frames_durations[0]
            except IndexError:
                # the list is empty, all duration frames were saved
                break
            frame_duration = frame_num / fps
            if frame_duration >= closest_duration:
                face_list = face_mtcnn.detect(face_detector, image)
                if len(face_list) > 0:
                    for face_idx, face in enumerate(face_list):
                        name = join(output_path, '{:04d}_face_{:02d}.png'.format(frame_num, face_idx))
                        cv2.imwrite(name, face)
                try:
                    # drop the duration spot from the list, 
                    # since this duration spot is already saved
                    saving_frames_durations.pop(0)
                except IndexError:
                    pass
            frame_num += 1
        reader.release()
    else:
        raise Exception('Wrong extract frames method: {}'.format(method))


def extract_method_videos(input_path, output_path, dataset, compression, frames_per_second):
    """Extracts all videos of a specified method and compression in the
    FaceForensics++ file structure"""
    videos_path = join(input_path, DATASET_PATHS[dataset], compression, 'videos')
    images_path = join(output_path, DATASET_PATHS[dataset], compression, 'images')
    face_detector = face_mtcnn.FaceDetector(image_size=320)
    for video in tqdm(os.listdir(videos_path)):
        image_folder = video.split('.')[0]
        extract_frames(join(videos_path, video),
                       join(images_path, image_folder), frames_per_second)


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input_path', '-i',type=str)
    p.add_argument('--output_path', '-o',type=str)
    p.add_argument('--dataset', '-d', type=str,
                   choices=list(DATASET_PATHS.keys()) + ['all'],
                   default='all')
    p.add_argument('--compression', '-c', type=str, choices=COMPRESSION,
                   default='c23')
    p.add_argument('--frames_per_second', '-fps',
                   type=float, default = 1)
    args = p.parse_args()

    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    if args.dataset == 'all':
        for dataset in DATASET_PATHS.keys():
            args.dataset = dataset
            extract_method_videos(**vars(args))
    elif args.dataset == 'deepfake_detection':
        for dataset in DEEPFEAKES_DETECTION_DATASET.keys():
            args.dataset = dataset
            extract_method_videos(**vars(args))
    else:
        extract_method_videos(**vars(args))