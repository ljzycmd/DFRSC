import os
import platform
import logging

import torch
import numpy as np
import math
import cv2
import copy
import random

from easydict import EasyDict as edit

from .augment import augment, distortion_map


from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class FastecRS(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.opt = copy.deepcopy(cfg)

        cfg = edit(cfg)
        self.cfg = cfg

        self.video_list = os.listdir(self.cfg.root_gt)
        self.video_list.sort()

        self.frames = []
        self.video_frame_dict = {}
        self.video_length_dict = {}

        for video_name in self.video_list:
            # Warning! change the video path in different format of video deblurring dataset
            video_path = os.path.join(self.cfg.root_gt, video_name)
            frames_in_video = list(filter(lambda x: True if x.endswith("rolling.png") else False, os.listdir(video_path)))
            frames_in_video.sort()

            frames_in_video = [os.path.join(video_name, frame) for frame in frames_in_video]

            self.frames += self.frame_sampling(
                frames_in_video, cfg.num_frames, cfg.interval, cfg.sampling, cfg.overlapping)

            self.video_frame_dict[video_name] = frames_in_video
            self.video_length_dict[video_name] = len(frames_in_video)

        assert self.frames, f"There are no frames in '{self.cfg.root_gt}'. "

        logging.info(
            f"Total samples {len(self.frames)} are loaded for {self.cfg.mode}!")

    def frame_sampling(self, frames_in_video, num_frames, interval, sampling="n_c", overlapping=True):
        """
        Args:
            frames_in_video: a frames list extract from one video.
            num_frames: input frames of the model.
            interval: the interval when sampling.
            sampling: the sampling mode.
            overlapping: a flag that decide overlapping sampling.
        """
        # sample length with inerval
        sampled_frames_length = (num_frames - 1) * interval + 1
        if sampling == "n_n" or sampling == "n_l":
            # non-overlapping sampling
            if overlapping:
                # avoid  1 - sampled_frames_length = 0, transfer it to positive index
                return frames_in_video[:len(frames_in_video) - sampled_frames_length + 1]
            else:
                # ensure the sampling frame can be sampled!
                return frames_in_video[:len(frames_in_video) - sampled_frames_length + 1:sampled_frames_length]

        elif sampling == "n_c":
            if overlapping:
                return frames_in_video[sampled_frames_length // 2: len(frames_in_video) - (sampled_frames_length // 2)]
            else:
                return frames_in_video[sampled_frames_length // 2: len(frames_in_video) - (sampled_frames_length // 2): sampled_frames_length]

        elif sampling == "n_r":
            if overlapping:
                return frames_in_video[sampled_frames_length - 1:]
            else:
                return frames_in_video[sampled_frames_length - 1::sampled_frames_length]

        # you can add some other sampling mode here.
        else:
            print("none sampling mode '{}' ".format(sampling))
            raise NotImplementedError

    def get_frames_name(self, frame_name, num_frames, interval, sampling="n_c"):
        """
        Args:
            frame_name: the frame's name corresponding to the idx.
            num_frames: the number of input frames of the model.
            interval: the interval when sampling.
            sampling: the sampling mode.
        """
        frame_idx = frame_name[:3]
        frame_idx_length = len(frame_idx)
        frame_idx = int(frame_idx)

        gt_frame_name_format = "{:0" + str(frame_idx_length) + "d}" + "_global_" + self.cfg.pos + ".png"
        input_frame_name_format = "{:0" + str(frame_idx_length) + "d}" + "_rolling.png"

        gt_frames_name = [gt_frame_name_format.format(frame_idx)]
        input_frames_name = []
        # when to read the frames, should pay attention to the name of frames
        if sampling == "n_c":
            input_frames_name = [input_frame_name_format.format(i) for i in range(
                frame_idx - (num_frames // 2) * interval, frame_idx + (num_frames // 2) * interval + 1, interval)]

        elif sampling == "n_n" or sampling == "n_l":
            input_frames_name = [input_frame_name_format.format(i) for i in range(
                frame_idx, frame_idx + interval * num_frames, interval)]
            if sampling == "n_n":
                gt_frames_name = [gt_frame_name_format.format(i) for i in range(
                    frame_idx, frame_idx + interval * num_frames, interval)]

        elif sampling == "n_r":
            input_frames_name = [input_frame_name_format.format(i) for i in range(
                frame_idx - num_frames * interval + 1, frame_idx + 1, interval)]

        else:
            raise NotImplementedError

        return gt_frames_name, input_frames_name

    def read_img_opencv(self, path):
        """
        read image by opencv
        return: Numpy float32, HWC, BGR, [0,1]
        """
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print("the path is None! {} !".format(path))
        img = img.astype(np.float32) / 255.
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        # some images have 4 channels
        if img.shape[2] > 3:
            img = img[:, :, :3]
        return img

    def __getitem__(self, idx):
        if platform.system() == "Windows":
            video_name, frame_name = self.frames[idx].split("\\")
        else:
            video_name, frame_name = self.frames[idx].split("/")

        video_length = self.video_length_dict[video_name]

        gt_frames_name, input_frames_name = self.get_frames_name(
            frame_name, self.cfg.num_frames, self.cfg.interval, self.cfg.sampling)

        assert len(input_frames_name) == self.cfg.num_frames, "Wrong frames length not equal the sampling frames {}".format(
            self.cfg.num_frames)

        rs_gs_consistency = False
        if self.cfg.get("rs_gs_consistency"):
            if random.random() < 0.5:
                rs_gs_consistency = True

        if rs_gs_consistency:
            # exchange the gt_frames_name and input_frames_name
            gs_suffix = "_global_" + self.cfg.pos + ".png"
            rs_suffix = "_rolling.png"
            gt_frames_name = [frame_name.replace(gs_suffix, rs_suffix) for frame_name in gt_frames_name]
            input_frames_name = [frame_name.replace(rs_suffix, gs_suffix) for frame_name in input_frames_name]

        gt_frames_path = os.path.join(self.cfg.root_gt, video_name, "{}")
        input_frames_path = os.path.join(self.cfg.root_gt, video_name, "{}")

        # Read images by opencv with format HWC, BGR, [0,1], TODO add other loading methods.
        gt_frames = [self.read_img_opencv(gt_frames_path.format(
            frame_name)) for frame_name in gt_frames_name]
        input_frames = [self.read_img_opencv(input_frames_path.format(
            frame_name)) for frame_name in input_frames_name]

        # stack and transpose with RGB style (n, c, h, w)
        gt_frames = np.stack(
            gt_frames, axis=0)[..., ::-1].transpose([0, 3, 1, 2])
        input_frames = np.stack(
            input_frames, axis=0)[..., ::-1].transpose([0, 3, 1, 2])

        if rs_gs_consistency:
            input_frames = input_frames[:, :, ::-1, :]
            gt_frames = gt_frames[:, :, ::-1, :]

        if self.cfg.get("time_map"):
            # generating distortion map (time map)
            H, W = input_frames.shape[-2:]
            ref_row = (H - 1) / 2 if self.cfg.pos == "middle" else 0
            encoding_map = distortion_map(H, W, ref_row=ref_row).numpy().repeat(self.cfg.num_frames, axis=0)
            # if rs_gs_consistency:
            #     encoding_map *= -1.
            input_frames = np.concatenate([input_frames, encoding_map], axis=1)  # (n, 4, h, w)

        # augmentaion while training...
        if self.cfg.mode == "train" and hasattr(self.cfg, "augmentation"):
            input_frames, gt_frames = augment(
                input_frames, gt_frames, self.cfg.augmentation)

        # To tensor with contingious array.
        gt_frames = torch.from_numpy(np.ascontiguousarray(gt_frames)).float()
        input_frames = torch.from_numpy(
            np.ascontiguousarray(input_frames)).float()

        return {
            "input_frames": input_frames,
            "gt_frames": gt_frames,
            "video_name": video_name,
            "video_length": video_length,
            "gt_names": gt_frames_name,
        }

    def __len__(self):
        return len(self.frames)
