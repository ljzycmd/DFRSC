# flake8: noqa
import os.path as osp
from basicsr.train import train_pipeline

from dataset import bsrsc
from dataset import fastec_rs
from dataset import carla_rs

from model import dfrsc, dfrsc_model


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    train_pipeline(root_path)
