# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import os
import glob
from typing import List, Tuple, Union

import cv2
import numpy as np
from scipy.io import loadmat

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager


__all__ = ["load_davis_instances", "register_davis_context"]

def load_davis_instances(name: str, dirname: str, mode: str, split: str):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    image_pths = sorted(glob.glob(os.path.join(dirname, "img", "*.jpg")))
    mask_pths = sorted(glob.glob(os.path.join(dirname, "gt", "*.png")))
    assert len(image_pths) == len(mask_pths)

    dicts = []
    for image_pth, mask_pth in zip(image_pths, mask_pths):
        r = {
            "file_name": image_pth,
            "mask_name": mask_pth,
        }
        dicts.append(r)
    return dicts

def register_davis_context(name, dirname, mode, split):
    DatasetCatalog.register("{}_{}".format(name, mode), lambda: load_davis_instances(name, dirname, mode, split))
    MetadataCatalog.get("{}_{}".format(name, mode)).set(
        dirname=dirname,
        thing_dataset_id_to_contiguous_id={},
    )

def register_all_davis(root):
    SPLITS = [
            ("openimage600_val", "open-image600", "Point", "val"),
            ("openimage600_val", "open-image600", "Scribble", "val"),
            ("openimage600_val", "open-image600", "Polygon", "val"),
            ("openimage600_val", "open-image600", "Circle", "val"),
            ("openimage600_val", "open-image600", "Box", "val"),
            ("ade600_val", "ADE600", "Point", "val"),
            ("ade600_val", "ADE600", "Scribble", "val"),
            ("ade600_val", "ADE600", "Polygon", "val"),
            ("ade600_val", "ADE600", "Circle", "val"),
            ("ade600_val", "ADE600", "Box", "val"),
            ("davis_val", "DAVIS345", "Point", "val"),
            ("davis_val", "DAVIS345", "Scribble", "val"),
            ("davis_val", "DAVIS345", "Polygon", "val"),
            ("davis_val", "DAVIS345", "Circle", "val"),
            ("davis_val", "DAVIS345", "Box", "val"),
            ("cocomini_val", "COCO_MVal", "Point", "val"),
            ("cocomini_val", "COCO_MVal", "Scribble", "val"),
            ("cocomini_val", "COCO_MVal", "Polygon", "val"),
            ("cocomini_val", "COCO_MVal", "Circle", "val"),
            ("cocomini_val", "COCO_MVal", "Box", "val"),
        ]

    for name, dirname, mode, split in SPLITS:
        register_davis_context(name, os.path.join(root, dirname), mode, split)
        MetadataCatalog.get("{}_{}".format(name, mode)).evaluator_type = "interactive"

_root = os.getenv("DATASET", "datasets")
register_all_davis(_root)