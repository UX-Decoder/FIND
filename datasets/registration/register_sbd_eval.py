# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import os
import glob
from typing import List, Tuple, Union

import numpy as np
from scipy.io import loadmat

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager


__all__ = ["load_sbd_instances", "register_sbd_context"]

def get_labels_with_sizes(x):
    obj_sizes = np.bincount(x.flatten())
    labels = np.nonzero(obj_sizes)[0].tolist()
    labels = [x for x in labels if x != 0]
    return labels, obj_sizes[labels].tolist()

def load_sbd_instances(name: str, dirname: str, mode: str, split: str):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with PathManager.open(os.path.join(dirname, split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)
   
    dicts = []
    for field in fileids:
        image_path = os.path.join(dirname, "img", "{}.jpg".format(field))
        inst_info_path = os.path.join(dirname, "inst", "{}.mat".format(field))

        instances_mask = loadmat(str(inst_info_path))['GTinst'][0][0][0].astype(np.int32)
        instances_ids, _ = get_labels_with_sizes(instances_mask)

        for instances_id in instances_ids:
            r = {
                "file_name": image_path,
                "inst_info_name": inst_info_path,
                "inst_id": instances_id,
            }
            dicts.append(r)
    return dicts

def register_sbd_context(name, dirname, mode, split):
    DatasetCatalog.register("{}_{}".format(name, mode), lambda: load_sbd_instances(name, dirname, mode, split))
    MetadataCatalog.get("{}_{}".format(name, mode)).set(
        dirname=dirname,
        thing_dataset_id_to_contiguous_id={},
    )

def register_all_sbd(root):
    SPLITS = [
            ("sbd_val", "SBD", "Point", "val"),
            ("sbd_val", "SBD", "Scribble", "val"),
            ("sbd_val", "SBD", "Polygon", "val"),
            ("sbd_val", "SBD", "Circle", "val"),
        ]
        
    for name, dirname, mode, split in SPLITS:
        register_sbd_context(name, os.path.join(root, dirname), mode, split)
        MetadataCatalog.get("{}_{}".format(name, mode)).evaluator_type = "interactive"

_root = os.getenv("DATASET", "datasets")
register_all_sbd(_root)