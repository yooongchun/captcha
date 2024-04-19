#! /usr/bin/env python3
# -*- coding:utf-8 -*-

import pathlib

proj_path = pathlib.Path(__file__).absolute().parent.parent

assets_path = proj_path / "assets"
vocabulary_path = assets_path / "vocabulary.txt"
assert vocabulary_path.exists(), f"vocabulary file not found in {vocabulary_path}"

dataset_path = proj_path / "dataset"
labeled_path = dataset_path / "labeled"

inference_path = proj_path / "inference"
