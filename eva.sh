#!/bin/bash
file="$1"
python evaluate_cityscapes.py --restore-from "$file"
python compute_iou.py ./data/Cityscapes/gtFine/val result/cityscapes
