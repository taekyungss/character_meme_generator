import cv2
import sys
import os
import glob
import argparse
import numpy as np
import torch
from pathlib import Path
from matplotlib import pyplot as plt
from typing import Any, Dict, List
from sam_segment import predict_masks_with_sam
from stable_diffusion_inpaint import replace_img_with_sd
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point


def get_latest_image(folder_path):
    image_files = glob.glob(os.path.join(folder_path, '*.jpg')) + glob.glob(os.path.join(folder_path, '*.png'))

    if not image_files:
        print("폴더 내에 이미지 파일이 없습니다.")
        return None
    latest_image = max(image_files, key=os.path.getctime)
    return latest_image


def Backgroud(point_coords, text_prompt): 
    folder_path = './image'
    input_img= get_latest_image(folder_path)
    coords_type = "key_in"
    point_labels = [1]
    sam_model_type = "vit_h"
    sam_ckpt = './pretrained_models/sam_vit_h_4b8939.pth'
    output_dir = './results'
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if coords_type == "click":
        latest_coords = get_clicked_point(input_img)
    elif coords_type == "key_in":
        latest_coords = point_coords
        # print(latest_coords)
    img = load_img_to_array(input_img)

    masks, _, _ = predict_masks_with_sam(
        img,
        [latest_coords],
        point_labels,
        model_type=sam_model_type,
        ckpt_p=sam_ckpt,
        device=device,
    )

    masks = masks.astype(np.uint8) * 255

    img_stem = Path(input_img).stem
    out_dir = Path(output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    result_images = []

    for idx, mask in enumerate(masks):
        mask_p = out_dir / f"mask_{idx}.png"
        img_replaced_p = out_dir / f"replaced_with_{Path(mask_p).name}"
        img_replaced = replace_img_with_sd(
            img, mask, text_prompt, device=device)
        save_array_to_img(img_replaced, img_replaced_p)
        
        result_images.append(img_replaced)
        
    return result_images
    
    
# print(Backgroud([212,222], "The object stands in front of Hansel and Gretel's confectionery"))