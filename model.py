import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
os.chdir("../")
import gradio as gr
from huggingface_hub.repocard import RepoCard
from diffusers import StableDiffusionPipeline
import torch
from matplotlib import pyplot as plt
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point
from replace_anything_function import Backgroud
import cv2
import glob
import numpy as np
from pathlib import Path
from typing import Any, Dict, List
from sam_segment import predict_masks_with_sam
from stable_diffusion_inpaint import replace_img_with_sd

def dreambooth_lora(prompt):
    lora_model_id = "ssarae/dreambooth_elmo_ver"
    card = RepoCard.load(lora_model_id)
    base_model_id = card.data.to_dict()["base_model"]

    pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)

    pipe = pipe.to("cuda")
    pipe.load_lora_weights(lora_model_id)
    
    text_prompt = prompt
    image = pipe(text_prompt, num_inference_steps=300).images[0]
    image.save("./image/image.png")
    return [image]


def Background(w,h,text_prompt2): 
    input_img = './image/image.png'
    coords_type = "key_in"
    point_labels = [1]
    sam_model_type = "vit_h"
    sam_ckpt = './pretrained_models/sam_vit_h_4b8939.pth'
    output_dir = './results'
    point_coords = [w,h]
    
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
            img, mask, text_prompt2, device=device)
        save_array_to_img(img_replaced, img_replaced_p)
        
        result_images.append(img_replaced)
    # print(result_images)    
    return result_images

