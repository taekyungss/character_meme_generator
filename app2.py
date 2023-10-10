import os
import sys
import gradio as gr
from huggingface_hub.repocard import RepoCard
from diffusers import StableDiffusionPipeline
import torch
from matplotlib import pyplot as plt
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point
from replace_anything_function import Backgroud
# import cv2
import glob
import numpy as np
from pathlib import Path
from typing import Any, Dict, List
from sam_segment import predict_masks_with_sam
from stable_diffusion_inpaint import replace_img_with_sd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def dreambooth_lora(prompt):
    lora_model_id = "ssarae/dreambooth_elmo_ver"
    card = RepoCard.load(lora_model_id)
    base_model_id = card.data.to_dict()["base_model"]

    pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.load_lora_weights(lora_model_id)
    
    text_prompt = prompt
    image = pipe(text_prompt, num_inference_steps=300).images[0]
    image.save("./image/image.png")
    return [image]

def get_latest_image(folder_path):
    image_files = glob.glob(os.path.join(folder_path, '*.jpg')) + glob.glob(os.path.join(folder_path, '*.png'))

    if not image_files:
        print("폴더 내에 이미지 파일이 없습니다.")
        return None
    latest_image = max(image_files, key=os.path.getctime)
    return latest_image

def Background(w,h,text_prompt2): 
    input_img = './image/image.png'
    # input_img= get_latest_image(folder_path)
    coords_type = "key_in"
    point_labels = [1]
    sam_model_type = "vit_h"
    sam_ckpt = './pretrained_models/sam_vit_h_4b8939.pth'
    output_dir = './results'
    point_coords = [w,h]

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

    result_images = []
    result_img_path = []
    for idx, mask in enumerate(masks):
        mask_p = out_dir / f"mask_{idx}.png"
        img_replaced_p = out_dir / f"replaced_with_{Path(mask_p).name}"
        img_replaced = replace_img_with_sd(img, mask, text_prompt2, device=device)
        save_array_to_img(img_replaced, img_replaced_p)
        
        result_img_path.append(img_replaced_p)
        result_images.append(img_replaced)

    images = [
        (f"./results/image/replaced_with_mask_{i}.png", f"result {i}") for i in range(3)
    ]

    return images
    
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    features = gr.State(None)
    
    orig_h = gr.State(None)
    orig_w = gr.State(None)
    input_h = gr.State(None)
    input_w = gr.State(None)

    with gr.Column(variant="panel", scale=1, min_width=600):
        text1 = gr.Textbox(label="캐릭터 형체 생성 prompt", font_size=16)
        btn = gr.Button("Generate image", scale=0)
        gallery = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery", 
            columns=[1], rows=[1], object_fit="contain", height="auto")
    btn.click(dreambooth_lora, inputs=[text1], outputs=gallery)

    with gr.Row().style(mobile_collapse=False, equal_height=True):
        with gr.Column(variant="panel"):
            with gr.Row():
                gr.Markdown("## Input Image")
            with gr.Row():
                img = gr.Image(label="Input Image").style(height=300)
        with gr.Column(variant="panel"):
            with gr.Row():
                gr.Markdown("## Pointed Image")
            with gr.Row():
                img_pointed = gr.Plot(label='Pointed Image')

            clear_button_image = gr.Button(value="Reset", label="Reset", variant="secondary").style(height=300, width = 300)

    with gr.Column(variant="panel", scale=1, min_width=600):
        text2 = gr.Textbox(label="배경 생성 prompt",font_size=16)
        w = gr.Number(label="Point Coordinate W")
        h = gr.Number(label="Point Coordinate H")
        
        btn = gr.Button("Generate image", scale=0)
        gallery = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery", 
            columns=[2], rows=[2], object_fit="contain", height=600)
        
    btn.click(Background, inputs=[w,h,text2], outputs=[gallery])

    def get_select_coords(img, evt: gr.SelectData):
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        fig = plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        show_points(plt.gca(), [[evt.index[0], evt.index[1]]], [1],
                    size=(width*0.04)**2)
        return evt.index[0], evt.index[1], fig

    img.select(get_select_coords, [img], [w, h, img_pointed])

    def reset(*args):
        return [None for _ in args]

    clear_button_image.click(
        reset,
        [img, features, img_pointed, w, h],
        [img, features, img_pointed, w, h]
    )
    
if __name__ == "__main__":
    demo.launch(share=True)
    