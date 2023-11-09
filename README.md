# character_meme_generator
> To create your own character image using **DreamBooth with LoRA** and **Inpaint Anything**
- This repository has not been cleaned up yet
## Overview
Have you ever heard of the term "meme" ? "Meme" refers to images or pictures that can express a specific word or feeling like emoticons. We usually enjoy collecting memes from websites like Pinterest, Naver, and Google. There are so many diverse memes out there, but it can be quite challenging to find the perfect meme that suits our needs. That's why we thought, "Why not create the memes we want ourselves?" and decided to start this project.


Here's flowchart depicting the models' workflow
<img src="./images/meme_architecture.png">
## Dataset
We can create 6 characters :
`Gromit`, `Pingu`, `Zzangu`, `Loopy` , `Kuromi`, and `Elmo`.

Normally, 5 to 10 images are enough, but due to the characters' less distinct features, we prepared 30 to 45 images to generate high-quality images.
| **Character**| **Number** |
| ------------ | ------ | 
| Gromit       | 19     |
| Pingu        | 45     | 
| Zzangu       | 39     |
| Loopy        | 37     |
| Kuromi       | 30     |
| Elmo         | 40     |




## Training
### DreamBooth with LoRA
This has trained the model with default settings, including 512x512 resolution, 8GB GPU memory occupied, 1 image per batch, a learning rate of 1e-4, and the training step is set to the value obtained by multiplying the number of training images by 200.

First, Initialize [🤗Accelerate](https://huggingface.co/docs/accelerate/index) environment with:
 ```
  accelerate config
 ```
 It is developed by Hugging Face.

 Then, Run the training script. 

 ```bash
accelerate launch train_dreambooth_lora.py \
--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
--instance_data_dir="images/zzangu" \
--instance_prompt="A wkdrn zzangu" \
--validation_prompt="A wkdrn zzangu standing" \
--resolution=512 \
--train_batch_size=1 \
--gradient_accumulation_steps=1 \
--learning_rate=1e-4 \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--max_train_steps=400 \
--validation_epochs=50 \
--seed="0" \
--push_to_hub
```

## Inference


## Demo
