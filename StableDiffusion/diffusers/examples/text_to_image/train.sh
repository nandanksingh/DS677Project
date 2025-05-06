 
export MODEL_NAME="CompVis/stable-diffusion-v1-4" #pre-trained model
export TRAIN_DIR="/project/sz457/ms3537/DS677/diffusers/examples/text_to_image/flickr30k_train" #path to our dataset

accelerate launch --mixed_precision="fp16" train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=10000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd_flickr30k_model10k"
 
 