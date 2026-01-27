#!/bin/bash

# First test dimensions
echo "Testing model dimensions..."
python test_dimensions.py

if [ $? -ne 0 ]; then
    echo "Dimension tests failed! Please fix errors before training."
    exit 1
fi

echo ""
echo "Dimension tests passed! Starting training..."
echo ""

python train_mlo_skin.py \
    --train_csv /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/train.csv \
    --val_csv /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/val.csv \
    --test_csv /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/test.csv \
    --img_dir /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/images \
    --pretrained_unet /home/ubuntu/Documents/Nikita/GenSeg/unet/unet.pkl \
    --epochs 50 \
    --batch_size 32 \
    --mask_ratio 0.75 \
    --loss_lambda 1.0 \
    --mae_lr 1.5e-4 \
    --cls_lr 5e-4 \
    --mask_lr 5e-5 \
    --unroll_steps_mae 2 \
    --unroll_steps_cls 1 \
    --unroll_steps_mask 1 \
    --valid_step 100 \
    --log_freq 10 \
    --output_dir ./checkpoints/mlo_full \
    --exp_name mlo_hybrid_unet \
    --wandb_mode online \
    --num_workers 8 \
    --device cuda

# python train_mlo.py \
#     --data_path /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/ \
#     --use_seg_init \
#     --seg_model_path /home/ubuntu/Documents/Nikita/GenSeg/unet/unet.pkl \
#     --epochs 100 \
#     --batch_size 16 \
#     --unroll_steps_mae 2 \
#     --unroll_steps_classifier 1 \
#     --unroll_steps_masking 1 \
#     --lr_mae 1.5e-4 \
#     --lr_classifier 5e-4 \
#     --lr_masking 5e-5 \
#     --wandb_mode online

# --train_samples 40 \