# #!/bin/bash

# # First test dimensions
# echo "Testing model dimensions..."
# python test_dimensions.py

# if [ $? -ne 0 ]; then
#     echo "Dimension tests failed! Please fix errors before training."
#     exit 1
# fi

# echo ""
# echo "Dimension tests passed! Starting training..."
# echo ""

python train_mlo_skin.py \
    --train_csv /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/train.csv \
    --val_csv /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/val.csv \
    --test_csv /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/test.csv \
    --img_dir /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/images \
    --unet_checkpoint /home/ubuntu/Documents/Nikita/GenSeg/unet/unet.pkl \
    --epochs 50 \
    --batch_size 32 \
    --mask_ratio 0.75 \
    --lr_mae 1.5e-4 \
    --lr_cls 5e-4 \
    --lr_mask 5e-5 \
    --unroll_steps_mae 2 \
    --unroll_steps_cls 1 \
    --unroll_steps_mask 1 \
    --valid_step 100 \
    --output_dir ./checkpoints/SKIN-LESION-MLO \
    --wandb_mode online \
    --num_workers 8 \
    --device cuda

#     --loss_lambda 1.0 \,     --log_freq 10 \,     --exp_name mlo_hybrid_mae \

# usage: train_mlo_skin.py [-h] 
# --train_csv TRAIN_CSV --val_csv VAL_CSV [--test_csv TEST_CSV] --img_dir IMG_DIR 
# --unet_checkpoint UNET_CHECKPOINT [--mae_model MAE_MODEL] [--input_size INPUT_SIZE]
# [--patch_size PATCH_SIZE] [--mask_ratio MASK_RATIO] [--norm_pix_loss] [--epochs EPOCHS] 
# [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS] [--lr_mae LR_MAE]
# [--lr_cls LR_CLS] [--lr_mask LR_MASK] [--weight_decay WEIGHT_DECAY] 
# [--unroll_steps_mae UNROLL_STEPS_MAE] [--unroll_steps_cls UNROLL_STEPS_CLS]
# [--unroll_steps_mask UNROLL_STEPS_MASK] [--valid_step VALID_STEP] [--baseline] [--use_fake_images] 
# [--output_dir OUTPUT_DIR] [--device DEVICE] [--seed SEED]
# [--wandb_project WANDB_PROJECT] [--wandb_name WANDB_NAME] [--wandb_mode WANDB_MODE]
# train_mlo_skin.py: error: the following arguments are required: --unet_checkpoint
