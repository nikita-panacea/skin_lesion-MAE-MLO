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

# run_train_mlo.sh

python train_mlo_skin.py \
    --data_path /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/ \
    --unet_path /home/ubuntu/Documents/Nikita/GenSeg/unet/unet.pkl \
    --epochs 50 \
    --batch_size 32 \
    --mask_ratio 0.75 \
    --lr_mae 1.5e-4 \
    --lr_classifier 5e-4 \
    --lr_masking 5e-5 \
    --unroll_mae 2 \
    --unroll_classifier 1 \
    --unroll_masking 1 \
    --val_freq 100 \
    --output_dir ./checkpoints/SKIN-LESION-MLO \
    --wandb_mode online \
    --num_workers 8 \
    --device cuda
    
    
    # --train_csv /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/train.csv \
    # --val_csv /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/val.csv \
    # --test_csv /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/test.csv \
    # --img_dir /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/images \

# usage: MLO-MAE Skin Lesion Training [-h] [--data_path DATA_PATH] [--unet_path UNET_PATH] [--epochs EPOCHS] [--batch_size BATCH_SIZE] 
# [--lr_mae LR_MAE] [--lr_classifier LR_CLASSIFIER]
# [--lr_masking LR_MASKING] [--weight_decay WEIGHT_DECAY] [--unroll_mae UNROLL_MAE] [--unroll_classifier UNROLL_CLASSIFIER] 
# [--unroll_masking UNROLL_MASKING][--val_freq VAL_FREQ] [--mask_ratio MASK_RATIO] [--norm_pix_loss] [--num_workers NUM_WORKERS] 
# [--device DEVICE] [--seed SEED] [--output_dir OUTPUT_DIR][--wandb_project WANDB_PROJECT] [--wandb_name WANDB_NAME] 
# [--wandb_mode {online,offline,disabled}]
# MLO-MAE Skin Lesion Training: error: unrecognized arguments: --train_csv /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/train.csv --val_csv /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/val.csv --test_csv /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/test.csv --img_dir /home/ubuntu/Documents/Nikita/ISIC_2019_dataset/Train_set/images --unet_checkpoint /home/ubuntu/Documents/Nikita/GenSeg/unet/unet.pkl --lr_cls 5e-4 --unroll_steps_mae 2 --unroll_steps_cls 1 --unroll_steps_mask 1 --valid_step 100
