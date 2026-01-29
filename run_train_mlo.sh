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