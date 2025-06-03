export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
torchrun --nproc_per_node=8 train_hummingbird.py --batch-size 1 \
--clip 0.5 --aesthetic 1.0 --pickscore 1.0 --hpsv2 1.0 --context 1.0 \
--grad_steps 5 \
--lr_image 0. --lr_unet 5e-6 \
--output-path /path/to/save/checkpoints \
--dataset vqav2gqa