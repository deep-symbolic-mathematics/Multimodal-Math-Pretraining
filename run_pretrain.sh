# MAX INPUT DIM 1
#Pretrain from scratch
python train.py --loss_type CLIP --normalize_y True --batch_size 256 --dump_path ./dump --max_input_dimension 1 --n_steps_per_epoch 1000 --max_epoch 100000 --exp_name B256 --exp_id run1-1d --lr 4e-5 --latent_dim 512 --save_periodic 10

#Load and Continue Pretraining on the pretrained model weights
python train.py --reload_model ./weights/snip-1d-normalized.pth --loss_type CLIP --batch_size 256 --dump_path ./dump --max_input_dimension 1 --n_steps_per_epoch 1000 --max_epoch 100000 --exp_name B256 --exp_id run1-1d --lr 4e-5 --latent_dim 512 --save_periodic 10


###########################################
# MAX INPUT DIM 10
#Pretrain from scratch
python train.py --loss_type CLIP --batch_size 256 --dump_path ./dump --max_input_dimension 10 --n_steps_per_epoch 1000 --max_epoch 100000 --exp_name B256 --exp_id run1-10d --lr 4e-5 --latent_dim 512 --save_periodic 10


#Load and Continue Pretraining on the pretrained model weights
python train.py --reload_model ./weights/snip-10dmax.pth --loss_type CLIP --batch_size 256 --dump_path ./dump --max_input_dimension 10 --n_steps_per_epoch 1000 --max_epoch 100000 --exp_name B256 --exp_id run1-10d --lr 4e-5 --latent_dim 512 --save_periodic 10