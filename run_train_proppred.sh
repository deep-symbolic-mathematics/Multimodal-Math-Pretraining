# Property Prediction - Example: NCR Property Prediction 

#Without Pretraining 
python train.py --is_proppred True --property_type ncr --reload_data functions,dump/data/ncr/train.prefix,dump/data/ncr/train.prefix, --normalize_y True --batch_size 16 --dump_path ./dump --max_input_dimension 1 --n_steps_per_epoch 625 --max_epoch 100000 --exp_name NCR_pred --exp_id run1 --lr 1e-5 --latent_dim 512 --save_periodic 10

#With Pretraining - Frozen Encoder
python train.py --reload_model ./weights/snip-1d-normalized.pth --is_proppred True --property_type ncr --freeze_encoder True --reload_data functions,dump/data/ncr/train.prefix,dump/data/ncr/train.prefix, --normalize_y True --batch_size 16 --dump_path ./dump --max_input_dimension 1 --n_steps_per_epoch 625 --max_epoch 100000 --exp_name NCR_pred --exp_id run1 --lr 1e-5 --latent_dim 512 --save_periodic 10

#With Pretraining - Finetune Ecnoder
python train.py --reload_model ./weights/snip-1d-normalized.pth --is_proppred True --property_type ncr --freeze_encoder False --reload_data functions,dump/data/ncr/train.prefix,dump/data/ncr/train.prefix, --normalize_y True --batch_size 16 --dump_path ./dump --max_input_dimension 1 --n_steps_per_epoch 625 --max_epoch 100000 --exp_name NCR_pred --exp_id run1 --lr 1e-5 --latent_dim 512 --save_periodic 10
