#Pretraining - Synthetic Data with 1D Input Observations 
python train.py --export_data True --dump_path ./dump --max_input_dimension 1 --n_steps_per_epoch 3125 --max_epoch 1000


#Pretraining - Synthetic Data with Max 10D Input Observations 
python train.py --export_data True --dump_path ./dump --max_input_dimension 10 --n_steps_per_epoch 3125 --max_epoch 1000


#Property Prediction - NCR Property 
python train.py --export_data True --is_proppred True --property_type ncr --dump_path ./dump --use_skeleton True --max_input_dimension 1 --n_steps_per_epoch 3125 --max_epoch 1000 --exp_name data --exp_id ncr_exported #SAVE 10K Train Data for NCR Prediction Task