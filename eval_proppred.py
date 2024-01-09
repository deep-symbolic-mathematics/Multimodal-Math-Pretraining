import os
import torch
import numpy as np
import datetime
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from model import SNIPPredictor
import torch
from itertools import cycle
from tqdm import tqdm
import argparse
from parsers import get_parser
from pathlib import Path
import snip
from snip.envs import build_env
from snip.model import check_model_params, build_modules
from snip.utils import bool_flag, initialize_exp
from snip.slurm import init_signal_handler, init_distributed_mode
from snip.trainer import Trainer
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100


def calculate_accuracy(pred_values, gt_values, tolerance_percentage):
    absolute_difference = np.abs(pred_values - gt_values)
    tolerance_threshold = (tolerance_percentage / 100.0) * gt_values
    within_tolerance = np.sum(absolute_difference <= tolerance_threshold)
    accuracy = (within_tolerance / len(pred_values)) * 100.0 
    return accuracy


def reload_checkpoint(params,modules, path, requires_grad=False):
    """
    Reload a checkpoint if we find one.
    """
    if path is None:
        path = "checkpoint.pth"
    assert os.path.isfile(path)

    data = torch.load(path, map_location="cpu")
    modules_to_load = ['embedder', 'encoder_y','encoder_f']
    if params.is_proppred:
        if params.property_type in ['ncr','upward','yavg','oscil']:
            print("Loading Symbolic Encoder for Numeric Property Prediction")
            modules_to_load = ['encoder_f'] #symbolic encoder (encoder_f) for numeric properties
        else:
            print("Loading Numeric Encoder for Symbolic Property Prediction")
            modules_to_load = ['embedder','encoder_y'] #numeric encoder (encoder_y) for symbolic properties
            
    for k in modules_to_load:
        assert k in data
        v = modules[k]
        weights = data[k]
        try:
            weights = data[k]
            v.load_state_dict(weights)
        except RuntimeError:  # remove the 'module.'
            weights = {name.partition(".")[2]: v for name, v in data[k].items()}
            v.load_state_dict(weights)
        v.requires_grad = requires_grad
    return modules


def main(params):
    #load model    
    model_type = 3
    # encoder_type = 'frozen'
    # target = 'NCR_pred'
    # path = 'dump/'+target+'/'+encoder_type+'_model.pth'
    path = params.reload_model
    if model_type == 2:
        params.use_skeleton = True
    elif model_type == 3:
        params.normalize_y = True
    elif model_type == 4:
        params.normalize_y = True
        params.use_skeleton = True

    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if params.batch_size_eval is None:
        params.batch_size_eval = int(1.5 * params.batch_size)
    params.is_slurm_job = False
    
    env = build_env(params)
    modules = build_modules(env, params)
    init_distributed_mode(params)
    if params.is_slurm_job:
        init_signal_handler()

    # CPU / CUDA
    if not params.cpu:
        params.device = 'cuda'
        assert torch.cuda.is_available()
    else:
        params.device = 'cpu'
    snip.utils.CUDA = not params.cpu

    if params.batch_size_eval is None:
        params.batch_size_eval = int(1.5 * params.batch_size)
    if params.eval_dump_path is None:
        params.eval_dump_path = Path(params.dump_path) / "evals_all"
        if not os.path.isdir(params.eval_dump_path):
            os.makedirs(params.eval_dump_path)

    #load data
    nsteps = 0
    total_seen = 0
    trainer = Trainer(modules, env, params)

    trainer.modules = reload_checkpoint(params,trainer.modules, path)
    model = SNIPPredictor(params = params, env=env, modules=trainer.modules)
    model = torch.nn.DataParallel(model)
    model.to(params.device)

    if params.reload_data != "":
        s = [x.split(",") for x in params.reload_data.split(";") if len(x) > 0]
        trainer.data_path = {task: (
                    train_path if train_path != "" else None,
                    valid_path if valid_path != "" else None,
                    test_path if test_path != "" else None,)
                for task, train_path, valid_path, test_path in s}

    gt_str_list = []
    z_f_list = []
    gt_ncr_list = []
    pred_ncr_list = []

    bn = torch.nn.BatchNorm1d(params.latent_dim, affine=False).to(params.device)
    
    for task_id in np.random.permutation(len(params.tasks)):
        task = params.tasks[task_id]

        trainer.inner_epoch = 0
        while trainer.inner_epoch < trainer.n_steps_per_epoch:
            samples, _ = trainer.get_batch(task)
            print("Trainer Iteration: ", trainer.inner_epoch)
            #model forwards
            with torch.no_grad():
                outputs = model(samples)
                encoded_f, property_output= outputs
                bs = encoded_f.shape[0]

                replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
                for i in range(bs):
                    gt_str = samples['tree'][i].infix()
                    for op,replace_op in replace_ops.items():
                        gt_str = gt_str.replace(op,replace_op)
                    gt_str_list.append(gt_str)
                    gt_ncr_list.append(samples['target_property'][i])

                z_f_list.append(encoded_f.squeeze().detach().cpu().numpy()) 

                for i in range(len(property_output)):
                    pred_ncr_list.append(property_output[i].item())

            trainer.inner_epoch += 1
            nsteps += 1

        gt_ncr = np.array(gt_ncr_list)
        pred_ncr = np.array(pred_ncr_list)
        r2_pred = r2_score(gt_ncr, pred_ncr)
        mse_pred = np.mean( (gt_ncr - pred_ncr) ** 2 )
        eps = 1e-12
        nmse_pred = np.mean( (gt_ncr - pred_ncr) ** 2 ) / ( np.mean( (gt_ncr - np.mean(gt_ncr))**2 )+eps)


        print("R2 pred: ", r2_pred)
        print("MSE pred: ", mse_pred)
        print("NMSE pred: ", nmse_pred)

        print("NORMALIZED GT VALUES: ")
        max_gt = np.max(gt_ncr)
        min_gt = np.min(gt_ncr)
        gt_ncr = (gt_ncr - min_gt) / (max_gt - min_gt)
        pred_ncr = (pred_ncr - min_gt) / (max_gt - min_gt)

        atol = 0.01
        acc_atol = np.sum(np.isclose(pred_ncr, gt_ncr, rtol=0.0, atol=atol))
        print(f"Accuracy within absolute {atol}% tolerance: {acc_atol/10:.2f}%")

        atol = 0.1
        acc_atol = np.sum(np.isclose(pred_ncr, gt_ncr, rtol=0.0, atol=atol))
        print(f"Accuracy within absolute {atol}% tolerance: {acc_atol/10:.2f}%")

        atol = 1
        acc_atol = np.sum(np.isclose(pred_ncr, gt_ncr, rtol=0.0, atol=atol))
        print(f"Accuracy within absolute {atol}% tolerance: {acc_atol/10:.2f}%")

        chance_ncr = np.ones_like(gt_ncr) * np.mean(gt_ncr)
        atol = 0.1
        acc_atol = np.sum(np.isclose(chance_ncr, gt_ncr, rtol=0.0, atol=atol))
        print(f"Chance Level Accuracy within absolute {atol}% tolerance: {acc_atol:.2f}%")

 
            
if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()
    # params.device = torch.device("cuda")
    params.batch_size =1 
    params.n_steps_per_epoch = 1000
    params.max_input_dimension = 1
    params.latent_dim = 512

    ### Uncomment for 1D properties (only comment for 10d)
    params.min_binary_ops_per_dim = 3 
    # params.reload_data = 'functions,dump/data/ncr/test.prefix,dump/data/ncr/test.prefix,'
    main(params)