from torch import nn
import torch.nn.functional as F
import torch
import sys 
import copy
import snip
from snip.model.transformer import TransformerModel
from snip.slurm import init_signal_handler, init_distributed_mode
from snip.utils import bool_flag, initialize_exp
from snip.model import check_model_params, build_modules
from snip.envs import build_env
from parsers import get_parser
from snip.utils import to_cuda
import os
import numpy as np
from pathlib import Path
from snip.trainer import Trainer
from collections import OrderedDict, defaultdict
import sympy as sp
from snip.model.model_wrapper import ModelWrapper
from snip.model.sklearn_wrapper import SymbolicTransformerRegressor


class SNIPPredictor(nn.Module): 
    def __init__(self, params, env, modules):
        super().__init__()
        self.modules = modules
        self.params = params
        self.env = env
        if self.params.is_proppred:
            if self.params.property_type in ['ncr','upward','yavg','oscil']:
                self.encoder_f, self.predictor = (
                    self.modules["encoder_f"],
                    self.modules["regressor"],
                    )
                self.encoder_f.eval()
                self.predictor.eval()
            else:
                self.embedder , self.encoder_y, self.predictor = (
                    self.modules["embedder"],
                    self.modules["encoder_y"],
                    self.modules["regressor"],
                    # self.modules["classifier"],
                    )
                self.embedder.eval()
                self.encoder_y.eval()
                self.predictor.eval()

    def forward(self,samples):
        if self.params.is_proppred:
            if self.params.property_type in ['ncr','upward','yavg','oscil']:
                if self.params.use_skeleton:
                    x2, len2 = self.env.batch_equations(
                        self.env.word_to_idx(
                            samples["skeleton_tree_encoded"], float_input=False))
                else:
                    x2, len2 = self.env.batch_equations(
                        self.env.word_to_idx(samples["tree_encoded"], float_input=False))
                alen = torch.arange(self.params.max_src_len, dtype=torch.long, device=len2.device) #modified
                pred_mask = (alen[:, None] < len2[None] - 1)  # do not predict anything given the last target word
                y = x2[1:].masked_select(pred_mask[:-1])
                assert len(y) == (len2 - 1).sum().item()
                x2, len2, y = to_cuda(x2, len2, y)
                encoded_f = self.encoder_f("fwd", x=x2, lengths=len2, causal=False)
                property_output = self.predictor(encoded_f)
                outputs = (encoded_f, property_output)
                return outputs
            
            else:
                x_to_fit = samples["x_to_fit"]
                y_to_fit = samples["y_to_fit"]
            
                x1 = []
                for seq_id in range(len(x_to_fit)):
                    x1.append([])
                    for seq_l in range(len(x_to_fit[seq_id])):
                        x1[seq_id].append([x_to_fit[seq_id][seq_l], y_to_fit[seq_id][seq_l]])

                x1, len1 = self.embedder(x1)
                encoded_y = self.encoder_y("fwd", x=x1, lengths=len1, causal=False)
                property_output = self.predictor(encoded_y)

                outputs = (encoded_y, property_output)
                return outputs