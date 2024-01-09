# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import torch
from .embedders import LinearPointEmbedder
from .transformer import TransformerModel
from .sklearn_wrapper import SymbolicTransformerRegressor
from .model_wrapper import ModelWrapper
import torch.nn as nn

logger = getLogger()


def check_model_params(params):
    """
    Check models parameters.
    """
    # model dimensions
    assert params.enc_emb_dim % params.n_enc_heads == 0
    assert params.dec_emb_dim % params.n_dec_heads == 0

    # reload a pretrained model
    if params.reload_model != "":
        print("Reloading model from ", params.reload_model)
        assert os.path.isfile(params.reload_model)


class MLPRegressor(nn.Module):
    def __init__(self, params):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(params.latent_dim, 128)
        self.fc2 = nn.Linear(128 , 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
class MLPClassifier(nn.Module):
    def __init__(self, params):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(params.latent_dim, 128)
        self.fc2 = nn.Linear(128 , 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


def build_modules(env, params):
    """
    Build modules.
    """
    modules = {}
    modules["embedder"] = LinearPointEmbedder(params, env)
    env.get_length_after_batching = modules["embedder"].get_length_after_batching

    modules["encoder_y"] = TransformerModel(
        params,
        env.float_id2word,
        is_encoder=True,
        with_output=False,
        use_prior_embeddings=True,
        positional_embeddings=params.enc_positional_embeddings,
    )
    modules["encoder_f"] = TransformerModel(
        params,
        env.equation_id2word,
        is_encoder=True,
        with_output=False,
        use_prior_embeddings=False,
        positional_embeddings=params.enc_positional_embeddings,
    )
    modules["decoder"] = TransformerModel(
        params,
        env.equation_id2word,
        is_encoder=False,
        with_output=True,
        use_prior_embeddings=False,
        positional_embeddings=params.dec_positional_embeddings,
    )
    
    modules["regressor"] = MLPRegressor(
        params,
    )
 
    modules["classifier"] = MLPClassifier(
        params,
    )

    # reload pretrained modules
    if params.reload_model != "":
        logger.info(f"Reloading modules from {params.reload_model} ...")
        reloaded = torch.load(params.reload_model)
        modules_to_load = ['embedder', 'encoder_y','encoder_f'] #SNIP modules
        if params.is_proppred:
            modules_to_load = ['encoder_f'] #symbolic encoder (encoder_f) for numeric properties
            modules_to_load = ['encoder_y'] #numeric encoder (encoder_y) for symbolic properties
        
        
        # for k, v in modules.items():
        #     assert k in reloaded
        for k in modules_to_load:
            assert k in reloaded
            v = modules[k]
            if all([k2.startswith("module.") for k2 in reloaded[k].keys()]):
                reloaded[k] = {
                    k2[len("module.") :]: v2 for k2, v2 in reloaded[k].items()
                }
            v.load_state_dict(reloaded[k])

    # log
    for k, v in modules.items():
        logger.debug(f"{v}: {v}")
    for k, v in modules.items():
        logger.info(
            f"Number of parameters ({k}): {sum([p.numel() for p in v.parameters() if p.requires_grad])}"
        )

    # cuda
    if not params.cpu:
        for v in modules.values():
            v.cuda()

    return modules
