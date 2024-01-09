# SNIP: A Multimodal Symbolic-Numeric Intergrated Pretraining for Math (MathCLIP)

Official Implementation of **[SNIP: Bridging Mathematical Symbolic and Numeric Realms with Unified Pre-training](https://arxiv.org/abs/2310.02227)**. 


## Overview
Insipired by the great performance of [CLIP](https://arxiv.org/abs/2310.02227) in vision-language representation learning, we introduce a multi-modal pre-training model for symbolic mathematics, known as **SNIP** for **Symbolic-Numeric Integrated Pre-training**, which emphasizes the significance of numeric-augmented representations in math representation learning. 

<p align="center">
<img src="./images/SNIP.gif" width="80%" /> 
 <br>
<b>SNIP, a pretrained multi-modal transformer model, bridges symbolic math equations and numeric data through contrastive learning, effectively encoding similarities between these domains.</b>
</p>



## Installation

The code requires some dependencies as specified in `environment.yml`. Please follow the relevant libraries to install or run:

```
conda env create -f environment.yml
```
This library requires `python>3.7`



## Pretrained Models
We've relased two pretrained SNIP models, each designed for different types of analysis. Download them [here](https://drive.google.com/drive/folders/1-UDCDQWQi7ZEHyTJryErQadtzXouhByT?usp=sharing). Here's what you'll find:

- **[SNIP-10dmax :](https://drive.usercontent.google.com/download?id=1Q3g6rzqkguHt0krolOKGh4OA5f3JwZLe&export=download&authuser=0&confirm=t&uuid=8de0d6d8-00b5-4820-9ca4-1132d655f02f&at=APZUnTU8MZN1sqlewgAWqVSS00O1:1704728992176)** This model handles **up to 10-dimensional inputs**. More info in Section 5 and Appendix D p.3 of our [paper](https://arxiv.org/pdf/2310.02227.pdf).

- **[SNIP-1d-normalized :](https://drive.usercontent.google.com/download?id=18oUGkH8lSKSNEsSgXJwHZtkzAI8o9fGR&export=download&authuser=0&confirm=t&uuid=f7557162-3bca-436d-91f7-424481b33ee8&at=APZUnTVgWecsokWK1dILPl0IIHBF:1704729034259)** This model is for **1-dimensional inputs** with **normalized targets**, great for focusing on function patterns. Details in Section 4 and Appendix D of our [paper](https://arxiv.org/pdf/2310.02227.pdf).

To use them, create a `weights/` folder in your project, download the checkpoints there, and use the `--reload_model` parameter with the model path, like `--reload_model ./weights/snip-1d-normalized.pth`."


## Pretraining Data Generation
For pretraining, we generate synthetic (symbolic, numeric) pairs for math functions, based on methods from [Deep Learning for Symbolic Mathematics](https://openreview.net/forum?id=S1eZYeHFDS) and [End-to-end Symbolic Regression with Transformers](https://openreview.net/forum?id=S1eZYeHFDS). Each pair includes data points $(x,y)$ and a math function $f$ such that $y=f(x)$. See `generate_datapoints` function [here](./snip/envs/generators.py) for more info. You can also adjust data generation settings [here](./snip/envs/environment.py). 

The data is generated on-the-fly during training, but if you want to create and analyze it beforehand, use `run_export_data.sh`:
```
python pretrain.py --export_data True --dump_path ./dump --max_input_dimension 10 --n_steps_per_epoch 3125 --max_epoch 1000
```
Your exported data will be in the `data.prefix` file.


## SNIP Pretraining
All training settings for SNIP are in `parsers.py`. SNIP uses Transformer encoders for both symbolic and numeric heads, which you can find in the `encoder_f` and `encoder_y` modules [here](./snip/model/__init__.py). For information on contrastive learning and training, look at this [file](./snip/trainer.py). Here's how you can start training:
```
python train.py --loss_type CLIP \
                --batch_size 256 \
                --dump_path ./dump \
                --max_input_dimension 10 \
                --n_steps_per_epoch 1000 \
                --max_epoch 100000 \
                --exp_id run1-10d \
                --lr 4e-5 \
                --latent_dim 512 \
                --save_periodic 10
```
Feel free to adjust training and data settings in `parsers.py` and `environment.py` under `snip/envs/`. After running the command, the model trained for every 10 (`save_periodic`) epochs is saved in `./dump/`.


## Using SNIP for Cross-modal Property Prediction
Here we have provided code to test SNIP representations for the cross-modal symbolic-to-numeric property prediction tasks, meaning that in these tasks, the input is the symbolic mathematical equation and the label is the propery defined based on numeric data observations. 

### Data Generation 
To try it out, start by generating data. For instance, to generate $10$k training examples for the **Non-Convexity Ratio (NCR)** prediction task (as explained in our [paper](https://arxiv.org/pdf/2310.02227.pdf)), use this command:
```
python train.py --export_data True --is_proppred True --property_type ncr --dump_path ./dump --max_input_dimension 1 --n_steps_per_epoch 625  --exp_name data --exp_id ncr
```

This saves data for `ncr` property in `./dump/data/ncr/`. To generate data for other properties, just change the `property_type` parameter. 

### Training 
For this task, we use a Transformer encoder architecure (to encode symbolic equation inputs), followed by a regression predictor head (to predict property). Training is done using Mean Squared Error (MSE) loss. Following are the commands for training different model variants defined in Sec 4 of [paper](https://arxiv.org/pdf/2310.02227.pdf). 

Supervised Model (without Pretrining):
```
python train.py --is_proppred True \
                --property_type ncr \
                --reload_data functions,dump/data/ncr/train.prefix,dump/data/ncr/train.prefix, \
                --normalize_y True \
                --batch_size 16 \
                --dump_path ./dump \
                --max_input_dimension 1 \
                --n_steps_per_epoch 625 \
                --max_epoch 100000 \
                --exp_name NCR_pred \
                --exp_id run1 \
                --lr 1e-5 \
                --latent_dim 512 \
                --save_periodic 10
```

SNIP Encoder (frozen):
```
python train.py --reload_model ./weights/snip-1d-normalized.pth --freeze_encoder True [other parameters] 
```

SNIP Encoder (finetune):
```
python train.py --reload_model ./weights/snip-1d-normalized.pth --freeze_encoder False [other parameters] 
```

With this command, the model saves automatically every 10 epochs. To use SNIP's encoder, you should activate `--reload_model` parameter with the path of model weights. You can also freeze the encoder with `--freeze_encoder True`.


### Inference
To test how well your models perform for each property prediction task, use the `run_eval_proppred.sh` script. For example, if you want to test the NCR property task, use this command:
```
python eval_proppred.py --is_proppred True \
                        --property_type ncr \
                        --reload_model dump/NCR/model.pth \
                        --reload_data functions,dump/data/ncr/test.prefix,dump/data/ncr/test.prefix,
```
This command will use the `--reload_model` parameter to load the weights of your trained model and test it against the dataset specified in the `--reload_data` path.




## Using SNIP for Symbolic Regression
For the use of SNIP for more complex tasks such as Symbolic Regression (uncovering symbolic math equqations from data: numeric-to-symbolic generation task), check [here](https://github.com/deep-symbolic-mathematics/Multimodal-Symbolic-Regression).


## Citation
If you find the paper or the repo helpful, please cite it with
<pre>
@article{meidani2023snip,
  title={SNIP: Bridging Mathematical Symbolic and Numeric Realms with Unified Pre-training},
  author={Meidani, Kazem and Shojaee, Parshin and Reddy, Chandan K and Farimani, Amir Barati},
  journal={arXiv preprint arXiv:2310.02227},
  year={2023}
}
</pre>


## License 
This repository is licensed under MIT licence.

This work is built on top of other open source projects, including [Deep Learning for Symbolic Mathematics](https://github.com/facebookresearch/SymbolicMathematics) and [Contrastive Language-Image Pretraining](https://github.com/openai/CLIP). We thank the original contributors of these works for open-sourcing their valuable source codes.


## Contact Us
For any questions or issues, you are welcome to open an issue in this repo, or contact us at parshinshojaee@vt.edu, and mmeidani@andrew.cmu.edu .