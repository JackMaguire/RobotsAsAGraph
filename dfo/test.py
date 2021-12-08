import argparse

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from tqdm import tqdm

import numpy as np

import robots_core
from robots_core.train import Tensors, make_tensors

import random

from scipy import sparse as sp
from spektral.data.utils import to_disjoint
from spektral.utils.sparse import sp_matrix_to_sp_tensor
from spektral.layers.ops.scatter import unsorted_segment_softmax
from spektral.layers import XENetConv, CrystalConv, ECCConv
from spektral.data.loaders import DisjointLoader
from spektral.data.dataset import Dataset
from spektral.data.dataset import Graph

import spektral

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument( "--model", help="Where should we save the output model?", required=True, type=str )
    #parser.add_argument( "--start_level", help="What level should we start at?", default=1, type=int )
    #parser.add_argument( "--stop_level", help="What level should we stop at?", default=9999, type=int )
    args = parser.parse_args()

    # load
    custom_objects = { "XENetConv": XENetConv }
    model = load_model( args.model, custom_objects=custom_objects )

    for layer in model.layers:
        print( layer.name, len(layer.weights) )
        for l in layer.weights:
            print( l.name, l.shape )
    

'''
For one XENet layer:
X_in 0
E_in 0
batch_normalization 4
batch_normalization/gamma:0 (10,)
batch_normalization/beta:0 (10,)
batch_normalization/moving_mean:0 (10,)
batch_normalization/moving_variance:0 (10,)
batch_normalization_1 4
batch_normalization_1/gamma:0 (8,)
batch_normalization_1/beta:0 (8,)
batch_normalization_1/moving_mean:0 (8,)
batch_normalization_1/moving_variance:0 (8,)
dense 2
dense/kernel:0 (10, 16)
dense/bias:0 (16,)
A_in 0
dense_1 2
dense_1/kernel:0 (8, 16)
dense_1/bias:0 (16,)
xe_net_conv 13
xe_net_conv/dense/kernel:0 (64, 32)
xe_net_conv/dense/bias:0 (32,)
xe_net_conv/dense_1/kernel:0 (32, 32)
xe_net_conv/dense_1/bias:0 (32,)
xe_net_conv/p_re_lu/alpha:0 (32,)
xe_net_conv/dense_2/kernel:0 (80, 16)
xe_net_conv/dense_2/bias:0 (16,)
xe_net_conv/dense_3/kernel:0 (32, 0)
xe_net_conv/dense_3/bias:0 (0,)
xe_net_conv/dense_4/kernel:0 (32, 1)
xe_net_conv/dense_4/bias:0 (1,)
xe_net_conv/dense_5/kernel:0 (32, 1)
xe_net_conv/dense_5/bias:0 (1,)
I_in 0
dense_2 2
dense_2/kernel:0 (16, 16)
dense_2/bias:0 (16,)
tf.math.reduce_max 0
dense_3 2
dense_3/kernel:0 (16, 1)
dense_3/bias:0 (1,)
tf.__operators__.add 0
tf.math.unsorted_segment_max 0
tf.compat.v1.gather 0
L_in 0
tf.compat.v1.shape 0
tf.math.subtract 0
tf.reshape 0
tf.math.exp 0
tf.math.multiply 0
tf.math.unsorted_segment_sum 0
tf.__operators__.add_1 0
tf.compat.v1.gather_1 0
tf.math.truediv 0
'''
