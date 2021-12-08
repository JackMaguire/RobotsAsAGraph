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

def print_layers( model ):
    for w in model.trainable_weights:
        print( w.name, w.shape )
    

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument( "--model", help="Where should we save the output model?", required=True, type=str )
    parser.add_argument( "--just_print_layers", dest='just_print_layers', action='store_true')
    parser.set_defaults( just_print_layers=False )

    args = parser.parse_args()

    # load
    custom_objects = { "XENetConv": XENetConv }
    model = load_model( args.model, custom_objects=custom_objects )

    if args.just_print_layers:
        print_layers( model )
        exit( 0 )
