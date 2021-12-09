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

from model_play_game import play

import nevergrad as ng

def print_layers( model ):
    for w in model.trainable_weights:
        print( w.name, w.shape )
    
def instrumentation( model, layer_names ):
    print( layer_names )
    print( len(layer_names) )

    data = {}

    for w in model.trainable_weights:
        print( w.name, w.shape ) 
        #print( w.numpy() )
        #w2 = w.numpy() + 1
        #print( w2 )
        #exit( 0 )
        if w.name in layer_names:
            shape = [ i for i in w.shape ]
            print( w.shape, shape )
            data[ w.name ] = ng.p.Array( shape=shape )

    return ng.p.Instrumentation( **data )

def run_single_loop( opt, model, scale_coeff ):
    sample = opt.ask()
    x = sample[1]

    print( x )
    # TODO reset model

    # the slow method...
    for w in model.trainable_weights:
        if w.name in x:
            #print( w.name, type(w) )
            #print( w, x[ w.name ].value )
            w.assign( w + ( scale_coeff * x[ w.name ].value ) )
            #print( w )
            
    round, n_tele, result = play( model )

    opt.tell( sample, -1 )

    return

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument( "--model", help="Where should we save the output model?", required=True, type=str )

    parser.add_argument( "--just_print_layers", dest='just_print_layers', action='store_true')
    parser.set_defaults( just_print_layers=False )

    parser.add_argument('--layers', nargs="*")

    parser.add_argument('--scale_coeff', default=10, type=float )

    return parser.parse_args()



if __name__ == '__main__':
    # args
    args = get_args()

    # load
    custom_objects = { "XENetConv": XENetConv }
    model = load_model( args.model, custom_objects=custom_objects )

    if args.just_print_layers:
        print_layers( model )
        exit( 0 )

    #play( model )

    inst = instrumentation( model, args.layers )

    #optname = "RealSpacePSO"
    optname = "NelderMead"

    opt = ng.optimizers.registry[ optname ]( parametrization=inst, budget=10000, num_workers=1 )

    for _ in range( 0, 10 ):
        run_single_loop( opt, model, args.scale_coeff )

    # https://github.com/facebookresearch/nevergrad/issues/180
    del opt
    exit( 0 )
