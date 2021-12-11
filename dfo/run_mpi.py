import argparse

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from tqdm import tqdm
import time

import numpy as np

import robots_core
from robots_core import MoveResult
#from robots_core.train import Tensors, make_tensors

import random

from scipy import sparse as sp

import spektral
from spektral.data.utils import to_disjoint
from spektral.utils.sparse import sp_matrix_to_sp_tensor
from spektral.layers.ops.scatter import unsorted_segment_softmax
from spektral.layers import XENetConv, CrystalConv, ECCConv
from spektral.data.loaders import DisjointLoader
from spektral.data.dataset import Dataset
from spektral.data.dataset import Graph

import sys

from model_play_game import play

import nevergrad as ng

#####
# MPI
from mpi4py import MPI

def send_bundle_to_node( comm, bundle, node ):
    comm.send( bundle, dest=node )

def get_next_message( comm ):
    status = MPI.Status()
    bundle = comm.recv( source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status )
    source = status.Get_source()
    return bundle, source




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

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument( "--model", help="Where should we save the output model?", required=True, type=str )

    parser.add_argument( "--opt", help="Which optimizer should we use?", default="NelderMead", type=str )

    parser.add_argument( "--just_print_layers", dest='just_print_layers', action='store_true')
    parser.set_defaults( just_print_layers=False )

    parser.add_argument('--layers', nargs="*")

    parser.add_argument('--scale_coeff', default=10, type=float )
    parser.add_argument('--model_dir', required=True, type=str )

    return parser.parse_args()

def load_model_from_disk( model_name ):
    custom_objects = { "XENetConv": XENetConv }
    model = load_model( model_name, custom_objects=custom_objects )
    return model

def run_head( args, comm, nprocs ):
    model = load_model_from_disk( args.model )
    inst = instrumentation( model, args.layers )
    opt = ng.optimizers.registry[ args.opt ]( parametrization=inst, budget=10000, num_workers=1 )

    t0 = time.time()

    score_for_iter0 = None

    for iter in range( 0, 999999 ):
        sample = opt.ask()
        x = sample[1]

        # Talk
        for i in range( 1, nprocs ):
            send_bundle_to_node( comm, x, i )

        # Listen
        running_score = float(0.0)
        for i in range( 1, nprocs ):
            score, source = get_next_message( comm )
            assert( isinstance( score, float ) )
            running_score += score
        running_score = running_score / float(nprocs-1)

        if score_for_iter0 == None:
            assert( iter == 0 )
            score_for_iter0 = running_score
        elif running_score < score_for_iter0:
            model.save( "{}/iter_{}.h5".format( args.model_dir, iter ) )


        opt.tell( sample, running_score )
        print( "HEAD", iter, time.time()-t0, running_score, "TODO save all this data" )
        sys.stdout.flush()

    # Clean up
    # https://github.com/facebookresearch/nevergrad/issues/180
    del opt



def run_scorer( args, comm, head_node_rank ):

    while True:

        # Listen
        x, source = get_next_message( comm )
        assert( source == head_node_rank )

        # load and assign fresh model
        model = load_model_from_disk( args.model )
        for w in model.trainable_weights:
            if w.name in x:
                w.assign( w + ( args.scale_coeff * x[ w.name ].value ) )
    
        # Score
        nloop = 5
        score = 0.0
        scores = []
        for _ in range( 0, nloop ):
            round, n_tele, result = play( model, verbose=False )
            if result == MoveResult.YOU_WIN_GAME:
                round += 1
            scores.append( round )
            score -= float(round)
        score = float(score)/float(nloop)

        #print( "WORKER", scores, score )
        sys.stdout.flush()

        # Talk
        send_bundle_to_node( comm, score, head_node_rank )

if __name__ == '__main__':

    # mpi
    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    rank = comm.Get_rank()
    head_node_rank = 0

    # args
    args = get_args()

    # load
    if args.just_print_layers:
        if rank == 0:
            print_layers( model )
        exit( 0 )

    if rank == head_node_rank:
        run_head( args, comm, nprocs )
    else:
        run_scorer( args, comm, head_node_rank )

