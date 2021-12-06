import argparse

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.losses import BinaryCrossentropy

#from tqdm import tqdm

import numpy as np

import robots_core
from robots_core import Board, RobotsGame
from robots_core.train import Tensors, make_tensors

from scipy import sparse as sp

import spektral
from spektral.data.utils import to_disjoint
from spektral.utils.sparse import sp_matrix_to_sp_tensor
from spektral.layers import XENetConv, CrystalConv, ECCConv

def move( game, model ) -> bool:
    #silly append pattern, but maximizes similarity to train.py
    Xs = []
    As = []
    Es = []
    Os = []

    dg = robots_core.graph.DenseGraph( game )

    Xs.append( np.asarray( dg.x ) )
    As.append( np.asarray( dg.a ) )
    Es.append( np.asarray( dg.e ) )
    
    x, a, e, i = to_disjoint( Xs, [sp.coo_matrix(a) for a in As], Es )

    legal_move_mask = x[:, -1]
    a = sp_matrix_to_sp_tensor(a)

    assert tf.keras.backend.is_sparse( a )
    inp = [x, a, e, i, legal_move_mask]
    pred = model( inp )
    #print( pred )

    pred = pred.numpy()[:,0]
    pred_node_i = np.argmax( pred )
    #print( pred.shape, pred_node_i )

    pred_node = dg.cached_nodes[ pred_node_i ]
    pred_node_int = int(pred_node.special_case)
    #print( pred_node.special_case, pred_node_int )

    tele = pred_node_int > 9 # TODO
    if tele:
        game_over = game.teleport()
    else:
        #print( pred_node.dx(), pred_node.dy() )
        game_over = game.move_human( pred_node.dx(), pred_node.dy() )
    
    return game_over

def maybe_cascade( game ) -> bool:
    if( game.board().move_is_cascade_safe( 0, 0 ) ):
        game_over = game.cascade()
        return game_over

def play( model, start_level: int ):
    n_safe_tele = min( 10, start_level )
    game = RobotsGame( start_level, n_safe_tele )

    print( game.round(), n_safe_tele )

    round = game.round()

    game_over = False
    while not game_over:
        if game.round() > round:
            round = game.round()
            print( "Starting round {} with {} safe teleports".format( round, game.n_safe_teleports_remaining() ) )

        maybe_cascade( game )
        game_over = move( game, model )

    print( game.round(), n_safe_tele, game.latest_result() )
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument( "--model", help="Where should we save the output model?", required=True, type=str )
    parser.add_argument( "--start_level", help="What level should we start at?", default=1, type=int )
    args = parser.parse_args()

    custom_objects = { "XENetConv": XENetConv }
    model = load_model( args.model, custom_objects=custom_objects )
    play( model, args.start_level )
