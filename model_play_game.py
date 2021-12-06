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


def call_make_tensors( game ):
    # TODO replace with actual method once we can recompile
    s = ','.join( str(i) for i in [
        game.board().get_stringified_representation(),
        game.n_safe_teleports_remaining(),
        game.round(),
        1
    ])
    return make_tensors( s )


def move( game, model ) -> bool:
    #silly append pattern, but maximizes similarity to train.py
    Xs = []
    As = []
    Es = []
    Os = []

    t = call_make_tensors( game )

    Xs.append( np.asarray( t.input_tensors.x ) )
    As.append( np.asarray( t.input_tensors.a ) )
    Es.append( np.asarray( t.input_tensors.e ) )
    
    x, a, e, i = to_disjoint( Xs, [sp.coo_matrix(a) for a in As], Es )

    legal_move_mask = x[:, -1]
    a = sp_matrix_to_sp_tensor(a)

    assert tf.keras.backend.is_sparse( a )
    inp = [x, a, e, i, legal_move_mask]
    pred = model( inp )
    print( pred )

    pred = pred.numpy()[:,0]
    pred_node_i = np.argmax( pred )
    print( pred.shape, pred_node_i )

    pred_node = t.input_tensors.cached_nodes[ pred_node_i ]
    print( int(pred_node.special_case) )

def play( model, start_level: int ):
    n_safe_tele = min( 10, start_level )
    game = RobotsGame( start_level, n_safe_tele )

    print( game.round(), n_safe_tele )
    move( game, model )

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument( "--model", help="Where should we save the output model?", required=True, type=str )
    parser.add_argument( "--start_level", help="What level should we start at?", default=1, type=int )
    args = parser.parse_args()

    custom_objects = { "XENetConv": XENetConv }
    model = load_model( args.model, custom_objects=custom_objects )
    play( model, args.start_level )
