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

class Loader( tf.keras.utils.Sequence ):
    def __init__( self, filename ):
        with open( filename ) as file:
            self.samples = [line.rstrip() for line in file.readlines() if len(line) > 5]
        print( "Loaded {} lines".format( len(self.samples) ) )

        self.on_epoch_end()

    def __len__( self ):
        return len( self.samples )

    def __getitem__( self, index ):
        s = self.samples[ index ]

        Xs = []
        As = []
        Es = []
        Os = []

        t = make_tensors( s )
        Xs.append( np.asarray( t.input_tensors.x ) )
        As.append( np.asarray( t.input_tensors.a ) )
        Es.append( np.asarray( t.input_tensors.e ) )
        Os.append( np.asarray( t.output_tensor   ) )

        x, a, e, i = to_disjoint( Xs, [sp.coo_matrix(a) for a in As], Es )

        legal_move_mask = x[:, -1]

        a = sp_matrix_to_sp_tensor(a)

        assert tf.keras.backend.is_sparse( a )
        return [x, a, e, i, legal_move_mask], np.vstack( Os ), t

    def on_epoch_end(self):
        pass

def get_move_from_dense_graph( t, pred_node_i ):
    pred_node = t.input_tensors.cached_nodes[ pred_node_i ]
    pred_node_int = int(pred_node.special_case)
    return pred_node_int

    tele = pred_node_int > 9
    if tele:
        return True, 0, 0
    else:
        return False, pred_node.dx(), pred_node.dy()

if __name__ == '__main__':
    #test2()
    #exit( 0 )

    parser = argparse.ArgumentParser()
    parser.add_argument( "--model1", help="From where should we load the first model?", required=True, type=str )
    parser.add_argument( "--model2", help="From where should we load the second model?", required=True, type=str )
    parser.add_argument( "--data", help="From where should we load the data?", required=True, type=str )
    args = parser.parse_args()


    loader = Loader( args.data )

    custom_objects = { "XENetConv": XENetConv }
    model1 = load_model( args.model1, custom_objects=custom_objects )
    model2 = load_model( args.model2, custom_objects=custom_objects )

    count = 0

    for x, y, t in loader:
        count += 1
        #x, y = *i
        #print( y )
        #print( t )
        
        my_move_node = np.argmax( y.flatten() )
        my_move = get_move_from_dense_graph( t, my_move_node )

        pred1 = model1( x ).numpy()
        pred1_move_node = np.argmax( pred1.flatten() )
        pred1_move = get_move_from_dense_graph( t, pred1_move_node )

        pred2 = model2( x ).numpy()
        pred2_move_node = np.argmax( pred2.flatten() )
        pred2_move = get_move_from_dense_graph( t, pred2_move_node )

        move2_conf_model1 = pred1[pred2_move_node][0]
        move2_conf_model2 = pred2[pred2_move_node][0]
        move1_conf_model1 = pred1[pred1_move_node][0]
        move1_conf_model2 = pred2[pred1_move_node][0]

        print( my_move, pred1_move, pred2_move )
        print( move1_conf_model1, move1_conf_model2 )
        print( move2_conf_model2, move2_conf_model1 )

        if count > 10:
            exit( 0 )
        
