import argparse

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from tqdm import tqdm

import numpy as np

import robots_core
import robots_core as rc
import robots_core.visualization as rc_vis
from robots_core.train import *

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
        return [x, a, e, i, legal_move_mask], np.vstack( Os ), t, s

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

def get_specialcase_from_dense_graph( t, pred_node_i ):
    pred_node = t.input_tensors.cached_nodes[ pred_node_i ]
    return pred_node.special_case

def picture( s, pred1_move, pred2_move, fileprefix ):

    dp = deserialize( s )
    game = dp.game

    filename = "{}_{}.svg".format( fileprefix, game.n_safe_teleports_remaining() )

    vs = rc_vis.VisSettings()
    ma1 = rc_vis.MoveAnnotation()
    #print( ma1.type, pred1_move )
    #exit( 0 )
    ma1.type = pred1_move
    ma1.rgb = "0,0,0"
    #vs.moves.append( ma1 )
    vs.append_move(ma1)

    ma2 = rc_vis.MoveAnnotation()
    ma2.type = pred2_move
    ma2.rgb = "182,3,252"
    #vs.moves.append( ma2 )
    vs.append_move(ma2)

    vs.label_elements = False

    pic = rc_vis.to_svg_string( game.board(), vs )
    #print( pic )
    with open( filename, 'r' ) as f:
        f.write( pic )

    

if __name__ == '__main__':
    #test2()
    #exit( 0 )

    parser = argparse.ArgumentParser()
    parser.add_argument( "--model1", help="From where should we load the first model?", required=True, type=str )
    parser.add_argument( "--model2", help="From where should we load the second model?", required=True, type=str )
    parser.add_argument( "--data", help="From where should we load the data?", required=True, type=str )
    parser.add_argument( "--file_prefix", help="Where should we dump these pictures?", required=True, type=str )
    args = parser.parse_args()


    loader = Loader( args.data )

    custom_objects = { "XENetConv": XENetConv }
    model1 = load_model( args.model1, custom_objects=custom_objects )
    model2 = load_model( args.model2, custom_objects=custom_objects )

    count = 0
    for x, y, t, s in loader:
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

        if pred1_move != pred2_move:
            pred1_case = get_specialcase_from_dense_graph( t, pred1_move_node )
            pred2_case = get_specialcase_from_dense_graph( t, pred2_move_node )

            fileprefix = args.file_prefix + str(count)
            picture( s, pred1_case, pred2_case, fileprefix )
            #exit( 0 )
