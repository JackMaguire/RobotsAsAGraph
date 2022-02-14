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
from robots_core.train import *
import robots_core.visualization as rc_vis

from scipy import sparse as sp

import spektral
from spektral.data.utils import to_disjoint
from spektral.utils.sparse import sp_matrix_to_sp_tensor
from spektral.layers import XENetConv, CrystalConv, ECCConv

def maybe_recurse( game, recursion_depth: int ):
    return robots_core.strategy.run_recursive_seach( game, 7, recursion_depth );

def move( game, model, recursion_depth, movie_maker ) -> bool:
    if maybe_recurse( game, recursion_depth ):
        return game.game_is_over()

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

    movie_maker.capture_frame( game, pred_node.special_case )

    tele = pred_node_int > 9 # TODO
    if tele:
        game_over = game.teleport()
    else:
        #print( pred_node.dx(), pred_node.dy() )
        game_over = game.move_human( pred_node.dx(), pred_node.dy() )
    
    return game_over

def maybe_cascade( game, movie_maker ) -> bool:
    if( game.board().move_is_cascade_safe( 0, 0 ) ):
        game_over = game.cascade()
        return game_over

class MovieMaker():
    def __init__( self, movie_prefix ):
        self.movie_prefix = movie_prefix
        self.frame_counter = 0

    def capture_frame( self, game, move = None ):
        filename = "{}frame_{}.svg".format( self.movie_prefix, str(self.frame_counter).zfill(5) )

        vs = rc_vis.VisSettings()

        if move is not None:
            ma = rc_vis.MoveAnnotation()
            ma.type = move
            ma.rgb = "0,0,0"
            vs.append_move(ma)

        vs.label_elements = False

        pic = rc_vis.to_svg_string( game.board(), vs )
        with open( filename, 'w' ) as f:
            f.write( pic )

        self.frame_counter += 1


def play( model, start_level: int = 1, stop_level: int = 999, recursion_depth: int = 0, verbose = True, movie_prefix : str = "" ):
    n_safe_tele = min( 10, start_level )
    game = RobotsGame( start_level, n_safe_tele )

    movie_maker = MovieMaker( movie_prefix )

    if verbose: print( game.round(), n_safe_tele )

    round = game.round()

    game_over = False
    while not game_over:
        if game.round() > round:
            round = game.round()
            if round == stop_level:
                break
            if verbose: print( "Starting round {} with {} safe teleports".format( round, game.n_safe_teleports_remaining() ) )

        movie_maker.capture_frame( game )

        game_over = maybe_cascade( game, movie_maker )
        if game_over:
            break

        game_over = move( game, model, recursion_depth=recursion_depth, movie_maker=movie_maker )

    if verbose: print( "FINAL", game.round(), game.n_safe_teleports_remaining(), game.latest_result() )

    return game.round(), game.n_safe_teleports_remaining(), game.latest_result()
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument( "--model", help="Where should we save the output model?", required=True, type=str )
    parser.add_argument( "--start_level", help="What level should we start at?", default=1, type=int )
    parser.add_argument( "--stop_level", help="What level should we stop at?", default=9999, type=int )
    parser.add_argument( "--recursion_depth", help="Recursion Depth?", default=0, type=int )
    parser.add_argument( "--movie_prefix", help="If given, where should we save the movie?", default="", type=str )
    args = parser.parse_args()

    custom_objects = { "XENetConv": XENetConv }
    model = load_model( args.model, custom_objects=custom_objects )
    play( model, args.start_level, args.stop_level, recursion_depth=args.recursion_depth, movie_prefix=args.movie_prefix )
