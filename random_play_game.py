import argparse

import numpy as np

import robots_core
from robots_core import Board, RobotsGame, forecast_move
from robots_core.train import Tensors, make_tensors

from scipy import sparse as sp

import random

def maybe_recurse( game, recursion_depth: int ):
    return robots_core.strategy.run_recursive_seach( game, 7, recursion_depth );

def get_all_legal_moves( board : robots_core.Board ):
    legal_moves = []
    for dx in range( -1, 2 ):
        for dy in range( -1, 2 ):
            if forecast_move( board, dx, dy ).legal:
                legal_moves.append( ( dx, dy ) )
    return legal_moves

def random_move( game, recursion_depth ) -> bool:
    if maybe_recurse( game, recursion_depth ):
        return game.game_is_over()

    legal_moves = get_all_legal_moves( game.board() )
    if len(legal_moves) == 0:
        game_over = game.teleport()
    else:
        dx, dy = random.choice( legal_moves )
        game_over = game.move_human( dx, dy )
    
    return game_over

def maybe_cascade( game ) -> bool:
    if( game.board().move_is_cascade_safe( 0, 0 ) ):
        game_over = game.cascade()
        return game_over

def play( start_level: int = 1, stop_level: int = 999, recursion_depth: int = 0, verbose = True ):
    n_safe_tele = min( 10, start_level )
    game = RobotsGame( start_level, n_safe_tele )

    if verbose: print( game.round(), n_safe_tele )

    round = game.round()

    game_over = False
    while not game_over:
        if game.round() > round:
            round = game.round()
            if round == stop_level:
                break
            if verbose: print( "Starting round {} with {} safe teleports".format( round, game.n_safe_teleports_remaining() ) )

        maybe_cascade( game )
        game_over = random_move( game, recursion_depth=recursion_depth )

    if verbose: print( "FINAL", game.round(), game.n_safe_teleports_remaining(), game.latest_result() )

    return game.round(), game.n_safe_teleports_remaining(), game.latest_result()
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument( "--start_level", help="What level should we start at?", default=1, type=int )
    parser.add_argument( "--stop_level", help="What level should we stop at?", default=9999, type=int )
    parser.add_argument( "--recursion_depth", help="Recursion Depth?", default=0, type=int )
    args = parser.parse_args()

    play( args.start_level, args.stop_level, recursion_depth=args.recursion_depth )
