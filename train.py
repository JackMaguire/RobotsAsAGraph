import numpy as np

import robots_core
from robots_core.train import Tensors, make_tensors

import random

from scipy import sparse as sp
from spektral.data.utils import to_disjoint
from spektral.layers.ops.scatter import unsorted_segment_softmax
from spektral.layers import XENetConv

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def test():
    t = make_tensors( "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001100000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000002000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000010000000010000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000,0,1,8" )
    print( t.input_tensors )
    print( t.output_tensor )

class Loader:
    def __init__( self, filename, batch_size=32 ):
        with open( filename ) as file:
            self.samples = [line.rstrip() for line in file.readlines() if len(line) > 5]
        print( "Loaded {} lines".format( len(self.samples) ) )

        self.batch_size = batch_size
        self.n_batches = int( len(self.samples) / batch_size )

    def __len__( self ):
        return self.n_batches

    def __getitem__( self, index ):
        subsamples = self.samples[ index*self.batch_size : (index+1)*self.batch_size ]

        Xs = []
        As = []
        Es = []
        Os = []

        for s in subsamples:
            t = make_tensors( s )
            Xs.append( np.asarray( t.input_tensors.x ) )
            As.append( np.asarray( t.input_tensors.a ) )
            Es.append( np.asarray( t.input_tensors.e ) )
            Os.append( np.asarray( t.output_tensor   ) )
            #print( np.asarray( t.output_tensor   ).shape )

        #print( np.asarray( Os ) )
        #print( np.asarray( Xs ).shape, "???" )
        #print( np.asarray( Os ).shape, "!!!" )

        x, a, e, i = to_disjoint( Xs, [sp.coo_matrix(a) for a in As], Es )

        legal_move_mask = x[:, -1]
        print( legal_move_mask.shape )

        #print( subsamples )
        return (x, a, e, i, legal_move_mask), np.vstack( Os )

    def on_epoch_end(self):
        random.shuffle( self.samples )

def build_model():
    F = 8
    S = 8

    Fh  = 16 # Hidden Node
    Sh  = 16 # Hidden Edge
    Skh = 16 # Hidden Stack

    X_in = Input( name="X_in", shape=(F,) )
    A_in = Input( name="A_in", shape=(None,), sparse=True )
    E_in = Input( name="E_in", shape=(S,) )
    I_in = Input( name="I_in", shape=(), dtype=tf.int32 )
    L_in = Input( name="L_in", shape=() ) #Legal move mask

    X = X_in
    A = A_in
    E = E_in
    I = I_in
    L = L_in

    X = BatchNormalization()(X)
    E = BatchNormalization()(E)

    X = Dense( Fh, activation='relu' )
    E = Dense( Sh, activation='relu' )

    print( X, A, E, I )

    nconv = 1
    for i in range( 0, nconv ):
        X, E = XENetConv( stack_channels=[Skh,Skh],
                          node_channels=Fh, edge_channels=Sh,
                          node_activation='relu', edge_activation='relu',
                          attention=True )( [X, A, E, I] )

    X = Dense( Fh, activation='relu' )

    X = Multiply()([X,L])
    X = unsorted_segment_softmax( X, I )

    model = Model( inputs=inp, outputs=out )
    model.compile( optimizer="adam", loss="binary_crossentropy" )

    return model

if __name__ == '__main__':
    #test()

    training_loader = Loader( "data/training_data.txt" )
    validation_loader = Loader( "data/validation_data.txt" )

    print( len( validation_loader ) )
    inps, outs = validation_loader[0]
    for i in range( 0, 4 ):
        print( i, inps[i].shape )
    print( outs.shape, sum(outs) )

    model = build_model()
    model.summary()
