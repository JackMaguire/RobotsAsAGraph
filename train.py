import argparse

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import numpy as np

import robots_core
from robots_core.train import Tensors, make_tensors

import random

from scipy import sparse as sp
from spektral.data.utils import to_disjoint
from spektral.utils.sparse import sp_matrix_to_sp_tensor
from spektral.layers.ops.scatter import unsorted_segment_softmax
from spektral.layers import XENetConv
from spektral.data.loaders import DisjointLoader
from spektral.data.dataset import Dataset
from spektral.data.dataset import Graph

import spektral

def test():
    t = make_tensors( "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001100000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000002000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000010000000010000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000,0,1,8" )
    print( t.input_tensors )
    print( t.output_tensor )

'''
class RobotsDataset( Dataset ):

    def __init__( self, filename ):
        with open( filename ) as file:
            self.samples = [line.rstrip() for line in file.readlines() if len(line) > 5]
        print( "Loaded {} lines".format( len(self.samples) ) )

        Dataset.__init__( self )

    def read( self ):
        return [ Graph( t.x, t.a, t.e, t.y ) ]
'''

class Loader( tf.keras.utils.Sequence ):
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
        a = sp_matrix_to_sp_tensor(a)
        print( type(a) )
        print( type(a).__mro__ )
        assert tf.keras.backend.is_sparse( a )
        return [x, a, e, i, legal_move_mask], np.vstack( Os )

    def on_epoch_end(self):
        random.shuffle( self.samples )

def build_model( nconv: int ):
    F = 8
    S = 8

    Fh  = 16 # Hidden Node
    Sh  = 16 # Hidden Edge
    Skh = 32 # Hidden Stack

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

    '''
    X = BatchNormalization()(X)
    E = BatchNormalization()(E)

    X = Dense( Fh, activation='relu' )(X)
    E = Dense( Sh, activation='relu' )(E)

    #print( X, A, E, I )

    for i in range( 0, nconv ):
        if( i == nconv-1 ):
            edge_channels=0
        else:
            edge_channels=Sh

        X, E = XENetConv( stack_channels=[Skh,Skh],
                          node_channels=Fh, edge_channels=edge_channels,
                          node_activation='relu', edge_activation='relu',
                          attention=True )( [X, A, E] )

    '''
    X = Dense( Fh, activation='relu' )(X)
    X = Dense( 1, activation='softplus' )(X)
    X = Multiply()([X,L])
    #X = unsorted_segment_softmax( X, I )

    Out = X

    model = Model( inputs=[X_in,A_in,E_in,I_in,L_in], outputs=Out )
    model.compile( optimizer="adam", loss="binary_crossentropy" )

    return model

if __name__ == '__main__':
    #test()

    parser = argparse.ArgumentParser()
    parser.add_argument( "--model", help="Where should we save the output model?", required=True, type=str )
    parser.add_argument( "--nconv", help="How Many XENet layers should we have?", required=True, type=int )
    args = parser.parse_args()


    training_loader = Loader( "data/training_data.txt" )
    validation_loader = Loader( "data/validation_data.txt" )

    '''
    print( len( validation_loader ) )
    inps, outs = validation_loader[0]
    for i in range( 0, 4 ):
        print( i, inps[i].shape )
    print( outs.shape, sum(outs) )
    '''

    model = build_model( args.nconv )
    model.summary()

    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=2,
        verbose=0,
        mode="auto",
        min_delta=0.001,
        cooldown=0,
        min_lr=0 )

    stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.002, patience=5, verbose=0, mode='min', baseline=None, restore_best_weights=True)

    callbacks=[lr_callback,stop]    

    #history = model.fit( x=training_loader, validation_data=validation_loader, epochs=1000, shuffle=False, callbacks=callbacks, batch_size=1 )
    #history = model.fit( x=training_loader, epochs=1000, shuffle=False, callbacks=callbacks, batch_size=1 )
    test_x = training_loader[0][0]
    test_y = training_loader[0][1]

    #test_x = [ np.expand_dims( x, axis=0 ) for x in test_x ]

    '''
    q = spektral.datasets.qm7.QM7()
    l = DisjointLoader( q )
    #i = q.read()
    samples = l.__next__()
    in_samples = samples[0]
    out_samples = samples[1]
    for i in in_samples:
        print( i.shape, type(i) )
    exit( 0 )

    print( test_x[ 0 ].shape )
    test_x[ 0 ] = test_x[ 0 ].reshape( (1, 846, 8) )#( (1, *test_x[ 0 ].shape) )

    for x in test_x:
        #x = x.reshape( (1,) + x.shape )
        print( x.shape, tf.keras.backend.is_sparse(x) )

    print( test_y.shape, tf.keras.backend.is_sparse(test_y) )

    '''
    for i in range( 0, 5 ):
        print( i, test_x[i].shape )
    print( test_y.shape, sum(test_y) )

    x = model( test_x )
    #print( x )
    x = x.numpy()
    for i in range( 0, len(x) ):
        print( x[i], test_x[-1][i], test_x[-2][i] )
    exit( 0 )

    #history = model.fit( x=test_x, y=test_y, epochs=1000, shuffle=False, callbacks=callbacks, batch_size=32 )
    

    model.save( args.model )
