import argparse

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

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
        #print( legal_move_mask.shape )

        #print( subsamples )
        a = sp_matrix_to_sp_tensor(a)
        #print( type(a) )
        #print( type(a).__mro__ )
        assert tf.keras.backend.is_sparse( a )
        return [x, a, e, i, legal_move_mask], np.vstack( Os )

    def on_epoch_end(self):
        random.shuffle( self.samples )

def build_model( nconv: int, compile: bool ):
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

    #X = BatchNormalization()(X)
    #E = BatchNormalization()(E)

    #X = Dense( Fh, activation='relu' )(X)
    #E = Dense( Sh, activation='relu' )(E)

    #print( X, A, E, I )

    for i in range( 0, nconv ):
        if( i == nconv-1 ):
            edge_channels=0
        else:
            edge_channels=Sh

        #'''
        X, E = XENetConv( stack_channels=[Skh,Skh],
                          node_channels=Fh, edge_channels=edge_channels,
                          node_activation='relu', edge_activation='relu',
                          attention=True )( [X, A, E] )
        #'''
        #X = ECCConv( Fh, activation='relu' )( [ X, A, E ] )

    #X = Dense( Fh, activation='relu' )(X)
    X = Dense( 1, activation='softplus' )(X)
    #X = Multiply()([X,L])
    #X = unsorted_segment_softmax( X, I )
    #X = Multiply()([X,L])

    Out = X

    model = Model( inputs=[X_in,A_in,E_in,I_in,L_in], outputs=Out )
    if compile:
        model.compile( optimizer="adam", loss="binary_crossentropy" )

    return model

def train_by_fit( model, training_loader, validation_loader ):
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
    '''
    test_x = training_loader[0][0]
    test_y = training_loader[0][1]

    for i in range( 0, 5 ):
        print( i, test_x[i].shape )
    print( test_y.shape, sum(test_y) )
    '''

def train_by_hand( model, training_loader, validation_loader ):
    #################
    # Config
    #################
    learning_rate = 1e-5  # Learning rate
    optimizer = Adam( learning_rate )
    loss_fn = BinaryCrossentropy()

    # see spektral/examples/graph_prediction/ogbg-mol-hiv_ecc.py
    def train_step(inputs, target):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            #print( target, predictions )
            #print( "TARGET", target )
            #print( "PREDICTIONS", predictions )
            #print( "!!!", loss_fn(target, predictions) )
            loss = loss_fn(target, predictions) + sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss


    for epoch in range( 0, 1 ):
        loss = 0
        step = 0
        for batch in training_loader:
            inp = batch[0]
            a_dense = tf.sparse.to_dense( inp[1] ).numpy()
            #print( a_dense )
            for i in range( 0, 5 ):
                if i == 1: continue
                print( i, np.any(np.isnan(inp[i])) )

            for i in range( 0, len(inp[2][0])):
                print( i, np.any(np.isnan(inp[2][:][i])) )

            outp = batch[1]
            print( np.any(np.isnan(outp)) )

            loss += train_step(*batch)
            step += 1
            #print( loss )
            print( step, "/" , len(training_loader), " ... ", loss.numpy()/step )
            exit( 0 )
            if step == 10:
                break
        print("Loss: {}".format(loss / step))

    x = model( test_x )

    #print( x )
    x = x.numpy()
    for i in range( 0, len(x) ):
        print( x[i][0], test_x[-1][i], test_x[-2][i], test_y[i][0] )
    exit( 0 )

    #history = model.fit( x=test_x, y=test_y, epochs=1000, shuffle=False, callbacks=callbacks, batch_size=32 )
    

if __name__ == '__main__':
    #test()

    parser = argparse.ArgumentParser()
    parser.add_argument( "--model", help="Where should we save the output model?", required=True, type=str )
    parser.add_argument( "--nconv", help="How Many XENet layers should we have?", required=True, type=int )
    args = parser.parse_args()


    training_loader = Loader( "data/training_data.txt", batch_size=1 )
    validation_loader = Loader( "data/validation_data.txt" )

    '''
    print( len( validation_loader ) )
    inps, outs = validation_loader[0]
    for i in range( 0, 4 ):
        print( i, inps[i].shape )
    print( outs.shape, sum(outs) )
    '''

    model = build_model( args.nconv, compile=False )
    model.summary()

    train_by_hand( model, training_loader, validation_loader )

    model.save( args.model )
