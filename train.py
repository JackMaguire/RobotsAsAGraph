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

def test():
    t = make_tensors( "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001100000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000002000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000010000000010000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000,0,1,8" )
    print( t.input_tensors )
    print( t.output_tensor )

def better_unsorted_segment_softmax(x, indices, mask):
    n_nodes = tf.reduce_max(indices)+1
    mask = tf.reshape( mask, tf.shape(x) )
    #print( mask.shape )
    #print( tf.exp( x ).shape )
    e_x = mask * tf.exp(
        x - tf.gather(tf.math.unsorted_segment_max(x, indices, n_nodes), indices)
    )
    e_x /= tf.gather(
        tf.math.unsorted_segment_sum(e_x, indices, n_nodes) + 1e-9, indices
    )
    return e_x

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
    def __init__( self, filename, batch_size=64, shuffle=True ):
        with open( filename ) as file:
            self.samples = [line.rstrip() for line in file.readlines() if len(line) > 5]
        print( "Loaded {} lines".format( len(self.samples) ) )

        self.batch_size = batch_size
        self.n_batches = int( len(self.samples) / batch_size )

        self.shuffle = shuffle

        self.on_epoch_end()

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
        if self.shuffle:
            random.shuffle( self.samples )

def build_model( nconv: int, compile: bool ):
    F = 10
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

        #'''
        X, E = XENetConv( stack_channels=[Skh,Skh],
                          node_channels=Fh, edge_channels=edge_channels,
                          node_activation='relu', edge_activation='relu',
                          attention=True )( [X, A, E] )
        #'''
        #X = ECCConv( Fh, activation='relu' )( [ X, A, E ] )

    X = Dense( Fh, activation='relu' )(X)
    X = Dense( 1, activation=None )(X)
    #X = Multiply()([X,L]) #Not Needed
    X = better_unsorted_segment_softmax( X, I, L )
    #X = Multiply()([X,L]) #Not Needed

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
def preview_preds( model, validation_loader, p ):
    test_x, test_y = validation_loader[0]
    x = model( test_x )

    data = []

    #print( x )
    x = x.numpy()
    for i in range( 0, len(x) ):
        if( test_y[i][0] > 0.5 ):
            if p:
                print( x[i][0], test_x[-1][i], test_x[-2][i], test_y[i][0] )
            else:
                data.append( [ x[i][0], test_x[-1][i], test_x[-2][i], test_y[i][0] ] )

    return np.asarray( data )

def evaluate_model( model, validation_loader, loss_fn ):
    y_true = []
    y_pred = []
    for batch in validation_loader:
        inputs, target = batch
        p = model(inputs, training=False)
        y_true.append(target)
        y_pred.append(p.numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    return loss_fn(y_true, y_pred)


def train_by_hand( model, training_loader, validation_loader, model_name ):
    #################
    # Config
    #################
    learning_rate = 1e-4  # Learning rate
    optimizer = Adam( learning_rate )
    loss_fn = BinaryCrossentropy()

    # see spektral/examples/graph_prediction/ogbg-mol-hiv_ecc.py
    def train_step(inputs, target):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_fn(target, predictions) + sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        #print( "TRAIN LOSS", loss.numpy() )
        return loss


    min_loss = 9999
    best_weights = model.get_weights()

    epoch_for_min_loss = -1
    epoch_for_last_lr_decrease = epoch_for_min_loss
    

    for epoch in range( 0, 1000 ):
        loss = 0
        step = 0
        for batch in tqdm(training_loader):
            loss += train_step(*batch)
            step += 1
            #print( step, "/" , len(training_loader), " ... ", loss.numpy()/step )

        validation_loss = evaluate_model( model, validation_loader, loss_fn )
        #print( "VAL LOSS", validation_loss )
        validation_loss_for_logic = np.round( 2*validation_loss, 4 )
        validation_loss = np.round( validation_loss, 5 )

        print("Epoch: {0} Training Loss: {1:.5f} Validation Loss: {2:.5f}".format(epoch, loss.numpy() / step, np.round( validation_loss, 5 )))

        if epoch == 0:
            # Freeze batch norm
            for layer in model.layers:
                if layer.name.startswith( "batch_normalization" ):
                    assert isinstance( layer, BatchNormalization )
                    print( "FREEZING", layer.name )
                    layer.trainable = False
                    layer._per_input_updates = {}
                else:
                    assert not isinstance( layer, BatchNormalization )

        if validation_loss_for_logic < min_loss:
            print( "NEW MIN" )
            min_loss = validation_loss_for_logic
            best_weights = model.get_weights()
            epoch_for_min_loss = epoch
            model.save( model_name + ".checkpoint.h5" )
        else:
            if epoch - epoch_for_min_loss == 5:
                return
            elif epoch - epoch_for_min_loss >= 2 and epoch - epoch_for_last_lr_decrease >= 2:
                print( "DECREASING LR" )
                epoch_for_last_lr_decrease = epoch
                learning_rate = learning_rate / 10.0
                optimizer = Adam( learning_rate )

        model.set_weights( best_weights )
           

def test2():
    F = 10
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

    X = BatchNormalization()(X)
    E = BatchNormalization()(E)

    X = Dense( 1, activation='relu' )(X)
    E = Dense( 1, activation='relu' )(E)

    Out = better_unsorted_segment_softmax( X, I, L )
    #Out = unsorted_segment_softmax( X, I )
    model = Model( inputs=[X_in,A_in,E_in,I_in,L_in], outputs=Out )
    
    validation_loader = Loader( "data/validation_data.txt", shuffle = False, batch_size=4 )

    #t = validation_loader[0]
    inputs, target = validation_loader[0]
    #print( inputs )

    o = model( inputs )

    for i in range( 0, len(inputs[-1]) ):
        print( inputs[-2][i], inputs[-1][i], o[i][0].numpy() )
    #print( inputs[0] )
    #print( o )

if __name__ == '__main__':
    #test2()
    #exit( 0 )

    parser = argparse.ArgumentParser()
    parser.add_argument( "--model", help="Where should we save the output model?", required=True, type=str )
    parser.add_argument( "--nconv", help="How Many XENet layers should we have?", required=True, type=int )
    args = parser.parse_args()


    training_loader = Loader( "data/training_data.txt" )
    validation_loader = Loader( "data/validation_data.txt", shuffle = False )

    model = build_model( args.nconv, compile=False )
    model.summary()

    train_by_hand( model, training_loader, validation_loader, args.model )
    data1 = preview_preds( model, validation_loader, p=False )

    model.save( args.model )

    custom_objects = { "XENetConv": XENetConv }
    model2 = load_model( args.model, custom_objects=custom_objects )
    data2 = preview_preds( model2, validation_loader, p=False )
    
    np.testing.assert_allclose( data1, data2 )
    print( "Save/Load test passed!" )
    
