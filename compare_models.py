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
        
