#import glob

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import argparse

import numpy as np

import sys
    
def make_model():

    XSize = 48
    ESize = 32

    F = 9 # TODO
    input = Input( shape=(45, 30, InputBuilder.NCnnInput ) )

    

    output = Dense( 10, activation='softmax' )( L )

    model = Model(inputs=input, outputs=output)
    model.compile( optimizer=adam, loss='categorical_crossentropy' )

    return model
