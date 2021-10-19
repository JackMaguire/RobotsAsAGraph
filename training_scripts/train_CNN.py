#import glob

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import argparse

import numpy as np

import sys
    
def make_model():

    F = 9 # TODO
    input = Input( shape=(45, 30, F ) )
    L = input

    L = Dense( 16 )( L )
    L = Conv2D(32, kernel_size=3, strides=1, padding="valid" )( L )
    L = Conv2D(32, kernel_size=3, strides=1, padding="valid" )( L )
    #L = Conv2D(32, kernel_size=3, strides=1, padding="valid" )( L )


    output = Dense( 10, activation='softmax' )( L )

    model = Model(inputs=input, outputs=output)
    model.compile( optimizer='adam', loss='categorical_crossentropy' )

    return model

if __name__ == '__main__':
    model = make_model()
    model.summary()
