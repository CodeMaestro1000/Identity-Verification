"""
Implementation of u-net model architecture
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dropout, MaxPool2D, Conv2DTranspose, concatenate

# implementation of convolutional_block for downsampling
def convolutional_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    """
      Convolutional downsampling block
    
      Arguments:
          inputs -- Input tensor
          n_filters -- Number of filters for the convolutional layers
          dropout_prob -- Dropout probability
          max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume
      Returns: 
          next_layer, skip_connection --  Next layer and skip connection outputs
      """
    # n_filters 3x3 convolutional layers with padding and relu activation
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)

    if dropout_prob > 0: # apply dropout if applicable
      conv = Dropout(dropout_prob)(conv)

    next_layer = MaxPool2D(2)(conv) if max_pooling else conv 

    skip_connection = conv
    return next_layer, skip_connection

def upsampling_block(expansive_input, contractive_input, n_filters=32):
    """
      Convolutional upsampling block
    
      Arguments:
          expansive_input -- Input tensor from previous layer
          contractive_input -- Input tensor from previous skip layer
          n_filters -- Number of filters for the convolutional layers
      Returns: 
          conv -- Tensor output
      """
    up = Conv2DTranspose(n_filters, 3, strides=(2,2), padding='same')(expansive_input)
    # Merge the previous output and the contractive_input
    merge = concatenate([up, contractive_input], axis=3)

    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)

    return conv

def unet_model(input_size=(128, 128, 3), n_filters=32, n_classes=1):
    """
    Unet model
    
      Arguments:
          input_size -- Input shape 
          n_filters -- Number of filters for the convolutional layers
          n_classes -- Number of output classes
      Returns: 
          model -- tf.keras.Model
    """
    # Encoder
    inputs = Input(input_size)
    conv_block1 = convolutional_block(inputs, n_filters)
    conv_block2 = convolutional_block(conv_block1[0], n_filters*2)
    conv_block3 = convolutional_block(conv_block2[0], n_filters*4)
    # Include a dropout_prob of 0.3 for this layer
    conv_block4 = convolutional_block(conv_block3[0], n_filters*8, dropout_prob=0.3) 
    # Include a dropout_prob of 0.3 for this layer, and avoid the max_pooling layer
    conv_block5 = convolutional_block(conv_block4[0], n_filters*16, dropout_prob=0.3, max_pooling=None)

    # Decoder
    # Use output_layer from conv_block5 and skip connection from conv_block4
    ublock6 = upsampling_block(conv_block5[0], conv_block4[1],  n_filters*8)
    ublock7 = upsampling_block(ublock6, conv_block3[1],  n_filters*4) # conv_block3 skip connection
    ublock8 = upsampling_block(ublock7, conv_block2[1],  n_filters*2)
    ublock9 = upsampling_block(ublock8, conv_block1[1],  n_filters)

    conv_block9 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(ublock9)
    conv_block10 = Conv2D(n_classes, 1, activation='sigmoid', padding='same')(conv_block9)
    model = tf.keras.Model(inputs=inputs, outputs=conv_block10)

    return model