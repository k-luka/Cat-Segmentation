import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, BatchNormalization, Dropout, MaxPooling2D,
    concatenate
)
from tensorflow.keras.models import Model

def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
    conv = Conv2D(
        n_filters, 3, activation='relu', padding='same', kernel_initializer='HeNormal'
    )(inputs)
    conv = Conv2D(
        n_filters, 3, activation='relu', padding='same', kernel_initializer='HeNormal'
    )(conv)
    conv = BatchNormalization()(conv, training=False)

    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)

    if max_pooling:
        next_layer = MaxPooling2D(pool_size=(2, 2))(conv)
    else:
        next_layer = conv

    skip_connection = conv
    return next_layer, skip_connection

def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32):
    up = Conv2DTranspose(
        n_filters, (3, 3), strides=(2, 2), padding='same'
    )(prev_layer_input)
    merge = concatenate([up, skip_layer_input], axis=3)

    conv = Conv2D(
        n_filters, 3, activation='relu', padding='same', kernel_initializer='HeNormal'
    )(merge)
    conv = Conv2D(
        n_filters, 3, activation='relu', padding='same', kernel_initializer='HeNormal'
    )(conv)

    return conv

def UNetCompiled(input_size=(128, 128, 3), n_filters=32, n_classes=3):
    """
    Combine both encoder and decoder blocks according to the U-Net research paper
    Return the model as output 
    """
    inputs = Input(input_size)

    # Encoder
    cblock1 = EncoderMiniBlock(inputs, n_filters, dropout_prob=0, max_pooling=True)
    cblock2 = EncoderMiniBlock(cblock1[0], n_filters*2, dropout_prob=0, max_pooling=True)
    cblock3 = EncoderMiniBlock(cblock2[0], n_filters*4, dropout_prob=0, max_pooling=True)
    cblock4 = EncoderMiniBlock(cblock3[0], n_filters*8, dropout_prob=0.3, max_pooling=True)
    cblock5 = EncoderMiniBlock(cblock4[0], n_filters*16, dropout_prob=0.3, max_pooling=False)

    # Decoder
    ublock6 = DecoderMiniBlock(cblock5[0], cblock4[1],  n_filters*8)
    ublock7 = DecoderMiniBlock(ublock6,     cblock3[1], n_filters*4)
    ublock8 = DecoderMiniBlock(ublock7,     cblock2[1], n_filters*2)
    ublock9 = DecoderMiniBlock(ublock8,     cblock1[1], n_filters)

    # Final Convolution
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='HeNormal')(ublock9)
    output_layer = Conv2D(n_classes, 1, padding='same')(conv)

    # Define the model
    model = Model(inputs=inputs, outputs=output_layer)
    return model