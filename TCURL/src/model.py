
import tensorflow as tf
from tensorflow.keras.layers import Layer, Reshape, Activation, Conv1D, Conv1DTranspose, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Input, Add, Concatenate, Embedding,LeakyReLU,Dense, BatchNormalization, Flatten, LayerNormalization, LSTM, ReLU, Bidirectional, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant
import numpy as np
# from keras.initializers import RandomNormal
from functools import partial


class PositionEncoding(Layer):
    def __init__(self,**kwargs):
        super(PositionEncoding,self).__init__(**kwargs)
    def build(self,input_shape):
        def get_position_encoding(seq_len,model_dim):
            position_encoding=np.zeros(shape=(seq_len,model_dim))
            for pos in range(seq_len):
                for i in range(model_dim):
                    position_encoding[pos,i]=pos/(np.power(10000,2*i/model_dim))
            position_encoding[::,::2]=np.sin(position_encoding[::,::2])
            position_encoding[::,1::2]=np.cos(position_encoding[::,1::2])
            return np.expand_dims(position_encoding,axis=0)
        seq_len,model_dim=input_shape.as_list()[1:3]
        self.position_encoding=self.add_weight(
            shape=(1,seq_len,model_dim),
            initializer=Constant(get_position_encoding(seq_len,model_dim)),
            trainable=False,
            name="position_encoding"
        )
        super(PositionEncoding,self).build(input_shape)

    def call(self,inputs):
        return self.position_encoding

# class PaddingMask(Layer):
#     """Split the input tensor into 2 tensors along the time dimension."""
#
#     def __init__(self):
#         super(PaddingMask, self).__init__()
#     def call(self, inputs):
#         # Expect the input to be 3D and mask to be 2D, split the input tensor into 2
#         # subtensors along the time axis (axis 1).
#         return tf.split(inputs, 2, axis=1)
#
#     def compute_mask(self, inputs, mask=None):
#         inputs =
#         return tf.split(mask, 2, axis=1)


# def create_padding_mask(seq):
#      seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
#      seq1 = tf.linalg.matmul(
#          [seq],
#          [seq],
#          transpose_a=True,
#          transpose_b=False,
#          adjoint_a=False,
#          adjoint_b=False,
#          a_is_sparse=False,
#          b_is_sparse=False,
#      )
#      seq1 = Add()([seq1, seq])
#      seq_t = tf.transpose(seq1)
#      # print(seq_t)
#      seq = Add()([seq1, seq_t])
#      # seq = tf.ones((200,200))
#      return seq  # (batch_size, 1, 1, seq_len)

def create_padding_mask_1(seq):
    # x = tf.compat.v1.placeholder(dtype=tf.float32, shape = (None, seq.shape[1], seq.shape[1]))
    seq1 = tf.cast(tf.math.equal(seq, 0), tf.float32)
    seq = tf.cast(tf.math.greater(seq, 3), tf.float32)
    mask = tf.linalg.matmul(
        [seq],
        [seq],
        transpose_a=True,
        transpose_b=False,
        adjoint_a=False,
        adjoint_b=False,
        a_is_sparse=False,
        b_is_sparse=False,
    )
    print(mask.shape)
    mask = Add()([mask, seq1])
    mask = Add()([mask, tf.transpose(mask,perm=[0,2,1])])
    mask = tf.cast(tf.math.greater(mask, 0), tf.float32)
    return mask  # (batch_size, 1, 1, seq_len)

def create_padding_mask_2(seq):
    seq1 = tf.cast(tf.math.equal(seq, 0), tf.float32)
    seq = tf.cast(tf.math.greater(seq, 3), tf.float32)
    mask = tf.linalg.matmul(
        [seq],
        [seq],
        transpose_a=True,
        transpose_b=False,
        adjoint_a=False,
        adjoint_b=False,
        a_is_sparse=False,
        b_is_sparse=False,
    )
    mask = Add()([mask, seq1])
    # mask = Add()([mask, seq])
    # seq1 = tf.linalg.matmul(
    #     [seq],
    #     [seq],
    #     transpose_a=True,
    #     transpose_b=False,
    #     adjoint_a=False,
    #     adjoint_b=False,
    #     a_is_sparse=False,
    #     b_is_sparse=False,
    # )
    return mask  # (batch_size, 1, 1, seq_len)

def create_padding_mask_3(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    seq1 = tf.linalg.matmul(
        [seq],
        [seq],
        transpose_a=True,
        transpose_b=False,
        adjoint_a=False,
        adjoint_b=False,
        a_is_sparse=False,
        b_is_sparse=False,
    )
    return seq1  # (batch_size, 1, 1, seq_len)

def create_padding_mask_4(seq):
    seq1 = tf.cast(tf.math.equal(seq, 0), tf.float32)
    seq2 = tf.linalg.matmul(
        [seq1],
        [seq1],
        transpose_a=True,
        transpose_b=False,
        adjoint_a=False,
        adjoint_b=False,
        a_is_sparse=False,
        b_is_sparse=False,
    )
    seq = tf.cast(tf.math.equal(seq, 1), tf.float32)
    seq = Add()([seq2, seq])
    return seq  # (batch_size, 1, 1, seq_len)

def create_padding_mask_5(seq):
    seq1 = tf.cast(tf.math.equal(seq, 0), tf.float32)
    seq2 = tf.cast(tf.math.greater(seq, 3), tf.float32)
    mask = tf.linalg.matmul(
        [seq2],
        [seq2],
        transpose_a=True,
        transpose_b=False,
        adjoint_a=False,
        adjoint_b=False,
        a_is_sparse=False,
        b_is_sparse=False,
    )
    mask = Add()([mask, seq1])
    seq = tf.cast(tf.math.equal(seq, 1), tf.float32)
    seq1 = tf.linalg.matmul(
        [seq],
        [seq],
        transpose_a=True,
        transpose_b=False,
        adjoint_a=False,
        adjoint_b=False,
        a_is_sparse=False,
        b_is_sparse=False,
    )
    # print(seq1)
    mask = Add()([mask, seq1])
    return mask  # (batch_size, 1, 1, seq_len)
# x = tf.constant([[0,0,0,1,1]])
# seq1 = create_padding_mask_1(x)
# seq2 = create_padding_mask_2(x)
# seq3 = create_padding_mask_3(x)
# seq4 = create_padding_mask_4(x)
# seq5 = create_padding_mask_5(x)
# print(seq1,seq2,seq3, seq4, seq5)

def create_look_ahead_mask(size):
  mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)
#
# ma = create_look_ahead_mask(10)
# print(ma)

def transformer_decoder(x, enc_output, mask_size, padding_mask):

    look_ahead_mask = create_look_ahead_mask(mask_size)
    if padding_mask != None:
        # look_ahead_mask = tf.multiply(padding_mask, look_ahead_mask)
        look_ahead_mask = Add()([look_ahead_mask, padding_mask])
    attn1 = MultiHeadAttention(num_heads=8, key_dim=64, dropout=0.2)(x, x, x, look_ahead_mask)
    x = Add()([x, attn1])
    out1 = LayerNormalization(epsilon=1e-6)(x)

    attn2 = MultiHeadAttention(num_heads=8, key_dim=64, dropout=0.2)(out1, enc_output, enc_output, padding_mask)
    out2 = Add()([attn2, out1])
    out2 = LayerNormalization(epsilon=1e-6)(out2)

    x = Dense(128)(out2)
    x = ReLU()(x)
    x = Dropout(0.2)(x)
    x = Dense(64)(x)
    x = ReLU()(x)
    x = Dropout(0.2)(x)
    out3 = Add()([out2, x])
    out3 = LayerNormalization(epsilon=1e-6)(out3)
    return out3

# input = tf.random.uniform((5, 200, 64))
# lstm = LSTM(300, return_sequences=True)
# out = lstm(input)
# print(out)


def transformer_encoder(inputs, out_num, padding_mask):
    x = inputs
    x1 = MultiHeadAttention(num_heads=8, key_dim=64, dropout=0.2)(x, x, x, padding_mask)
    x2 = Add()([x, x1])
    x = LayerNormalization(epsilon=1e-6)(x2)
    x = Dense(128)(x)
    x = ReLU()(x)
    x = Dropout(0.2)(x)
    x = Dense(out_num)(x)
    x = ReLU()(x)
    x = Dropout(0.2)(x)
    x3 = Add()([x2, x])
    x3 = LayerNormalization(epsilon=1e-6)(x3)
    return x3

# define the standalone discriminator model
def define_discriminator(in_shape=(200,67), n_classes=2):
    # weight initialization
    #init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=(in_shape))
    input = in_image
    input = Conv1D(64, 3, strides=1)(input)
    # encoding = PositionEncoding()(input)
    # input = Add()([input, encoding])

    res = Bidirectional(LSTM(64, return_sequences=True))(input)
    res = Bidirectional(LSTM(32))(res)
    res = Dense(64, activation='relu')(res)
    # res = Dense(32, activation='tanh')(res)

    # out1 = Dense(1, activation='sigmoid')(res)
    res = Flatten()(res)
    out2 = Dense(n_classes, activation='softmax')(res)

    # model = Model(in_image, [out1, out2],name="Discriminator")
    model = Model(in_image, out2, name="Discriminator")
    opt = Adam(lr=0.0002, beta_1=0.5)

    # model.compile(loss=['mse', 'categorical_crossentropy'], optimizer=opt)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    model.summary()
    return model

def define_generator_2(latent_dim=(50,), signal_shape=(200, 67), label_shape=(2,)):
    # weight initialization
    # init = RandomNormal(stddev=0.02)
    depth = 4  # 32
    dropout = 0.25
    dim = signal_shape[0]  #

    # signal_input
    in_signal = Input(shape=signal_shape)
    si = in_signal
    # si = Reshape((280,1))(in_signal)

    # label input
    in_label = Input(shape=label_shape)
    # embedding for categorical input
    li = Embedding(2, 50)(in_label)
    # linear multiplication
    n_nodes = 200 * 1
    li = Dense(n_nodes)(li)

    # reshape to additional channel
    li = Reshape((200, 2))(li)

    # noise  input
    in_lat = Input(shape=latent_dim)
    lat = Reshape((1, 50))(in_lat)

    n_nodes = dim * depth
    gen = Dense(n_nodes)(lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((dim, depth))(gen)
    # merge image gen and label input
    # print(gen.shape, li.shape, si.shape)
    merge = Concatenate()([gen, li, si])  # gen=200,32 x li=200,2 x si=200,67 ## Uncomment this
    # target = merge
    # target = Conv1D(32, 3, strides=1, padding='same')(target)
    # tar_encoding = PositionEncoding()(target)
    # target = Add()([target, tar_encoding])

    # merge = Concatenate()([gen, li]) #gen=280,32 li=280,5
    input = Conv1D(64, 3, strides=1, padding='same')(merge)

    encoding = PositionEncoding()(input)
    print("encoding: ", encoding.shape)
    input = Add()([input, encoding])

    gen = Conv1D(64, 3, strides=1, padding='same')(input)
    gen = LayerNormalization(epsilon=1e-6)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv1D(64, 3, strides=2, padding='same')(gen)
    gen = LayerNormalization(epsilon=1e-6)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv1D(128, 3, strides=1, padding='same')(gen)
    gen = LayerNormalization(epsilon=1e-6)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv1D(128, 3, strides=2, padding='same')(gen)
    gen = LayerNormalization(epsilon=1e-6)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv1D(256, 3, strides=1, padding='same')(gen)
    gen = LayerNormalization(epsilon=1e-6)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv1D(256, 3, strides=2, padding='same')(gen)
    gen = LayerNormalization(epsilon=1e-6)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv1DTranspose(256, 3, strides=2, padding='same')(gen)
    gen = LayerNormalization(epsilon=1e-6)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv1DTranspose(128, 3, strides=2, padding='same')(gen)
    gen = LayerNormalization(epsilon=1e-6)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv1DTranspose(64, 3, strides=2, padding='same')(gen)
    gen = LayerNormalization(epsilon=1e-6)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    x = input
    num_encoder = 3
    num_decoder = 3
    for i in range(num_encoder):
        x = transformer_encoder(x)
    enc_out = x
    x = gen
    for i in range(num_decoder):
        x = transformer_decoder(x, enc_out, 200)
    dec_out = x

    # gen = Reshape((200,67))(gen)

    gen = Conv1D(67, 3, strides=1, padding='same')(dec_out)
    out_layer = Activation('sigmoid')(gen)
    # print("out_layer: ", out_layer.shape)

    model = Model([in_signal, in_lat, in_label], out_layer, name="Generator")
    # model = Model([in_lat, in_label], out_layer,name="Generator")
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='mse', optimizer=opt)
    model.summary()
    return model

def define_generator(latent_dim=(50,), signal_shape=(200, 67), label_shape=(2,)):
    # weight initialization
    # init = RandomNormal(stddev=0.02)
    depth = 4  # 32
    dropout = 0.25
    dim = signal_shape[0]  #

    # signal_input
    in_signal = Input(shape=signal_shape)
    si = in_signal
    # si = Reshape((280,1))(in_signal)

    # label input
    in_label = Input(shape=label_shape)
    # embedding for categorical input
    li = Embedding(2, 50)(in_label)
    # linear multiplication
    n_nodes = 200 * 1
    li = Dense(n_nodes)(li)

    # reshape to additional channel
    li = Reshape((200, 2))(li)

    # noise  input
    in_lat = Input(shape=latent_dim)
    lat = Reshape((1, 50))(in_lat)

    n_nodes = dim * depth
    gen = Dense(n_nodes)(lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((dim, depth))(gen)
    # merge image gen and label input
    # print(gen.shape, li.shape, si.shape)
    merge = Concatenate()([gen, li, si])  # gen=200,32 x li=200,2 x si=200,67 ## Uncomment this
    # target = merge
    # target = Conv1D(32, 3, strides=1, padding='same')(target)
    # tar_encoding = PositionEncoding()(target)
    # target = Add()([target, tar_encoding])


    # merge = Concatenate()([gen, li]) #gen=280,32 li=280,5
    input = Conv1D(32, 3, strides=1, padding='same')(merge)

    encoding = PositionEncoding()(input)
    print("encoding: ", encoding.shape)
    input = Add()([input, encoding])

    gen = Conv1D(32, 3, strides=1, padding='same')(input)
    gen = LayerNormalization(epsilon=1e-6)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv1D(32, 3, strides=2, padding='same')(gen)
    gen = LayerNormalization(epsilon=1e-6)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv1D(64, 3, strides=1, padding='same')(gen)
    gen = LayerNormalization(epsilon=1e-6)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv1D(64, 3, strides=2, padding='same')(gen)
    gen = LayerNormalization(epsilon=1e-6)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    x = gen
    num_encoder = 2
    num_decoder = 2
    for i in range(num_encoder):
        x = transformer_encoder(x)
    enc_out = x
    x = gen
    for i in range(num_decoder):
        x = transformer_decoder(x, enc_out, 50)
    dec_out = x

    gen = Conv1DTranspose(64, 3, strides=2, padding='same')(dec_out)
    gen = LayerNormalization(epsilon=1e-6)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv1DTranspose(32, 3, strides=2, padding='same')(gen)
    gen = LayerNormalization(epsilon=1e-6)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)


    # gen = Reshape((200,67))(gen)

    gen = Conv1D(67, 3, strides=1, padding='same')(gen)
    out_layer = Activation('sigmoid')(gen)
    # print("out_layer: ", out_layer.shape)

    model = Model([in_signal, in_lat, in_label], out_layer, name="Generator")
    # model = Model([in_lat, in_label], out_layer,name="Generator")
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='mse', optimizer=opt)
    model.summary()
    return model


def define_discriminator_1(in_shape=(200, 67), n_classes=2):
    # weight initialization
    # init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=(in_shape))
    # downsample to 14x14
    fe = Conv1D(16, 3, strides=1, padding='same')(in_image)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv1D(16, 3, strides=2, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)
    # normal
    fe = Conv1D(32, 3, strides=1, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv1D(32, 3, strides=2, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # downsample to 7x7
    fe = Conv1D(128, 3, strides=1, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Conv1D(128, 3, strides=2, padding='same')(fe)
    fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.2)(fe)

    # flatten feature maps
    fe = Flatten()(fe)
    # dense_1 = Dense(256)(fe)
    # dense_1 = LeakyReLU()(dense_1)
    # dense_1 = Dense(64)(dense_1)
    # real/fake output
    out1 = Dense(1, activation='sigmoid')(fe)
    # class label output
    out2 = Dense(n_classes, activation='softmax')(fe)
    # define model
    model = Model(in_image, [out1, out2], name="Discriminator")
    # model = Model(in_image,  out2, name="Discriminator")
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['mse', 'categorical_crossentropy'], optimizer=opt)
    # model.compile(loss='categorical_crossentropy', optimizer=opt)
    model.summary()
    return model

# define the standalone discriminator model
def define_discriminator_2(in_shape=(200,67), n_classes=2):
    # weight initialization
    #init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=(in_shape))
    input = Conv1D(64, 3, strides=1, padding='same')(in_image)
    encoding = PositionEncoding()(input)
    x = Add()([input, encoding])

    in1 = transformer_encoder(x)
    in2 = transformer_encoder(in1)
    in3 = transformer_encoder(in2)

    # downsample to 14x14
    # fe = Conv1D(16, 3, strides=1, padding='same')(x)
    # fe = LayerNormalization(epsilon=1e-6)(fe)
    # fe = LeakyReLU(alpha=0.2)(fe)
    # fe = Dropout(0.2)(fe)
    # normal
    # fe = Conv1D(32, 3, strides=1, padding='same')(fe)
    # fe = LayerNormalization(epsilon=1e-6)(fe)
    # fe = LeakyReLU(alpha=0.2)(fe)
    # fe = Dropout(0.2)(fe)

    # downsample to 7x7
    # fe = Conv1D(64, 3, strides=1, padding='same')(fe)
    # fe = LayerNormalization(epsilon=1e-6)(fe)
    # fe = LeakyReLU(alpha=0.2)(fe)
    # fe = Dropout(0.2)(fe)

    res = transformer_decoder(x, in3, 200)
    res = Flatten()(res)
    # res = transformer_encoder(in3)
    out1 = Dense(1, activation='sigmoid')(res)
    out2 = Dense(n_classes, activation='softmax')(res)

    model = Model(in_image, [out1, out2],name="Discriminator")
    opt = Adam(lr=0.0002, beta_1=0.5)

    model.compile(loss=['mse', 'categorical_crossentropy'], optimizer=opt)
    model.summary()
    return model

def cnn_mhsa(in_shape=(200,67), n_classes=2):
    # weight initialization
    #init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=(in_shape))
    x = Conv1D(64, 3, strides=1, padding='same')(in_image)
    encoding = PositionEncoding()(x)
    x1 = Add()([x, encoding])

    in1 = x1

    in2 = MultiHeadAttention(num_heads=8, key_dim=64)(in1,in1)
    in2 = Add()([in1, in2])
    in2 = LayerNormalization(epsilon=1e-6)(in2)
    in3 = Dense(64)(in1)
    in3 = ReLU()(in3)
    in3 = Add()([in3, in2])
    in3 = LayerNormalization(epsilon=1e-6)(in3)

    in1 = in3

    in2 = MultiHeadAttention(num_heads=8, key_dim=64)(in1,in1)
    in2 = Add()([in1, in2])
    in2 = LayerNormalization(epsilon=1e-6)(in2)
    in3 = Dense(64)(in1)
    in3 = ReLU()(in3)
    in3 = Add()([in3, in2])
    in3 = LayerNormalization(epsilon=1e-6)(in3)

    in1 = in3

    in2 = MultiHeadAttention(num_heads=8, key_dim=64)(in1,in1)
    in2 = Add()([in1, in2])
    in2 = LayerNormalization(epsilon=1e-6)(in2)
    in3 = Dense(64)(in1)
    in3 = ReLU()(in3)
    in3 = Add()([in3, in2])
    in3 = LayerNormalization(epsilon=1e-6)(in3)

    fe = Conv1D(64, 3, strides=1, padding='same')(x)
    fe = Add()([fe, x])
    fe = LayerNormalization(epsilon=1e-6)(fe)
    fe1 = ReLU()(fe)
    fe = Dense(64)(fe1)
    fe = ReLU()(fe)
    fe = Add()([fe, fe1])
    fe = LayerNormalization(epsilon=1e-6)(fe)

    x = fe
    fe = Conv1D(64, 3, strides=1, padding='same')(x)
    fe = Add()([fe, x])
    fe = LayerNormalization(epsilon=1e-6)(fe)
    fe1 = ReLU()(fe)
    fe = Dense(64)(fe1)
    fe = ReLU()(fe)
    fe = Add()([fe, fe1])
    fe = LayerNormalization(epsilon=1e-6)(fe)

    x = fe
    fe = Conv1D(64, 3, strides=1, padding='same')(x)
    fe = Add()([fe, x])
    fe = LayerNormalization(epsilon=1e-6)(fe)
    fe1 = ReLU()(fe)
    fe = Dense(64)(fe1)
    fe = ReLU()(fe)
    fe = Add()([fe, fe1])
    fe = LayerNormalization(epsilon=1e-6)(fe)

    x1 = in3
    x2 = fe
    res = MultiHeadAttention(num_heads=8, key_dim=64)(x1,x1,fe)
    res = Add()([res, x2])
    res1 = LayerNormalization(epsilon=1e-6)(res)
    res = Dense(64)(res)
    res = ReLU()(res)
    res = Add()([res, res1])
    res2 = LayerNormalization(epsilon=1e-6)(res)
    res = Flatten()(res2)

    # out1 = Dense(1, activation='sigmoid')(res)
    out2 = Dense(n_classes, activation='softmax')(res)

    # model = Model(in_image, [out1, out2],name="Discriminator")
    model = Model(in_image,  out2, name="Discriminator")
    opt = Adam(lr=0.0002, beta_1=0.5)

    # model.compile(loss=['mse', 'categorical_crossentropy'], optimizer=opt)
    model.compile(loss= 'categorical_crossentropy', optimizer=opt)
    model.summary()
    return model

def define_discriminator_4(in_shape=(200,67), n_classes=2):
    # weight initialization
    #init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=(in_shape))
    x = in_image
    # x = in_image[:,:,1:67]

    # encoding = PositionEncoding()(x)
    # x = Add()([x, encoding])
    # padding_mask = create_padding_mask_3(in_image[:, :, 0][0])

    padding_mask = None
    x = Conv1D(64, 3, strides=1, padding='same')(x)

    in1 = transformer_encoder(x, 64, padding_mask)
    in2 = transformer_encoder(in1, 64, padding_mask)
    in3 = transformer_encoder(in2, 64, padding_mask)
    # in4 = transformer_encoder(in3)
    # in5 = transformer_encoder(in4)
    # in6 = transformer_encoder(in5)
    # in7 = transformer_encoder(in6)
    # fe = Conv1D(64, 3, strides=1, padding='same')(x)
    # fe = LayerNormalization(epsilon=1e-6)(fe)
    # fe = ReLU()(fe)
    # fe = Dropout(0.2)(fe)
    # downsample to 14x14
    fe = Conv1D(64, 3, strides=1, padding='same')(x)
    fe = LayerNormalization(epsilon=1e-6)(fe)
    fe = ReLU()(fe)
    fe = Dropout(0.2)(fe)
    #
    fe = Conv1D(64, 3, strides=1, padding='same')(fe)
    fe = LayerNormalization(epsilon=1e-6)(fe)
    fe = ReLU()(fe)
    fe = Dropout(0.2)(fe)
    #
    fe = Conv1D(64, 3, strides=1, padding='same')(fe)
    fe = LayerNormalization(epsilon=1e-6)(fe)
    fe = ReLU()(fe)
    fe = Dropout(0.2)(fe)

    # fe = spatial_attention_v4(fe)

    # normal
    # fe = Conv1D(64, 3, strides=1, padding='same')(fe)
    # fe = LayerNormalization(epsilon=1e-6)(fe)
    # fe = ReLU()(fe)
    # fe = Dropout(0.2)(fe)
    #
    # fe = Conv1D(64, 3, strides=1, padding='same')(fe)
    # fe = LayerNormalization(epsilon=1e-6)(fe)
    # fe = ReLU()(fe)
    # fe = Dropout(0.2)(fe)
    #
    # fe = Conv1D(64, 3, strides=1, padding='same')(fe)
    # fe = LayerNormalization(epsilon=1e-6)(fe)
    # fe = ReLU()(fe)
    # fe = Dropout(0.2)(fe)

    res = transformer_decoder(fe, in3, 200, padding_mask)
    # res = transformer_decoder(res, in3,200)
    # res = transformer_decoder(res, in3, 200)
    res = Flatten()(res)
    # res = Dense(128)(res)
    # res = ReLU()(res)
    # res = Dense(64)(res)

    # out1 = Dense(1, activation='sigmoid')(res)
    out2 = Dense(n_classes, activation='softmax')(res)

    # model = Model(in_image, [out1, out2],name="Discriminator")
    model = Model(in_image, out2, name="Discriminator")
    opt = Adam(lr=0.0002, beta_1=0.5)

    model.compile(loss='categorical_crossentropy', optimizer=opt)
    model.summary()
    return model

# model = define_discriminator_4()

def define_discriminator_5(in_shape=(200,67), n_classes=2):
    # weight initialization
    #init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=(in_shape))
    x = Conv1D(64, 3, strides=1, padding='same')(in_image)
    encoding = PositionEncoding()(x)
    x = Add()([x, encoding])

    x1 = Reshape((50, 4*64))(x)
    in1 = transformer_encoder(x1, 4*64)
    in1 = Reshape((100, 2*64))(in1)
    in2 = transformer_encoder(in1, 2*64)
    in2 = Reshape((200, 1*64))(in2)
    in3 = transformer_encoder(in2, 1*64)
    # in4 = transformer_encoder(in3)
    # in5 = transformer_encoder(in4)
    # in6 = transformer_encoder(in5)
    # in7 = transformer_encoder(in6)
    # fe = Conv1D(64, 3, strides=1, padding='same')(x)
    # fe = LayerNormalization(epsilon=1e-6)(fe)
    # fe = ReLU()(fe)
    # fe = Dropout(0.2)(fe)
    # downsample to 14x14
    fe = Conv1D(64, 3, strides=1, padding='same')(x)
    fe = LayerNormalization(epsilon=1e-6)(fe)
    fe = ReLU()(fe)
    fe = Dropout(0.2)(fe)
    #
    fe = Conv1D(64, 3, strides=1, padding='same')(fe)
    fe = LayerNormalization(epsilon=1e-6)(fe)
    fe = ReLU()(fe)
    fe = Dropout(0.2)(fe)
    #
    fe = Conv1D(64, 3, strides=1, padding='same')(fe)
    fe = LayerNormalization(epsilon=1e-6)(fe)
    fe = ReLU()(fe)
    fe = Dropout(0.2)(fe)

    # fe = spatial_attention_v4(fe)

    # normal
    # fe = Conv1D(64, 3, strides=1, padding='same')(fe)
    # fe = LayerNormalization(epsilon=1e-6)(fe)
    # fe = ReLU()(fe)
    # fe = Dropout(0.2)(fe)
    #
    # fe = Conv1D(64, 3, strides=1, padding='same')(fe)
    # fe = LayerNormalization(epsilon=1e-6)(fe)
    # fe = ReLU()(fe)
    # fe = Dropout(0.2)(fe)
    #
    # fe = Conv1D(64, 3, strides=1, padding='same')(fe)
    # fe = LayerNormalization(epsilon=1e-6)(fe)
    # fe = ReLU()(fe)
    # fe = Dropout(0.2)(fe)

    res = transformer_decoder(fe, in3, 200)
    # res = transformer_decoder(res, in3,200)
    # res = transformer_decoder(res, in3, 200)
    res = Flatten()(res)
    # res = Dense(128)(res)
    # res = ReLU()(res)
    # res = Dense(64)(res)

    out1 = Dense(1, activation='sigmoid')(res)
    out2 = Dense(n_classes, activation='softmax')(res)

    model = Model(in_image, [out1, out2],name="Discriminator")
    opt = Adam(lr=0.0002, beta_1=0.5)

    model.compile(loss=['mse', 'categorical_crossentropy'], optimizer=opt)
    model.summary()
    return model

def define_discriminator_3(in_shape=(200,67), n_classes=2):
    # weight initialization
    #init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=(in_shape))
    input = Conv1D(64, 3, strides=1, padding='same')(in_image)
    encoding = PositionEncoding()(input)
    x = Add()([input, encoding])

    in1 = transformer_encoder(x,64,None)
    in2 = transformer_encoder(in1,64,None)
    in3 = transformer_encoder(in2,64,None)

    res = transformer_encoder(in3,64,None)
    # res = transformer_encoder(res)
    res = Flatten()(res)
    # out1 = Dense(1, activation='sigmoid')(res)
    out2 = Dense(n_classes, activation='softmax')(res)

    # model = Model(in_image, [out1, out2],name="Discriminator")
    model = Model(in_image, out2, name="Discriminator")
    opt = Adam(lr=0.0002, beta_1=0.5)

    # model.compile(loss=['mse', 'categorical_crossentropy'], optimizer=opt)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    model.summary()
    return model

# define the standalone generator model
def define_generator_1(latent_dim=(50,),signal_shape=(200,67), label_shape=(2,)):
    # weight initialization
    #init = RandomNormal(stddev=0.02)
    depth = 4 #32
    dropout = 0.25
    dim = signal_shape[0] #
    
    # signal_input
    in_signal = Input(shape=signal_shape)
    si = in_signal
    #si = Reshape((280,1))(in_signal)
    
    # label input
    in_label = Input(shape=label_shape)
    # embedding for categorical input
    li = Embedding(2, 50)(in_label)
    # linear multiplication
    n_nodes = 200 * 1
    li = Dense(n_nodes)(li)

    # reshape to additional channel
    li = Reshape((200,2))(li)
    
    # noise  input
    in_lat = Input(shape=latent_dim)
    lat = Reshape((1,50))(in_lat)

    n_nodes = dim*depth
    gen = Dense(n_nodes)(lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((dim, depth))(gen)
    # merge image gen and label input
    print(gen.shape, li.shape, si.shape)
    merge = Concatenate()([gen, li, si]) #gen=200,32 x li=200,2 x si=200,67 ## Uncomment this
    print(merge.shape)
    #merge = Concatenate()([gen, li]) #gen=280,32 li=280,5
 

    gen = Conv1D(32, 3, strides=1, padding='same')(merge)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv1D(32, 3, strides=2, padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv1D(64, 3, strides=1, padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv1D(64, 3, strides=2, padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv1D(128, 3, strides=1, padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv1D(128, 3, strides=2, padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv1DTranspose(128, 3, strides=2, padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv1DTranspose(64, 3, strides=2, padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    gen = Conv1DTranspose(32, 3, strides=2, padding='same')(gen)
    gen = BatchNormalization()(gen)
    gen = LeakyReLU(alpha=0.2)(gen)


    #gen = Reshape((200,67))(gen)

    gen = Conv1D(67, 3, strides=1, padding='same')(gen)
    out_layer = Activation('sigmoid')(gen)
    # print("out_layer: ", out_layer.shape)

    model = Model([in_signal,in_lat, in_label], out_layer,name="Generator")
    #model = Model([in_lat, in_label], out_layer,name="Generator")
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='mse', optimizer=opt)
    model.summary()
    return model

def define_gan(g_model, d_model,latent_dim=(200,67), signal_shape=(200,67),label_shape=(2,)):
    #in_signal = Input(shape=signal_shape)
    #in_label = Input(shape=label_shape)
    #in_lat = Input(shape=latent_dim)
    # make weights in the discriminator not trainable
    d_model.trainable = False
    print("g_model.output: ", g_model.output.shape)
    # connect the outputs of the generator to the inputs of the discriminator
    [out1,out2] = d_model(g_model.output)
    # define gan model as taking noise and label and outputting real/fake and label outputs
    #model = Model(g_model.input, gan_output)
    model = Model([g_model.input[0],g_model.input[1],g_model.input[2]],[out1,out2])
    #model = Model([g_model.input[0],g_model.input[1]],[out1,out2])
    #model = Model([in_signal,in_lat, in_label],[out1,out2])
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['mse', 'categorical_crossentropy'], optimizer=opt,loss_weights=[1,10])
    model.summary()
    return model