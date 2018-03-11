import numpy as np
import tensorflow as tf
from utils import conv2d_flipkernel

def VI_Block(X, S1, S2, config):
    k    = config.k    # Number of value iterations performed
    ch_i = config.ch_i # Channels in input layer
    ch_h = config.ch_h # Channels in initial hidden layer
    ch_q1 = config.ch_q1 # Channels in q layer (~transfer actions)
    ch_q2 = config.ch_q2  # Channels in q layer (~new actions)
    state_batch_size = config.statebatchsize # k+1 state inputs for each channel

    bias  = tf.Variable(np.random.randn(1, 1, 1, ch_h)    * 0.01, dtype=tf.float32, name = 'bias')
    # weights from inputs to q layer (~reward in Bellman equation)
    w0    = tf.Variable(np.random.randn(3, 3, ch_i, ch_h) * 0.01, dtype=tf.float32, name = 'w0')
    w1    = tf.Variable(np.random.randn(1, 1, ch_h, 1)    * 0.01, dtype=tf.float32, name = 'w1')
    # reward function weights for transfer actions
    w     = tf.Variable(np.random.randn(3, 3, 1, ch_q1)    * 0.01, dtype=tf.float32, trainable = False, name = 'w')
    # feedback weights from v layer into q layer (~transition probabilities in Bellman equation)
    w_fb  = tf.Variable(np.random.randn(3, 3, 1, ch_q1)    * 0.01, dtype=tf.float32, trainable = False, name = 'w_fb')
    # new kernel weights for new acitions in target domain
    w_new = tf.Variable(np.random.randn(3, 3, 1, ch_q2) * 0.01, dtype=tf.float32, name='w_new')
    # feedback weights from v layer into q layer (~transition probabilities in Bellman equation)
    w_fb_new = tf.Variable(np.random.randn(3, 3, 1, ch_q2) * 0.01, dtype=tf.float32, name='w_fb_new')
    #transfer weights to leverage pre-trained kernels
    w_transfer = tf.Variable(np.random.randn(ch_q1)* 0.01, dtype=tf.float32, trainable = True, name="w_transfer")
    #output layer (~total actions)
    w_o_8 = tf.Variable(np.random.randn(ch_q1+ch_q2, 8)     * 0.01, dtype=tf.float32, name='w_o')

    # compute weighted kernels
    w_fb = tf.transpose(w_fb, perm=[3, 0, 1, 2])
    w_fb_mul = w_transfer[0] * w_fb[0]  # w_mul init
    w_fb_mul = tf.reshape(w_fb_mul, [1, 3, 3, 1])

    if ch_q1 > 1:
        for i in range(ch_q1 - 1):
            w_q = w_transfer[i + 1] * w_fb[i + 1]
            w_q = tf.reshape(w_q, [1, 3, 3, 1])
            w_fb_mul = tf.concat([w_fb_mul, w_q], 0)

    w_fb_mul = tf.transpose(w_fb_mul, perm=[1, 2, 3, 0])

    #concat w to 8 actions
    w_8 = tf.concat([w, w_new], 3)
    w_fb_8 = tf.concat([w_fb_mul, w_fb_new], 3)


    # initial conv layer over image+reward prior
    h = conv2d_flipkernel(X, w0, name="h0") + bias
    r = conv2d_flipkernel(h, w1, name="r")
    q = conv2d_flipkernel(r, w_8, name="q")
    v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")

    for i in range(0, k-1):
        rv = tf.concat([r, v], 3)
        wwfb = tf.concat([w_8, w_fb_8], 2)
        q = conv2d_flipkernel(rv, wwfb, name="q")
        v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")

    # do one last convolution
    q = conv2d_flipkernel(tf.concat([r, v], 3),
                          tf.concat([w_8, w_fb_8], 2), name="q")

    # CHANGE TO THEANO ORDERING
    # Since we are selecting over channels, it becomes easier to work with
    # the tensor when it is in NCHW format vs NHWC
    q = tf.transpose(q, perm=[0, 3, 1, 2])

    # Select the conv-net channels at the state position (S1,S2).
    # This intuitively corresponds to each channel representing an action, and the convnet the Q function.
    # The tricky thing is we want to select the same (S1,S2) position *for each* channel and for each sample
    # TODO: performance can be improved here by substituting expensive
    #       transpose calls with better indexing for gather_nd
    bs = tf.shape(q)[0]
    rprn = tf.reshape(tf.tile(tf.reshape(tf.range(bs), [-1, 1]), [1, state_batch_size]), [-1])
    ins1 = tf.cast(tf.reshape(S1, [-1]), tf.int32)
    ins2 = tf.cast(tf.reshape(S2, [-1]), tf.int32)
    idx_in = tf.transpose(tf.stack([ins1, ins2, rprn]), [1, 0])
    q_out = tf.gather_nd(tf.transpose(q, [2, 3, 0, 1]), idx_in, name="q_out")

    # add logits
    logits = tf.matmul(q_out, w_o_8)
    # softmax output weights
    output = tf.nn.softmax(logits, name="output")
    return logits, output

