import tensorflow as tf
from tensorflow.contrib.layers import flatten


mu = 0
sigma = 0.1

activation = tf.nn.relu

def linear(x, W, b):
    return tf.add(tf.matmul(x, W), b)

def conv2d(x, W, b, strides=1, padding='VALID'):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
    return tf.nn.bias_add(x, b)

def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)

def maxpool2d(x, k=2, s=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, s, s, 1],
        padding='SAME')

def gen_keep_prob():
    return tf.placeholder(tf.float32, name="keep_prob")

def LeNet(x):

    weights = {
        'conv1' : tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma)),
        'conv2' : tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma)),
        'flat3' : tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma)),
        'flat4' : tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma)),
        'flat5' : tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    }
    biases = {
        'conv1' : tf.Variable(tf.zeros(6)),
        'conv2' : tf.Variable(tf.zeros(16)),
        'flat3' : tf.Variable(tf.zeros(120)),
        'flat4' : tf.Variable(tf.zeros(84)),
        'flat5' : tf.Variable(tf.zeros(43))
    }

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1 = conv2d(x, weights['conv1'], biases['conv1'])

    # Activation
    conv1 = activation(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = maxpool2d(conv1, 2)

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2 = conv2d(conv1, weights['conv2'], biases['conv2'])

    # Activation.
    conv2 = activation(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = maxpool2d(conv2, 2)

    # Flatten. Input = 5x5x16. Output = 400.
    flat3 = tf.reshape(conv2, [-1, weights['flat3'].get_shape().as_list()[0]])

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    flat3 = linear(flat3, weights['flat3'], biases['flat3'])

    # Activation.
    flat3 = activation(flat3)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    flat4 = linear(flat3, weights['flat4'], biases['flat4'])

    # Activation.
    flat4 = activation(flat4)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    logits = linear(flat4, weights['flat5'], biases['flat5'])

    return logits


def LeNet3(x):

    weights = {
        'conv1' : tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma)),
        'conv2' : tf.Variable(tf.truncated_normal(shape=(3, 3, 6, 16), mean = mu, stddev = sigma)),
        'conv3' : tf.Variable(tf.truncated_normal(shape=(3, 3, 16, 32), mean = mu, stddev = sigma)),
        'flat4' : tf.Variable(tf.truncated_normal(shape=(800, 200), mean = mu, stddev = sigma)),
        'flat5' : tf.Variable(tf.truncated_normal(shape=(200, 84), mean = mu, stddev = sigma)),
        'flat6' : tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    }
    biases = {
        'conv1' : tf.Variable(tf.zeros(6)),
        'conv2' : tf.Variable(tf.zeros(16)),
        'conv3' : tf.Variable(tf.zeros(32)),
        'flat4' : tf.Variable(tf.zeros(200)),
        'flat5' : tf.Variable(tf.zeros(84)),
        'flat6' : tf.Variable(tf.zeros(43))
    }

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1 = conv2d(x, weights['conv1'], biases['conv1'])

    # Activation
    conv1 = activation(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = maxpool2d(conv1, 2)

    # Layer 2: Convolutional. Output = 12x12x16.
    conv2 = conv2d(conv1, weights['conv2'], biases['conv2'])

    # Activation.
    conv2 = activation(conv2)

    # Layer 3: Convolutional. Output = 10x10x32.
    conv3 = conv2d(conv2, weights['conv3'], biases['conv3'])

    # Activation.
    conv3 = activation(conv3)

    # Pooling. Input = 10x10x32. Output = 5x5x32.
    conv3 = maxpool2d(conv3, 2)

    # Flatten. Input = 5x5x32. Output = 800.
    flat4 = tf.reshape(conv3, [-1, weights['flat4'].get_shape().as_list()[0]])

    # Layer 4: Fully Connected. Input = 800. Output = 200.
    flat4 = linear(flat4, weights['flat4'], biases['flat4'])

    # Activation.
    flat4 = activation(flat4)

    # Layer 5: Fully Connected. Input = 200. Output = 84.
    flat5 = linear(flat4, weights['flat5'], biases['flat5'])

    # Activation.
    flat5 = activation(flat5)

    # Layer 6: Fully Connected. Input = 84. Output = 43.
    logits = linear(flat5, weights['flat6'], biases['flat6'])

    return logits


def LeNet3_dropout_fc(x, keep_prob):

    weights = {
        'conv1' : tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma)),
        'conv2' : tf.Variable(tf.truncated_normal(shape=(3, 3, 6, 16), mean = mu, stddev = sigma)),
        'conv3' : tf.Variable(tf.truncated_normal(shape=(3, 3, 16, 32), mean = mu, stddev = sigma)),
        'flat4' : tf.Variable(tf.truncated_normal(shape=(800, 200), mean = mu, stddev = sigma)),
        'flat5' : tf.Variable(tf.truncated_normal(shape=(200, 84), mean = mu, stddev = sigma)),
        'flat6' : tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    }
    biases = {
        'conv1' : tf.Variable(tf.zeros(6)),
        'conv2' : tf.Variable(tf.zeros(16)),
        'conv3' : tf.Variable(tf.zeros(32)),
        'flat4' : tf.Variable(tf.zeros(200)),
        'flat5' : tf.Variable(tf.zeros(84)),
        'flat6' : tf.Variable(tf.zeros(43))
    }

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1 = conv2d(x, weights['conv1'], biases['conv1'])

    # Activation
    conv1 = activation(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = maxpool2d(conv1, 2)

    # Layer 2: Convolutional. Output = 12x12x16.
    conv2 = conv2d(conv1, weights['conv2'], biases['conv2'])

    # Activation.
    conv2 = activation(conv2)

    # Layer 3: Convolutional. Output = 10x10x32.
    conv3 = conv2d(conv2, weights['conv3'], biases['conv3'])

    # Activation.
    conv3 = activation(conv3)

    # Pooling. Input = 10x10x32. Output = 5x5x32.
    conv3 = maxpool2d(conv3, 2)

    # Flatten. Input = 5x5x32. Output = 800.
    flat4 = tf.reshape(conv3, [-1, weights['flat4'].get_shape().as_list()[0]])

    # Layer 4: Fully Connected. Input = 800. Output = 200.
    flat4 = linear(flat4, weights['flat4'], biases['flat4'])

    # Activation.
    flat4 = activation(flat4)

    # Dropout
    flat4 = dropout(flat4, keep_prob)

    # Layer 5: Fully Connected. Input = 200. Output = 84.
    flat5 = linear(flat4, weights['flat5'], biases['flat5'])

    # Activation.
    flat5 = activation(flat5)

    # Dropout
    flat5 = dropout(flat5, keep_prob)

    # Layer 6: Fully Connected. Input = 84. Output = 43.
    logits = linear(flat5, weights['flat6'], biases['flat6'])

    return logits


def LeNet_dropout(x, keep_prob):

    weights = {
        'conv1' : tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma)),
        'conv2' : tf.Variable(tf.truncated_normal(shape=(9, 9, 6, 16), mean = mu, stddev = sigma)),
        'flat3' : tf.Variable(tf.truncated_normal(shape=(1600, 400), mean = mu, stddev = sigma)),
        'flat4' : tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma)),
        'flat5' : tf.Variable(tf.truncated_normal(shape=(120, 43), mean = mu, stddev = sigma))
    }
    biases = {
        'conv1' : tf.Variable(tf.zeros(6)),
        'conv2' : tf.Variable(tf.zeros(16)),
        'flat3' : tf.Variable(tf.zeros(400)),
        'flat4' : tf.Variable(tf.zeros(120)),
        'flat5' : tf.Variable(tf.zeros(43))
    }

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1 = conv2d(x, weights['conv1'], biases['conv1'])

    # Activation
    conv1 = activation(conv1)

    # Dropout
    conv1 = dropout(conv1, keep_prob)

    # Layer 2: Convolutional. Input = 28x28x6. Output = 10x10x16.
    conv2 = conv2d(conv1, weights['conv2'], biases['conv2'], strides=2)

    # Activation.
    conv2 = activation(conv2)

    # Dropout
    conv2 = dropout(conv2, keep_prob)

    # Flatten. Input = 10x10x16. Output = 1600.
    flat3 = tf.reshape(conv2, [-1, weights['flat3'].get_shape().as_list()[0]])

    # Layer 3: Fully Connected. Input = 1600. Output = 400.
    flat3 = linear(flat3, weights['flat3'], biases['flat3'])

    # Activation.
    flat3 = activation(flat3)

    # Layer 4: Fully Connected. Input = 400. Output = 120.
    flat4 = linear(flat3, weights['flat4'], biases['flat4'])

    # Activation.
    flat4 = activation(flat4)

    # Layer 5: Fully Connected. Input = 120. Output = 43.
    logits = linear(flat4, weights['flat5'], biases['flat5'])

    return logits


def Multi_Scale_LeNet(x):

    weights = {
        'conv1' : tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma)),
        'conv2' : tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma)),
        'flat3' : tf.Variable(tf.truncated_normal(shape=(1576, 400), mean = mu, stddev = sigma)),
        'flat4' : tf.Variable(tf.truncated_normal(shape=(400, 100), mean = mu, stddev = sigma)),
        'flat5' : tf.Variable(tf.truncated_normal(shape=(100, 43), mean = mu, stddev = sigma))
    }
    biases = {
        'conv1' : tf.Variable(tf.zeros(6)),
        'conv2' : tf.Variable(tf.zeros(16)),
        'flat3' : tf.Variable(tf.zeros(400)),
        'flat4' : tf.Variable(tf.zeros(100)),
        'flat5' : tf.Variable(tf.zeros(43))
    }

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1 = conv2d(x, weights['conv1'], biases['conv1'])

    # Activation
    conv1 = activation(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = maxpool2d(conv1, 2)

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2 = conv2d(conv1, weights['conv2'], biases['conv2'])

    # Activation.
    conv2 = activation(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = maxpool2d(conv2, 2)

    # Flatten. Input = 14x14x6 + 5x5x16. Output = 1576.
    flat_conv1 = tf.reshape(conv1, [-1, 14*14*6])
    flat_conv2 = tf.reshape(conv2, [-1, 5*5*16])
    flat3 = tf.concat(1, [flat_conv1, flat_conv2])

    # Layer 3: Fully Connected. Input = 1576. Output = 400.
    flat3 = linear(flat3, weights['flat3'], biases['flat3'])

    # Activation.
    flat3 = activation(flat3)

    # Layer 4: Fully Connected. Input = 400. Output = 100.
    flat4 = linear(flat3, weights['flat4'], biases['flat4'])

    # Activation.
    flat4 = activation(flat4)

    # Layer 5: Fully Connected. Input = 100. Output = 43.
    logits = linear(flat4, weights['flat5'], biases['flat5'])

    return logits


def Multi_Scale_LeNet_pooling_dropout(x, keep_prob):

    weights = {
        'conv1' : tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma)),
        'conv2' : tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma)),
        'flat3' : tf.Variable(tf.truncated_normal(shape=(1576, 400), mean = mu, stddev = sigma)),
        'flat4' : tf.Variable(tf.truncated_normal(shape=(400, 100), mean = mu, stddev = sigma)),
        'flat5' : tf.Variable(tf.truncated_normal(shape=(100, 43), mean = mu, stddev = sigma))
    }
    biases = {
        'conv1' : tf.Variable(tf.zeros(6)),
        'conv2' : tf.Variable(tf.zeros(16)),
        'flat3' : tf.Variable(tf.zeros(400)),
        'flat4' : tf.Variable(tf.zeros(100)),
        'flat5' : tf.Variable(tf.zeros(43))
    }

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1 = conv2d(x, weights['conv1'], biases['conv1'])

    # Activation
    conv1 = activation(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = maxpool2d(conv1, 2)

    # Dropout
    conv1 = dropout(conv1, keep_prob)

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2 = conv2d(conv1, weights['conv2'], biases['conv2'])

    # Activation.
    conv2 = activation(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = maxpool2d(conv2, 2)

    # Dropout
    conv2 = dropout(conv2, keep_prob)

    # Flatten. Input = 14x14x6 + 5x5x16. Output = 1576.
    flat_conv1 = tf.reshape(conv1, [-1, 14*14*6])
    flat_conv2 = tf.reshape(conv2, [-1, 5*5*16])
    flat3 = tf.concat(1, [flat_conv1, flat_conv2])

    # Layer 3: Fully Connected. Input = 1576. Output = 400.
    flat3 = linear(flat3, weights['flat3'], biases['flat3'])

    # Activation.
    flat3 = activation(flat3)

    # Dropout
    flat3 = dropout(flat3, keep_prob)

    # Layer 4: Fully Connected. Input = 400. Output = 100.
    flat4 = linear(flat3, weights['flat4'], biases['flat4'])

    # Activation.
    flat4 = activation(flat4)

    # Dropout
    flat4 = dropout(flat4, keep_prob)

    # Layer 5: Fully Connected. Input = 100. Output = 43.
    logits = linear(flat4, weights['flat5'], biases['flat5'])

    return logits


def Multi_Scale_LeNet_pooling_dropout_fc(x, keep_prob):

    weights = {
        'conv1' : tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma)),
        'conv2' : tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma)),
        'flat3' : tf.Variable(tf.truncated_normal(shape=(1576, 400), mean = mu, stddev = sigma)),
        'flat4' : tf.Variable(tf.truncated_normal(shape=(400, 100), mean = mu, stddev = sigma)),
        'flat5' : tf.Variable(tf.truncated_normal(shape=(100, 43), mean = mu, stddev = sigma))
    }
    biases = {
        'conv1' : tf.Variable(tf.zeros(6)),
        'conv2' : tf.Variable(tf.zeros(16)),
        'flat3' : tf.Variable(tf.zeros(400)),
        'flat4' : tf.Variable(tf.zeros(100)),
        'flat5' : tf.Variable(tf.zeros(43))
    }

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1 = conv2d(x, weights['conv1'], biases['conv1'])

    # Activation
    conv1 = activation(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = maxpool2d(conv1, 2)

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2 = conv2d(conv1, weights['conv2'], biases['conv2'])

    # Activation.
    conv2 = activation(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = maxpool2d(conv2, 2)

    # Flatten. Input = 14x14x6 + 5x5x16. Output = 1576.
    flat_conv1 = tf.reshape(conv1, [-1, 14*14*6])
    flat_conv2 = tf.reshape(conv2, [-1, 5*5*16])
    flat3 = tf.concat(1, [flat_conv1, flat_conv2])

    # Layer 3: Fully Connected. Input = 1576. Output = 400.
    flat3 = linear(flat3, weights['flat3'], biases['flat3'])

    # Activation.
    flat3 = activation(flat3)

    # Dropout
    flat3 = dropout(flat3, keep_prob)

    # Layer 4: Fully Connected. Input = 400. Output = 100.
    flat4 = linear(flat3, weights['flat4'], biases['flat4'])

    # Activation.
    flat4 = activation(flat4)

    # Dropout
    flat4 = dropout(flat4, keep_prob)

    # Layer 5: Fully Connected. Input = 100. Output = 43.
    logits = linear(flat4, weights['flat5'], biases['flat5'])

    return logits


def Inception_dropout_fc(x, keep_prob):

    weights = {
        'conv1' : tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma)),

        'conv2_a_1_1' : tf.Variable(tf.truncated_normal(shape=(1, 1, 6, 8), mean = mu, stddev = sigma)),
        'conv2_b_1_1' : tf.Variable(tf.truncated_normal(shape=(1, 1, 6, 8), mean = mu, stddev = sigma)),
        'conv2_b_3_3' : tf.Variable(tf.truncated_normal(shape=(3, 3, 8, 32), mean = mu, stddev = sigma)),
        'conv2_c_1_1' : tf.Variable(tf.truncated_normal(shape=(1, 1, 6, 8), mean = mu, stddev = sigma)),
        'conv2_c_5_5' : tf.Variable(tf.truncated_normal(shape=(5, 5, 8, 16), mean = mu, stddev = sigma)),
        'conv2_d_1_1' : tf.Variable(tf.truncated_normal(shape=(1, 1, 6, 8), mean = mu, stddev = sigma)),

        'conv3_a_1_1' : tf.Variable(tf.truncated_normal(shape=(1, 1, 64, 80), mean = mu, stddev = sigma)),
        'conv3_b_1_1' : tf.Variable(tf.truncated_normal(shape=(1, 1, 64, 80), mean = mu, stddev = sigma)),
        'conv3_b_3_3' : tf.Variable(tf.truncated_normal(shape=(3, 3, 80, 232), mean = mu, stddev = sigma)),
        'conv3_c_1_1' : tf.Variable(tf.truncated_normal(shape=(1, 1, 64, 80), mean = mu, stddev = sigma)),
        'conv3_c_5_5' : tf.Variable(tf.truncated_normal(shape=(5, 5, 80, 120), mean = mu, stddev = sigma)),
        'conv3_d_1_1' : tf.Variable(tf.truncated_normal(shape=(1, 1, 64, 80), mean = mu, stddev = sigma)),

        'full_conn1' : tf.Variable(tf.truncated_normal(shape=(8192, 1420), mean = mu, stddev = sigma)),
        'full_conn2' : tf.Variable(tf.truncated_normal(shape=(1420, 250), mean = mu, stddev = sigma)),
        'full_conn3' : tf.Variable(tf.truncated_normal(shape=(250, 43), mean = mu, stddev = sigma))
    }

    biases = {
        'conv1' : tf.Variable(tf.zeros(6)),
        'conv2_a_1_1' : tf.Variable(tf.zeros(8)),
        'conv2_b_1_1' : tf.Variable(tf.zeros(8)),
        'conv2_b_3_3' : tf.Variable(tf.zeros(32)),
        'conv2_c_1_1' : tf.Variable(tf.zeros(8)),
        'conv2_c_5_5' : tf.Variable(tf.zeros(16)),
        'conv2_d_1_1' : tf.Variable(tf.zeros(8)),
        'conv3_a_1_1' : tf.Variable(tf.zeros(80)),
        'conv3_b_1_1' : tf.Variable(tf.zeros(80)),
        'conv3_b_3_3' : tf.Variable(tf.zeros(232)),
        'conv3_c_1_1' : tf.Variable(tf.zeros(80)),
        'conv3_c_5_5' : tf.Variable(tf.zeros(120)),
        'conv3_d_1_1' : tf.Variable(tf.zeros(80)),
        'full_conn1' : tf.Variable(tf.zeros(1420)),
        'full_conn2' : tf.Variable(tf.zeros(250)),
        'full_conn3' : tf.Variable(tf.zeros(43))
    }

    # Layer 1: Normal Convolution. Input = 32x32x3. Output = 32x32x6
    conv1 = conv2d(x, weights['conv1'], biases['conv1'], padding='SAME')
    
    # Activation
    conv1 = activation(conv1)

    # Pooling. Input = 32x32x6. Output 16x16x6.
    conv1 = maxpool2d(conv1, 2)

    
    # Layer 2: Inception. Input 16x16x6. Output = 16x16x64
    conv2_a = conv2d(conv1, weights['conv2_a_1_1'], biases['conv2_a_1_1'], padding='SAME')
    conv2_b = conv2d(conv1, weights['conv2_b_1_1'], biases['conv2_b_1_1'], padding='SAME')
    conv2_b = conv2d(conv2_b, weights['conv2_b_3_3'], biases['conv2_b_3_3'], padding='SAME')
    conv2_c = conv2d(conv1, weights['conv2_c_1_1'], biases['conv2_c_1_1'], padding='SAME')
    conv2_c = conv2d(conv2_c, weights['conv2_c_5_5'], biases['conv2_c_5_5'], padding='SAME')
    conv2_d = maxpool2d(conv1, 3, 1)
    conv2_d = conv2d(conv2_d, weights['conv2_d_1_1'], biases['conv2_d_1_1'], padding='SAME')

    conv2 = tf.concat(3, [conv2_a, conv2_b, conv2_c, conv2_d])

    # Activation
    conv2 = activation(conv2)

    # Pooling. Input = 16x16x64. Output 8x8x64.
    conv2 = maxpool2d(conv2, 2)


    # Layer 3: Inception. Input 8x8x64. Output = 8x8x512
    conv3_a = conv2d(conv2, weights['conv3_a_1_1'], biases['conv3_a_1_1'], padding='SAME')
    conv3_b = conv2d(conv2, weights['conv3_b_1_1'], biases['conv3_b_1_1'], padding='SAME')
    conv3_b = conv2d(conv3_b, weights['conv3_b_3_3'], biases['conv3_b_3_3'], padding='SAME')
    conv3_c = conv2d(conv2, weights['conv3_c_1_1'], biases['conv3_c_1_1'], padding='SAME')
    conv3_c = conv2d(conv3_c, weights['conv3_c_5_5'], biases['conv3_c_5_5'], padding='SAME')
    conv3_d = maxpool2d(conv2, 3, 1)
    conv3_d = conv2d(conv3_d, weights['conv3_d_1_1'], biases['conv3_d_1_1'], padding='SAME')

    conv3 = tf.concat(3, [conv3_a, conv3_b, conv3_c, conv3_d])

    # Activation
    conv3 = activation(conv3)

    # Pooling. Input = 8x8x512. Output = 4x4x512
    conv3 = maxpool2d(conv3, 2)


    # Layer 4: Fully Connected. Input 8192. Output 1420
    flat = tf.reshape(conv3, [-1, 4*4*512])
    full1 = linear(flat, weights['full_conn1'], biases['full_conn1'])

    # Activation
    full1 = activation(full1)

    # Dropout
    full1 = dropout(full1, keep_prob)

    # Layer 5: Fully Connected. Input 1420. Output 250
    full2 = linear(full1, weights['full_conn2'], biases['full_conn2'])

    # Activation
    full2 = activation(full2)

    # Dropout
    full2 = dropout(full2, keep_prob)

    # Layer 6: Fully Connected. Input 250. Output 43
    logits = linear(full2, weights['full_conn3'], biases['full_conn3'])

    return logits


def Inception2_dropout_fc(x, keep_prob):

    ### Based on Traffic Sign Classification Using Deep Inception Based Convolution Networks
    ###   by Mrinhal Haloi

    weights = {
        'conv1' : tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma)),

        'conv2_a_1_1' : tf.Variable(tf.truncated_normal(shape=(1, 1, 6, 8), mean = mu, stddev = sigma)),
        'conv2_b_1_1' : tf.Variable(tf.truncated_normal(shape=(1, 1, 6, 8), mean = mu, stddev = sigma)),
        'conv2_b_3_3' : tf.Variable(tf.truncated_normal(shape=(3, 3, 8, 32), mean = mu, stddev = sigma)),
        'conv2_c_1_1' : tf.Variable(tf.truncated_normal(shape=(1, 1, 6, 8), mean = mu, stddev = sigma)),
        'conv2_c_5_5' : tf.Variable(tf.truncated_normal(shape=(5, 5, 8, 16), mean = mu, stddev = sigma)),
        'conv2_d_1_1' : tf.Variable(tf.truncated_normal(shape=(1, 1, 6, 8), mean = mu, stddev = sigma)),
        'conv2_d_3_3' : tf.Variable(tf.truncated_normal(shape=(3, 3, 8, 8), mean = mu, stddev = sigma)),

        'conv3_a_1_1' : tf.Variable(tf.truncated_normal(shape=(1, 1, 64, 80), mean = mu, stddev = sigma)),
        'conv3_b_1_1' : tf.Variable(tf.truncated_normal(shape=(1, 1, 64, 80), mean = mu, stddev = sigma)),
        'conv3_b_3_3' : tf.Variable(tf.truncated_normal(shape=(3, 3, 80, 232), mean = mu, stddev = sigma)),
        'conv3_c_1_1' : tf.Variable(tf.truncated_normal(shape=(1, 1, 64, 80), mean = mu, stddev = sigma)),
        'conv3_c_5_5' : tf.Variable(tf.truncated_normal(shape=(5, 5, 80, 120), mean = mu, stddev = sigma)),
        'conv3_d_1_1' : tf.Variable(tf.truncated_normal(shape=(1, 1, 64, 80), mean = mu, stddev = sigma)),
        'conv3_d_3_3' : tf.Variable(tf.truncated_normal(shape=(3, 3, 80, 80), mean = mu, stddev = sigma)),

        'full_conn1' : tf.Variable(tf.truncated_normal(shape=(8192, 1420), mean = mu, stddev = sigma)),
        'full_conn2' : tf.Variable(tf.truncated_normal(shape=(1420, 250), mean = mu, stddev = sigma)),
        'full_conn3' : tf.Variable(tf.truncated_normal(shape=(250, 43), mean = mu, stddev = sigma))
    }

    biases = {
        'conv1' : tf.Variable(tf.zeros(6)),
        'conv2_a_1_1' : tf.Variable(tf.zeros(8)),
        'conv2_b_1_1' : tf.Variable(tf.zeros(8)),
        'conv2_b_3_3' : tf.Variable(tf.zeros(32)),
        'conv2_c_1_1' : tf.Variable(tf.zeros(8)),
        'conv2_c_5_5' : tf.Variable(tf.zeros(16)),
        'conv2_d_1_1' : tf.Variable(tf.zeros(8)),
        'conv2_d_3_3' : tf.Variable(tf.zeros(8)),
        'conv3_a_1_1' : tf.Variable(tf.zeros(80)),
        'conv3_b_1_1' : tf.Variable(tf.zeros(80)),
        'conv3_b_3_3' : tf.Variable(tf.zeros(232)),
        'conv3_c_1_1' : tf.Variable(tf.zeros(80)),
        'conv3_c_5_5' : tf.Variable(tf.zeros(120)),
        'conv3_d_1_1' : tf.Variable(tf.zeros(80)),
        'conv3_d_3_3' : tf.Variable(tf.zeros(80)),
        'full_conn1' : tf.Variable(tf.zeros(1420)),
        'full_conn2' : tf.Variable(tf.zeros(250)),
        'full_conn3' : tf.Variable(tf.zeros(43))
    }

    # Layer 1: Normal Convolution. Input = 32x32x3. Output = 32x32x6
    conv1 = conv2d(x, weights['conv1'], biases['conv1'], padding='SAME')
    
    # Activation
    conv1 = activation(conv1)

    # Pooling. Input = 32x32x6. Output 16x16x6.
    conv1 = maxpool2d(conv1, 2)

    
    # Layer 2: Inception. Input 16x16x6. Output = 16x16x64
    conv2_a = conv2d(conv1, weights['conv2_a_1_1'], biases['conv2_a_1_1'], padding='SAME')
    conv2_b = conv2d(conv1, weights['conv2_b_1_1'], biases['conv2_b_1_1'], padding='SAME')
    conv2_b = conv2d(conv2_b, weights['conv2_b_3_3'], biases['conv2_b_3_3'], padding='SAME')
    conv2_c = conv2d(conv1, weights['conv2_c_1_1'], biases['conv2_c_1_1'], padding='SAME')
    conv2_c = conv2d(conv2_c, weights['conv2_c_5_5'], biases['conv2_c_5_5'], padding='SAME')
    conv2_d = conv2d(conv1, weights['conv2_d_1_1'], biases['conv2_d_1_1'], padding='SAME')
    conv2_d = conv2d(conv2_d, weights['conv2_d_3_3'], biases['conv2_d_3_3'], padding='SAME')
    conv2_d = maxpool2d(conv2_d, 3, 1)

    conv2 = tf.concat(3, [conv2_a, conv2_b, conv2_c, conv2_d])

    # Activation
    conv2 = activation(conv2)

    # Pooling. Input = 16x16x64. Output 8x8x64.
    conv2 = maxpool2d(conv2, 2)


    # Layer 3: Inception. Input 8x8x64. Output = 8x8x512
    conv3_a = conv2d(conv2, weights['conv3_a_1_1'], biases['conv3_a_1_1'], padding='SAME')
    conv3_b = conv2d(conv2, weights['conv3_b_1_1'], biases['conv3_b_1_1'], padding='SAME')
    conv3_b = conv2d(conv3_b, weights['conv3_b_3_3'], biases['conv3_b_3_3'], padding='SAME')
    conv3_c = conv2d(conv2, weights['conv3_c_1_1'], biases['conv3_c_1_1'], padding='SAME')
    conv3_c = conv2d(conv3_c, weights['conv3_c_5_5'], biases['conv3_c_5_5'], padding='SAME')
    conv3_d = conv2d(conv2, weights['conv3_d_1_1'], biases['conv3_d_1_1'], padding='SAME')
    conv3_d = conv2d(conv3_d, weights['conv3_d_3_3'], biases['conv3_d_3_3'], padding='SAME')
    conv3_d = maxpool2d(conv3_d, 3, 1)

    conv3 = tf.concat(3, [conv3_a, conv3_b, conv3_c, conv3_d])

    # Activation
    conv3 = activation(conv3)

    # Pooling. Input = 8x8x512. Output = 4x4x512
    conv3 = maxpool2d(conv3, 2)


    # Layer 4: Fully Connected. Input 8192. Output 1420
    flat = tf.reshape(conv3, [-1, 4*4*512])
    full1 = linear(flat, weights['full_conn1'], biases['full_conn1'])

    # Activation
    full1 = activation(full1)

    # Dropout
    full1 = dropout(full1, keep_prob)

    # Layer 5: Fully Connected. Input 1420. Output 250
    full2 = linear(full1, weights['full_conn2'], biases['full_conn2'])

    # Activation
    full2 = activation(full2)

    # Dropout
    full2 = dropout(full2, keep_prob)

    # Layer 6: Fully Connected. Input 250. Output 43
    logits = linear(full2, weights['full_conn3'], biases['full_conn3'])

    return logits


def Multi_Scale_LeNet3_pooling_dropout_fc(x, keep_prob):

    weights = {
        'conv1' : tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma)),
        'conv2' : tf.Variable(tf.truncated_normal(shape=(3, 3, 6, 16), mean = mu, stddev = sigma)),
        'conv3' : tf.Variable(tf.truncated_normal(shape=(3, 3, 16, 32), mean = mu, stddev = sigma)),
        'flat4' : tf.Variable(tf.truncated_normal(shape=(2552, 800), mean = mu, stddev = sigma)),
        'flat5' : tf.Variable(tf.truncated_normal(shape=(800, 200), mean = mu, stddev = sigma)),
        'flat6' : tf.Variable(tf.truncated_normal(shape=(200, 84), mean = mu, stddev = sigma)),
        'flat7' : tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    }
    biases = {
        'conv1' : tf.Variable(tf.zeros(6)),
        'conv2' : tf.Variable(tf.zeros(16)),
        'conv3' : tf.Variable(tf.zeros(32)),
        'flat4' : tf.Variable(tf.zeros(800)),
        'flat5' : tf.Variable(tf.zeros(200)),
        'flat6' : tf.Variable(tf.zeros(84)),
        'flat7' : tf.Variable(tf.zeros(43))
    }

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1 = conv2d(x, weights['conv1'], biases['conv1'])

    # Activation
    conv1 = activation(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = maxpool2d(conv1, 2)

    # Layer 2: Convolutional. Output = 12x12x16.
    conv2 = conv2d(conv1, weights['conv2'], biases['conv2'])

    # Activation.
    conv2 = activation(conv2)

    # Layer 3: Convolutional. Output = 10x10x32.
    conv3 = conv2d(conv2, weights['conv3'], biases['conv3'])

    # Activation.
    conv3 = activation(conv3)

    # Pooling. Input = 10x10x32. Output = 5x5x32.
    conv3 = maxpool2d(conv3, 2)

    # Flatten. Input = 14x14x6 + 6x6x16 + 5x5x32. Output = 2552.
    flat_conv1 = tf.reshape(conv1, [-1, 14*14*6])
    flat_conv2 = tf.reshape(maxpool2d(conv2, 2), [-1, 6*6*16])
    flat_conv3 = tf.reshape(conv3, [-1, 5*5*32])
    flat3 = tf.concat(1, [flat_conv1, flat_conv2, flat_conv3])

    # Layer 4: Full Connected. Input 2552. Output 800.
    flat4 = linear(flat3, weights['flat4'], biases['flat4'])

    # Activation.
    flat4 = activation(flat4)

    # Dropout
    flat4 = dropout(flat4, keep_prob)

    # Layer 5: Fully Connected. Input = 800. Output = 200.
    flat5 = linear(flat4, weights['flat5'], biases['flat5'])

    # Activation.
    flat5 = activation(flat5)

    # Dropout
    flat5 = dropout(flat5, keep_prob)

    # Layer 6: Fully Connected. Input = 200. Output = 84.
    flat6 = linear(flat5, weights['flat6'], biases['flat6'])

    # Activation.
    flat6 = activation(flat6)

    # Dropout
    flat6 = dropout(flat6, keep_prob)

    # Layer 7: Fully Connected. Input = 84. Output = 43.
    logits = linear(flat6, weights['flat7'], biases['flat7'])

    return logits


def Multi_Scale_LeNet_dropout(x, keep_prob):

    print("Don't waste time running this one")
    return

    ### Did not work as well as LeNet and LeNet dropout
    ### Things to try:
    ### 1) add max pooling back in
    ### 2) add 3rd convolution layer
    ### 3) dropout only the fully connected layers

    weights = {
        'conv1' : tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma)),
        'conv2' : tf.Variable(tf.truncated_normal(shape=(9, 9, 6, 16), mean = mu, stddev = sigma)),
        'flat3' : tf.Variable(tf.truncated_normal(shape=(6304, 1260), mean = mu, stddev = sigma)),
        'flat4' : tf.Variable(tf.truncated_normal(shape=(1260, 252), mean = mu, stddev = sigma)),
        'flat5' : tf.Variable(tf.truncated_normal(shape=(252, 43), mean = mu, stddev = sigma))
    }
    biases = {
        'conv1' : tf.Variable(tf.zeros(6)),
        'conv2' : tf.Variable(tf.zeros(16)),
        'flat3' : tf.Variable(tf.zeros(1260)),
        'flat4' : tf.Variable(tf.zeros(252)),
        'flat5' : tf.Variable(tf.zeros(43))
    }

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1 = conv2d(x, weights['conv1'], biases['conv1'])

    # Activation
    conv1 = activation(conv1)

    # Dropout
    conv1 = dropout(conv1, keep_prob)

    # Layer 2: Convolutional. Input = 28x28x6. Output = 10x10x16.
    conv2 = conv2d(conv1, weights['conv2'], biases['conv2'], strides=2)

    # Activation.
    conv2 = activation(conv2)

    # Dropout
    conv2 = dropout(conv2, keep_prob)

    # Flatten. Input = 28x28x6 + 10x10x16. Output = 6304.
    flat_conv1 = tf.reshape(conv1, [-1, 28*28*6])
    flat_conv2 = tf.reshape(conv2, [-1, 10*10*16])
    flat3 = tf.concat(1, [flat_conv1, flat_conv2])

    # Layer 3: Fully Connected. Input = 6304. Output = 1260.
    flat3 = linear(flat3, weights['flat3'], biases['flat3'])

    # Activation.
    flat3 = activation(flat3)

    # Layer 4: Fully Connected. Input = 1260. Output = 252.
    flat4 = linear(flat3, weights['flat4'], biases['flat4'])

    # Activation.
    flat4 = activation(flat4)

    # Layer 5: Fully Connected. Input = 252. Output = 43.
    logits = linear(flat4, weights['flat5'], biases['flat5'])

    return logits

