import argparse
import logging
import shutil

import chainer
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.propagate = False

parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser.add_argument('--run_id', type=int, default=1)
parser.add_argument('--dataset', type=str, default='mnist',
                    help='binary_minst | mnist | cifar10 | svhn')
parser.add_argument('--missingness_type', type=str, default='independent',
                    help='independent (MCAR) | dependent (MNAR)')
parser.add_argument('--missingness_complexity', type=str, default='simple',
                    help='simple | complex')
parser.add_argument('--marginal_ll_mc_samples', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--save_imgs', type=str2bool, default=False)
parser.add_argument('--likelihood', type=str, default='bernoulli',
                    help='bernoulli | continuous_bernoulli | logistic_mixture')
parser.add_argument('--k', type=int, default=3, help='number of logistic mixture components')

args = parser.parse_args()

if args.save_imgs:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

# see: https://arxiv.org/pdf/1202.2745.pdf for conv architectures
# Multi-column Deep Neural Networks for Image Classification
if 'mnist' in args.dataset:
    img_channels = 3
    data = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = data.load_data()
    y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)

    if args.dataset == 'binary_mnist':
        x_train = x_train/255.0
        x_test = x_test/255.0

        x_train[x_train >= 0.5] = 1.0
        x_train[x_train < 0.5] = 0.0
        x_test[x_test >= 0.5] = 1.0
        x_test[x_test < 0.5] = 0.0
    else:
        x_train = x_train/255.0
        x_test = x_test/255.0

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                          train_size=50000,
                                                          test_size=10000)
    
    z_dim = 50
    input_units = x_train.shape[1] * x_train.shape[2]
    encoder_layers = [(20, 5, 2), (40, 5, 2)]
    encoder_shapes = [(28, 28, 1), (14, 14, 20), (7, 7, 40)]
    encoder_shapes_pre_pool = encoder_shapes
    p_x_layers = [7*7*20,
                  ([-1, 7, 7, 20], 40, 5, 2, tf.nn.relu),
                  (None, 20, 5, 2, tf.nn.relu)]
    p_x_b_z_layers = [(None, 10, 5, 1, tf.nn.relu),
                      (None, 10, 5, 1, tf.nn.relu),
                      (None, 3 * args.k, 3, 1, None) if args.likelihood == 'logistic_mixture'
                      else (None, 1, 3, 1, None)]
elif 'svhn' in args.dataset:
    img_channels = 3
    train, test = chainer.datasets.get_svhn(withlabel=True, scale=255.0)
    x_train = np.asarray(map(lambda t: t[0], train))
    x_test = np.asarray(map(lambda t: t[0], test))
    y_train = np.asarray(map(lambda t: t[1], train))
    y_test = np.asarray(map(lambda t: t[1], test))

    x_train = np.concatenate((np.expand_dims(x_train[:, 0, :, :], -1),
                              np.expand_dims(x_train[:, 1, :, :], -1),
                              np.expand_dims(x_train[:, 2, :, :], -1)), axis=3)
    x_test = np.concatenate((np.expand_dims(x_test[:, 0, :, :], -1),
                             np.expand_dims(x_test[:, 1, :, :], -1),
                             np.expand_dims(x_test[:, 2, :, :], -1)), axis=3)

    x_train = x_train/255.0
    x_test = x_test/255.0

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                          train_size=63257,
                                                          test_size=10000)

    z_dim = 200
    input_units = x_train.shape[1] * x_train.shape[2]
    encoder_layers = [(40, 3, 2), (60, 3, 2), (60, 5, 2)]
    encoder_shapes = [(32, 32, 3), (16, 16, 40), (8, 8, 60), (4, 4, 60)]
    encoder_shapes_pre_pool = encoder_shapes
    p_x_layers = [4*4*60,
                  ([-1, 4, 4, 60], 60, 3, 2, tf.nn.relu),
                  (None, 60, 3, 2, tf.nn.relu),
                  (None, 40, 5, 2, tf.nn.relu)]
    p_x_b_z_layers = [(None, 30, 5, 1, tf.nn.relu),
                      (None, 30, 5, 1, tf.nn.relu),
                      (None, 9 * args.k, 3, 1, None) if args.likelihood == 'logistic_mixture'
                      else (None, 3, 3, 1, None)]
elif args.dataset == 'cifar10':
    img_channels = 3
    data = tf.keras.datasets.cifar10

    (x_train, y_train), (x_test, y_test) = data.load_data()
    y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)

    x_train = x_train/255.0
    x_test = x_test/255.0

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                          train_size=40000,
                                                          test_size=10000)

    z_dim = 200
    input_units = x_train.shape[1] * x_train.shape[2]
    encoder_layers = [(40, 3, 2), (60, 3, 2), (60, 5, 2)]
    encoder_shapes = [(32, 32, 3), (16, 16, 40), (8, 8, 60), (4, 4, 60)]
    encoder_shapes_pre_pool = encoder_shapes
    p_x_layers = [4*4*60,
                  ([-1, 4, 4, 60], 60, 3, 2, tf.nn.relu),
                  (None, 60, 3, 2, tf.nn.relu),
                  (None, 40, 5, 2, tf.nn.relu)]
    p_x_b_z_layers = [(None, 30, 5, 1, tf.nn.relu),
                      (None, 30, 5, 1, tf.nn.relu),
                      (None, 9 * args.k, 3, 1, None) if args.likelihood == 'logistic_mixture'
                      else (None, 3, 3, 1, None)]

max_epochs = 400
max_epochs_without_improvement = 12
img_dim = x_train.shape[1]
missingness_block_size = 7
if 'mnist' in args.dataset:
    num_missing_blocks = 8
elif args.dataset == 'svhn':
    num_missing_blocks = 9
elif args.dataset == 'cifar10':
    num_missing_blocks = 9

def encoder(inputs, model_type=None):
    net = inputs

    for i, layer in enumerate(encoder_layers):
        if isinstance(layer, tuple):
            filters, kernel_size, strides = layer

            if i == 0:
                if 'mnist' in args.dataset:
                    in_channels = 1
                elif args.dataset == 'cifar10' or args.dataset == 'svhn':
                    in_channels = 3
                
                if '_ind' in model_type:
                    in_channels += 1
            else:
                in_channels = encoder_layers[i-1][0]

            w = tf.get_variable('layer_{0}_w'.format(i),
                                shape=(kernel_size, kernel_size, in_channels, filters),
                                initializer=tf.contrib.layers.xavier_initializer())

            net = tf.nn.conv2d(net, w, strides=(1, strides, strides, 1), padding='SAME')

            bias = tf.get_variable('layer_{0}_bias'.format(i),
                                   shape=encoder_shapes_pre_pool[i+1][2],
                                   initializer=tf.initializers.zeros())

            net = tf.nn.bias_add(net, bias)
            net = tf.nn.relu(net)
        else:
            net = tf.contrib.layers.flatten(net)
            net = tf.contrib.layers.fully_connected(net, layer, activation_fn=tf.nn.relu)
            
    net = tf.contrib.layers.flatten(net)

    mu = tf.contrib.layers.fully_connected(net, z_dim, activation_fn=None)
    sigma = tf.contrib.layers.fully_connected(net, z_dim,
                                              activation_fn=tf.nn.softplus,
                                              biases_initializer=tf.constant_initializer(
                                                np.log(np.e - np.ones(z_dim)))) + 1e-3
    
    return mu, sigma

def decoder(z, b=None):
    net = z
    for i, layer in enumerate(p_x_layers):
        if isinstance(layer, tuple):
            input_size, filters, kernel_size, strides, activation = layer
            if input_size is not None:
                net = tf.reshape(net, input_size)

            net = tf.layers.conv2d_transpose(net, filters, kernel_size,
                                             strides=strides, activation=activation,
                                             padding='SAME')

            bias = tf.get_variable('p_x_layer_{0}_bias'.format(i),
                                   shape=p_x_layers[i][1],
                                   initializer=tf.initializers.zeros())

            net = tf.nn.bias_add(net, bias)
        else:
            net = tf.contrib.layers.fully_connected(net, layer, activation_fn=tf.nn.relu)

    if b is not None:
        if args.dataset in ('svhn', 'cifar10'):
            b = b[:, :, :, 0]
        b = tf.expand_dims(b, axis=3)
        net = tf.concat([net, b], axis=3)
    
    for i, layer in enumerate(p_x_b_z_layers):
        _, filters, kernel_size, strides, activation = layer

        if i == 0:
            if b is not None:
                in_channels = p_x_layers[-1][1] + 1
            else:
                in_channels = p_x_layers[-1][1]
        else:
            in_channels = p_x_b_z_layers[i-1][1]

        w = tf.get_variable('p_x_b_z_layer_{0}_w'.format(i),
                            shape=(kernel_size, kernel_size, in_channels, filters),
                            initializer=tf.contrib.layers.xavier_initializer())

        net = tf.nn.conv2d(net, w, strides=(1, strides, strides, 1), padding='SAME')

        bias = tf.get_variable('p_x_b_z_layer_{0}_bias'.format(i),
                               shape=p_x_b_z_layers[i][1],
                               initializer=tf.initializers.zeros())

        net = tf.nn.bias_add(net, bias)
        
        if activation is not None:
            net = activation(net)

    if args.likelihood == 'logistic_mixture':
        assert 'mnist' not in args.dataset
        mean_logit = []
        scale_logit = []
        pi_logit = []
        for i in range(img_channels):
            mean_logit.append(net[:, :, :, i*args.k:(i+1)*args.k])
            scale_logit.append(net[:, :, :, (img_channels*args.k + i*args.k):(img_channels*args.k + (i+1)*args.k)])
            pi_logit.append(net[:, :, :, (2*img_channels*args.k + i*args.k):(2*img_channels*args.k + (i+1)*args.k)])
    else:
        mean_logit = net
        scale_logit = None
        pi_logit = None

    return mean_logit, scale_logit, pi_logit

def log_sum_exp(x):
    """credit: https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py"""
    axis = len(x.get_shape())-1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keepdims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x-m2), axis))

def cont_bern_log_norm(lam, l_lim=0.49, u_lim=0.51):
    """credit: https://github.com/gabloa/cont_bern
    Computes the log normalizing constant of a continuous Bernoulli distribution in a numerically stable way.
    Returns the log normalizing constant for lam in (0, l_lim) U (u_lim, 1)
    and a Taylor approximation in [l_lim, u_lim]"""
    cut_lam = tf.where(tf.logical_or(tf.less(lam, l_lim), tf.greater(lam, u_lim)), lam, l_lim * tf.ones_like(lam))
    log_norm = tf.log(tf.abs(2.0 * tf.atanh(1 - 2.0 * cut_lam))) - tf.log(tf.abs(1 - 2.0 * cut_lam))
    taylor = 4.0/3.0 * tf.pow(lam - 0.5, 2) + 104.0/45.0 * tf.pow(lam - 0.5, 4)
    return tf.where(tf.logical_or(tf.less(lam, l_lim), tf.greater(lam, u_lim)), log_norm, taylor)

def model(features, labels, mode, params):
    if 'mnist' in args.dataset:
        x = tf.reshape(tf.feature_column.input_layer(features, params['feature_columns'][0]),
                       [-1, img_dim, img_dim])
        x_input = tf.reshape(tf.feature_column.input_layer(features, params['feature_columns'][1]),
                       [-1, img_dim, img_dim])
        b = tf.reshape(tf.feature_column.input_layer(features, params['feature_columns'][2]),
                       [-1, img_dim, img_dim])

        mu, sigma, = encoder(tf.stack([x_input, b], axis=3)
                             if '_ind' in params['model_type']
                             else tf.expand_dims(x_input, -1),
                             model_type=params['model_type'])
    elif args.dataset == 'cifar10' or args.dataset == 'svhn':
        x = tf.reshape(tf.feature_column.input_layer(features, params['feature_columns'][0]),
                       [-1, img_dim, img_dim, 3])
        x_input = tf.reshape(tf.feature_column.input_layer(features, params['feature_columns'][1]),
                       [-1, img_dim, img_dim, 3])
        b = tf.reshape(tf.feature_column.input_layer(features, params['feature_columns'][2]),
                       [-1, img_dim, img_dim, 3])

        mu, sigma, = encoder(tf.concat([x_input, tf.expand_dims(b[:, :, :, 0], axis=3)], axis=3)
                             if '_ind' in params['model_type']
                             else x_input,
                             model_type=params['model_type'])

    q_z = tf.distributions.Normal(mu, sigma)
    
    p_z = tf.distributions.Normal(loc=np.zeros(z_dim, dtype=np.float32), scale=np.ones(z_dim, dtype=np.float32))
    
    kl = tf.reduce_mean(tf.reduce_sum(tf.distributions.kl_divergence(q_z, p_z), axis=1))

    def log_probs(z_sample):
        x_logits, scale_logit, pi_logit = decoder(z_sample,
                                                  b=b if '_decoder_b' in params['model_type'] else None)
        if args.dataset == 'mnist':
            x_logits = tf.squeeze(x_logits, axis=-1)
        if 'bernoulli' in args.likelihood:
            x_pred = tf.nn.sigmoid(x_logits)
        if args.likelihood == 'logistic_mixture':
            scale = [tf.exp(scale_logit[i]) + 1e-2
                     for i in range(img_channels)]

            pi = [tf.nn.softmax(pi_logit[i], axis=3)
                  for i in range(img_channels)]

            if args.k == 1:
                x_pred = tf.concat([tf.expand_dims(tf.reduce_sum(x_logits[i], axis=3), axis=3)
                                    for i in range(img_channels)], axis=3)
            else:
                x_pred = tf.concat([tf.expand_dims(tf.reduce_sum(pi[i] * x_logits[i], axis=3), axis=3)
                                    for i in range(img_channels)], axis=3)

        if 'bernoulli' in args.likelihood:
            # valid for even real valued MNIST: http://ruishu.io/2018/03/19/bernoulli-vae/
            log_prob_full = -1 * tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_logits)
            if args.likelihood == 'continuous_bernoulli':
                # log_prob_full += cont_bern_log_norm(tf.clip_by_value(x_pred, 1e-2, 1.0 - 1e-2))
                log_prob_full += cont_bern_log_norm(x_pred)
                
            log_prob = tf.reduce_mean(tf.reduce_sum(b * log_prob_full,
                                                    axis=[2,1] if 'mnist' in args.dataset else [3,2,1]))
            imputation_log_prob = tf.reduce_mean(tf.reduce_sum((1.0 - b) * log_prob_full,
                                                 axis=[2,1] if 'mnist' in args.dataset else [3,2,1]))
        elif args.likelihood == 'logistic_mixture':
            log_prob_full = []
            for i in range(img_channels):
                x_expanded = tf.concat([tf.expand_dims(x[:, :, :, i], axis=3)
                                        for _ in range(args.k)], axis=3)
                centered_x = x_expanded - x_logits[i]
                upper_in = (centered_x + (1.0/255.0))/scale[i]
                upper_cdf = tf.nn.sigmoid(upper_in)
                lower_in = (centered_x - (1.0/255.0))/scale[i]
                lower_cdf = tf.nn.sigmoid(lower_in)
                cdf_delta = upper_cdf - lower_cdf
                
                mid_in = centered_x/scale[i]
                log_pdf_mid = mid_in - scale_logit[i] - 2.0 * tf.nn.softplus(mid_in)
                log_cdf_plus = upper_in - tf.nn.softplus(upper_in)
                log_one_minus_cdf_min = -tf.nn.softplus(lower_in)
                
                log_prob_comp = tf.where(x_expanded < -0.999,
                                         log_cdf_plus,
                                         tf.where(x_expanded > 0.999,
                                                  log_one_minus_cdf_min,
                                                  tf.where(cdf_delta > 1e-5,
                                                           tf.log(tf.maximum(cdf_delta, 1e-12)),
                                                           log_pdf_mid - np.log(127.5))))
                if args.k > 1:
                    log_prob_comp += tf.log(pi[i])

                log_prob_full.append(log_sum_exp(log_prob_comp))

            log_prob_full = tf.concat([tf.expand_dims(log_prob_full[i], axis=3)
                                       for i in range(img_channels)], axis=3)

            log_prob = tf.reduce_mean(
                tf.reduce_sum(b * log_prob_full,
                              axis=[3,2,1]))
            imputation_log_prob = tf.reduce_mean(
                tf.reduce_sum((1.0 - b) * log_prob_full,
                              axis=[3,2,1]))

        return x_pred, log_prob, imputation_log_prob

    def log_probs_single_mc_sample():
        z_sample = q_z.sample()

        with tf.variable_scope('log_probs', reuse=tf.AUTO_REUSE):
            x_pred, log_prob, imputation_log_prob = log_probs(z_sample)
        return x_pred, log_prob, imputation_log_prob
    def log_probs_deterministic():
        z_sample = mu

        with tf.variable_scope('log_probs', reuse=tf.AUTO_REUSE):
            x_pred, log_prob, imputation_log_prob = log_probs(z_sample)
        return x_pred, log_prob, imputation_log_prob

    if mode == tf.estimator.ModeKeys.TRAIN:
        x_pred, log_prob, imputation_log_prob = log_probs_single_mc_sample()
    elif mode == tf.estimator.ModeKeys.EVAL:
        x_pred, log_prob, imputation_log_prob = log_probs_deterministic()
    elif mode == tf.estimator.ModeKeys.PREDICT:
        x_pred, _, _ = log_probs_deterministic()
        return tf.estimator.EstimatorSpec(mode,
            predictions={'x': tf.clip_by_value(x_pred if 'mnist' in args.dataset else (x_pred/2.0 + 0.5), 0.0, 1.0),
                         'z': mu},
            export_outputs={'y': tf.estimator.export.ClassificationOutput(
                scores=tf.clip_by_value(x_pred if 'mnist' in args.dataset else (x_pred/2.0 + 0.5), 0.0, 1.0))})
        
    elbo = log_prob - kl
    loss = -elbo

    tf.summary.scalar('elbo', elbo)
    tf.summary.scalar('kl', kl)
    tf.summary.scalar('log_prob_x', log_prob)
    tf.summary.scalar('loss', loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        se_full = tf.square(x - x_pred)
        mse = tf.reduce_mean(
            tf.reduce_sum(b * se_full,
                          axis=[2,1] if args.dataset == 'mnist' else [3,2,1]))
        imputation_mse = tf.reduce_mean(
            tf.reduce_sum((1.0 - b) * se_full,
                          axis=[2,1] if args.dataset == 'mnist' else [3,2,1]))

        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=
                                          {'log_prob_x': tf.metrics.mean(log_prob),
                                           'imputation_log_prob':  tf.metrics.mean(
                                               imputation_log_prob),
                                           'mse_x': tf.metrics.mean(mse),
                                           'imputation_mse': tf.metrics.mean(imputation_mse),
                                           'elbo': tf.metrics.mean(elbo),
                                           'kl': tf.metrics.mean(kl)})

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss,
            global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


if args.missingness_type == 'independent' and args.missingness_complexity == 'simple':
    b_train = np.ones_like(x_train)
    b_valid = np.ones_like(x_valid)
    b_test = np.ones_like(x_test)
    for b in range(x_train.shape[0]):
        for _ in range(num_missing_blocks):
            x = np.random.choice(img_dim)
            y = np.random.choice(img_dim)
            if 'mnist' in args.dataset:
                b_train[b,
                        max(x - missingness_block_size/2, 0):min(x + missingness_block_size/2, img_dim),
                        max(y - missingness_block_size/2, 0):min(y + missingness_block_size/2, img_dim)] = 0.0
            elif args.dataset in ('cifar10', 'svhn'):
                b_train[b,
                        max(x - missingness_block_size/2, 0):min(x + missingness_block_size/2, img_dim),
                        max(y - missingness_block_size/2, 0):min(y + missingness_block_size/2, img_dim),
                        :] = 0.0
    for b in range(x_valid.shape[0]):
        for _ in range(num_missing_blocks):
            x = np.random.choice(img_dim)
            y = np.random.choice(img_dim)
            if 'mnist' in args.dataset:
                b_valid[b,
                        max(x - missingness_block_size/2, 0):min(x + missingness_block_size/2, img_dim),
                        max(y - missingness_block_size/2, 0):min(y + missingness_block_size/2, img_dim)] = 0.0
            elif args.dataset in ('cifar10', 'svhn'):
                b_valid[b,
                        max(x - missingness_block_size/2, 0):min(x + missingness_block_size/2, img_dim),
                        max(y - missingness_block_size/2, 0):min(y + missingness_block_size/2, img_dim),
                        :] = 0.0
    for b in range(x_test.shape[0]):
        for _ in range(num_missing_blocks):
            x = np.random.choice(img_dim)
            y = np.random.choice(img_dim)
            if 'mnist' in args.dataset:
                b_test[b,
                        max(x - missingness_block_size/2, 0):min(x + missingness_block_size/2, img_dim),
                        max(y - missingness_block_size/2, 0):min(y + missingness_block_size/2, img_dim)] = 0.0
            elif args.dataset in ('cifar10', 'svhn'):
                b_test[b,
                        max(x - missingness_block_size/2, 0):min(x + missingness_block_size/2, img_dim),
                        max(y - missingness_block_size/2, 0):min(y + missingness_block_size/2, img_dim),
                        :] = 0.0

elif args.missingness_type == 'dependent' or args.missingness_complexity == 'complex':
    if 'mnist' in args.dataset:
        mnar_blocks = [[5, 10], [5, 12], [7, 5], [7, 6], [9, 3], [9, 4],
                       [11, 2], [11, 3], [13, 1], [15, 1]]
    else:
        mnar_blocks = [[5, 12], [7, 5], [7, 6], [9, 3], [9, 4],
                       [11, 2], [11, 3], [13, 2], [15, 1], [17, 1]]
    if args.missingness_type == 'dependent':
        mnar_blocks = np.random.permutation(mnar_blocks).tolist()

    b_train = np.ones_like(x_train)
    b_valid = np.ones_like(x_valid)
    b_test = np.ones_like(x_test)
    for b in range(x_train.shape[0]):
        if args.missingness_type == 'dependent':
            missingness_block_size, num_missing_blocks = mnar_blocks[int(y_train[b])]
        else:
            missingness_block_size, num_missing_blocks = mnar_blocks[np.random.choice(len(mnar_blocks))]
        
        for _ in range(num_missing_blocks):
            x = missingness_block_size/2 + np.random.choice(img_dim - missingness_block_size)
            y = missingness_block_size/2 + np.random.choice(img_dim - missingness_block_size)
            if 'mnist' in args.dataset:
                b_train[b,
                        max(x - missingness_block_size/2, 0):min(x + missingness_block_size/2, img_dim),
                        max(y - missingness_block_size/2, 0):min(y + missingness_block_size/2, img_dim)] = 0.0
            elif args.dataset in ('cifar10', 'svhn'):
                b_train[b,
                        max(x - missingness_block_size/2, 0):min(x + missingness_block_size/2, img_dim),
                        max(y - missingness_block_size/2, 0):min(y + missingness_block_size/2, img_dim),
                        :] = 0.0
    for b in range(x_valid.shape[0]):
        if args.missingness_type == 'dependent':
            missingness_block_size, num_missing_blocks = mnar_blocks[int(y_valid[b])]
        else:
            missingness_block_size, num_missing_blocks = mnar_blocks[np.random.choice(len(mnar_blocks))]
        
        for _ in range(num_missing_blocks):
            x = missingness_block_size/2 + np.random.choice(img_dim - missingness_block_size)
            y = missingness_block_size/2 + np.random.choice(img_dim - missingness_block_size)
            if 'mnist' in args.dataset:
                b_valid[b,
                        max(x - missingness_block_size/2, 0):min(x + missingness_block_size/2, img_dim),
                        max(y - missingness_block_size/2, 0):min(y + missingness_block_size/2, img_dim)] = 0.0
            elif args.dataset in ('cifar10', 'svhn'):
                b_valid[b,
                        max(x - missingness_block_size/2, 0):min(x + missingness_block_size/2, img_dim),
                        max(y - missingness_block_size/2, 0):min(y + missingness_block_size/2, img_dim),
                        :] = 0.0
    for b in range(x_test.shape[0]):
        if args.missingness_type == 'dependent':
            missingness_block_size, num_missing_blocks = mnar_blocks[int(y_test[b])]
        else:
            missingness_block_size, num_missing_blocks = mnar_blocks[np.random.choice(len(mnar_blocks))]

        for _ in range(num_missing_blocks):
            x = missingness_block_size/2 + np.random.choice(img_dim - missingness_block_size)
            y = missingness_block_size/2 + np.random.choice(img_dim - missingness_block_size)
            if 'mnist' in args.dataset:
                b_test[b,
                       max(x - missingness_block_size/2, 0):min(x + missingness_block_size/2, img_dim),
                       max(y - missingness_block_size/2, 0):min(y + missingness_block_size/2, img_dim)] = 0.0
            elif args.dataset in ('cifar10', 'svhn'):
                b_test[b,
                       max(x - missingness_block_size/2, 0):min(x + missingness_block_size/2, img_dim),
                       max(y - missingness_block_size/2, 0):min(y + missingness_block_size/2, img_dim),
                       :] = 0.0

print 'Train fraction of pixels missing:', 1.0 - np.sum(b_train)/np.sum(np.ones_like(b_train))
print 'Valid fraction of pixels missing:', 1.0 - np.sum(b_valid)/np.sum(np.ones_like(b_valid))
print 'Test fraction of pixels missing:', 1.0 - np.sum(b_test)/np.sum(np.ones_like(b_test))

mu_hat = np.true_divide(np.sum(b_train * x_train, axis=0), np.sum(b_train, axis=0))
sigma_hat = np.true_divide(np.sum(b_train * (x_train - mu_hat)**2, axis=0), np.sum(b_train, axis=0))
sigma_hat = np.clip(sigma_hat, np.mean(sigma_hat)/10.0, np.percentile(sigma_hat, 95))
print 'Average pixel intensity:', np.mean(mu_hat)
print 'Average pixel variance:', np.mean(sigma_hat)

for model_type, title in [('VAE', 'Zero imputation'),
                   ('VAE_mean_imp', 'Mean imputation'),
                   ('VAE_ind', 'Zero imputation w/ encoder indicator variables'),
                   ('VAE_ind_decoder_b', 'Zero imputation w/ encoder + decoder indicator variables')]:
# for model_type, title in [('VAE_ind_decoder_b', 'Zero imputation w/ encoder + decoder indicator variables')]:
    if '_mean_imp' in model_type:
        x_train_input = b_train * x_train + (1.0 - b_train) * mu_hat
        x_valid_input = b_valid * x_valid + (1.0 - b_valid) * mu_hat
        x_test_input = b_test * x_test + (1.0 - b_test) * mu_hat
    else:
        x_train_input = b_train * x_train
        x_valid_input = b_valid * x_valid
        x_test_input = b_test * x_test

    if args.save_imgs and 'mnist' in args.dataset:
        def draw_digit(data, row, col, n):
            size = 28
            plt.subplot(row, col, n)
            plt.imshow(data)
            plt.gray()
            plt.axis('off')

        show_size = 10
        total = 0
        fig = plt.figure(figsize=(20,20), dpi=30)
        for i in range(show_size):
            for j in range(show_size):
                if i % 2 == 0:
                    draw_digit(x_train[(i/2)*show_size + j], show_size, show_size, total+1)
                else:
                    draw_digit(x_train_input[((i - 1)/2)*show_size + j], show_size, show_size, total+1)

                total += 1
        fig.suptitle(title, fontsize=35, verticalalignment='center', y=0.92)
        plt.savefig('{0}_{1}_{2}_{3}.png'.format(args.dataset, model_type, args.missingness_type,
                                                 args.missingness_complexity),
                    bbox_inches='tight')
        plt.close()
    elif args.save_imgs and args.dataset in ('cifar10', 'svhn'):
        def draw_digit(data, row, col, n):
            size = 32
            plt.subplot(row, col, n)
            plt.imshow(data)
            plt.axis('off')

        show_size = 10
        total = 0
        fig = plt.figure(figsize=(20,20), dpi=30)
        for i in range(show_size):
            for j in range(show_size):
                if i % 2 == 0:
                    draw_digit(x_train[(i/2)*show_size + j], show_size, show_size, total+1)
                else:
                    draw_digit(x_train_input[((i - 1)/2)*show_size + j], show_size, show_size, total+1)

                total += 1
        fig.suptitle(title, fontsize=35, verticalalignment='center', y=0.92)
        plt.savefig('{0}_{1}_{2}_{3}.png'.format(args.dataset, model_type, args.missingness_type,
                                                 args.missingness_complexity),
                    bbox_inches='tight')
        plt.close()

    feature_columns = [tf.feature_column.numeric_column(key='x', shape=[img_dim, img_dim]),
                       tf.feature_column.numeric_column(key='x_input', shape=[img_dim, img_dim]),
                       tf.feature_column.numeric_column(key='b', shape=[img_dim, img_dim]),
                       tf.feature_column.numeric_column(key='multiple_mc_samples', shape=[])]

    model_dir = '{0}_conv_{1}_{2}_{3}'.format(args.dataset, model_type,
                                              args.missingness_type, args.run_id)
    vae = tf.estimator.Estimator(
        model_fn=model,
        model_dir=model_dir,
        params={'feature_columns': feature_columns,
                'model_type': model_type},
        config=tf.estimator.RunConfig(
            save_summary_steps=100000,
            save_checkpoints_steps=100000,
            keep_checkpoint_max=2*max_epochs_without_improvement + 5,
            log_step_count_steps=100000))
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        {'x': x_train if 'mnist' in args.dataset else 2.0 * x_train - 1.0,
         'x_input': x_train_input if 'mnist' in args.dataset else 2.0 * x_train_input - 1.0,
         'b': b_train,
         'multiple_mc_samples': np.zeros(x_train.shape[0])},
        shuffle=True, batch_size=args.batch_size)

    valid_input_fn = tf.estimator.inputs.numpy_input_fn(
        {'x': x_valid if 'mnist' in args.dataset else 2.0 * x_valid - 1.0,
         'x_input': x_valid_input if 'mnist' in args.dataset else 2.0 * x_valid_input - 1.0,
         'b': b_valid,
         'multiple_mc_samples': np.ones(x_valid.shape[0])},
        shuffle=False, batch_size=args.batch_size)
    
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        {'x': x_test if 'mnist' in args.dataset else 2.0 * x_test - 1.0,
         'x_input': x_test_input if 'mnist' in args.dataset else 2.0 * x_test_input - 1.0,
         'b': b_test,
         'multiple_mc_samples': np.ones(x_test.shape[0])},
        shuffle=False, batch_size=args.batch_size)

    image_plot_input_fn = tf.estimator.inputs.numpy_input_fn(
        {'x': x_test[:30] if 'mnist' in args.dataset else 2.0 * x_test[:30] - 1.0,
         'x_input': x_test_input[:30] if 'mnist' in args.dataset else 2.0 * x_test_input[:30] - 1.0,
         'b': b_test[:30],
         'multiple_mc_samples': np.ones(x_test[:30].shape[0])},
        shuffle=False, batch_size=args.batch_size)

    best_valid_eval = None
    best_checkpoint = None
    best_estimator = None
    epochs_since_improvement = 0
    for i in range(max_epochs):
        vae.train(steps=None, input_fn=train_input_fn)

        eval_result = vae.evaluate(input_fn=valid_input_fn)
        logging.info('End of epoch evaluation (valid set): ' + str(eval_result))

        if args.save_imgs and 'mnist' in args.dataset:
            recons = vae.predict(image_plot_input_fn)
            images = [res['x'] for res in recons]

            def draw_digit(data, row, col, n):
                size = 28
                plt.subplot(row, col, n)
                plt.imshow(data)
                plt.gray()
                plt.axis('off')

            show_size = 9
            total = 0
            fig = plt.figure(figsize=(20,20), dpi=30)
            for i in range(show_size):
                for j in range(show_size):
                    if i % 3 == 0:
                        draw_digit(x_test[(i/3)*show_size + j], show_size, show_size, total+1)
                    elif i % 3 == 1:
                        draw_digit(x_test_input[((i - 1)/3)*show_size + j], show_size, show_size, total+1)
                    else:
                        draw_digit(images[((i - 2)/3)*show_size + j], show_size, show_size, total+1)

                    total += 1
            fig.suptitle(title, fontsize=35, verticalalignment='center', y=0.92)
            plt.savefig('recon_{0}_{1}_{2}_{3}.png'.format(args.dataset, model_type, args.missingness_type, 
                                                           args.missingness_complexity),
                        bbox_inches='tight')
            plt.close()
        elif args.save_imgs and args.dataset in ('cifar10', 'svhn'):
            recons = vae.predict(image_plot_input_fn)
            images = [res['x'] for res in recons]
            
            def draw_digit(data, row, col, n):
                size = 32
                plt.subplot(row, col, n)
                plt.imshow(data)
                plt.axis('off')

            show_size = 9
            total = 0
            fig = plt.figure(figsize=(20,20), dpi=30)
            for i in range(show_size):
                for j in range(show_size):
                    if i % 3 == 0:
                        draw_digit(x_test[(i/3)*show_size + j], show_size, show_size, total+1)
                    elif i % 3 == 1:
                        draw_digit(x_test_input[((i - 1)/3)*show_size + j], show_size, show_size, total+1)
                    else:
                        draw_digit(images[((i - 2)/3)*show_size + j], show_size, show_size, total+1)

                    total += 1
            fig.suptitle(title, fontsize=35, verticalalignment='center', y=0.92)
            plt.savefig('recon_{0}_{1}_{2}_{3}.png'.format(args.dataset, model_type, args.missingness_type,
                                                           args.missingness_complexity),
                        bbox_inches='tight')
            plt.close()

        if best_valid_eval is None or eval_result['log_prob_x'] > best_valid_eval:
            best_checkpoint = vae.latest_checkpoint()
            best_valid_eval = eval_result['log_prob_x']
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= max_epochs_without_improvement:
                break

    ### missing data imputation evaluation
    
    eval_result = vae.evaluate(
        input_fn=test_input_fn,
        checkpoint_path=best_checkpoint)
    logging.info('Test set evaluation: {0}'.format(eval_result))

    if args.save_imgs and 'mnist' in args.dataset:
        recons = vae.predict(image_plot_input_fn, checkpoint_path=best_checkpoint)
        images = [res['x'] for res in recons]

        def draw_digit(data, row, col, n):
            size = 28
            plt.subplot(row, col, n)
            plt.imshow(data)
            plt.gray()
            plt.axis('off')

        show_size = 9
        total = 0
        fig = plt.figure(figsize=(20, 20), dpi=30)
        for i in range(show_size):
            for j in range(show_size):
                if i % 3 == 0:
                    draw_digit(x_test[(i/3)*show_size + j], show_size, show_size, total+1)
                elif i % 3 == 1:
                    draw_digit(x_test_input[((i - 1)/3)*show_size + j], show_size, show_size, total+1)
                else:
                    draw_digit(images[((i - 2)/3)*show_size + j], show_size, show_size, total+1)

                total += 1
        fig.suptitle(title, fontsize=35, verticalalignment='center', y=0.92)
        plt.savefig('recon_{0}_{1}_{2}_{3}.png'.format(args.dataset, model_type, args.missingness_type,
                                                       args.missingness_complexity),
                    bbox_inches='tight')
        plt.close()
    elif args.save_imgs and args.dataset in ('cifar10', 'svhn'):
        recons = vae.predict(image_plot_input_fn, checkpoint_path=best_checkpoint)
        images = [res['x'] for res in recons]
        
        def draw_digit(data, row, col, n):
            size = 32
            plt.subplot(row, col, n)
            plt.imshow(data)
            plt.axis('off')

        show_size = 9
        total = 0
        fig = plt.figure(figsize=(20, 20), dpi=30)
        for i in range(show_size):
            for j in range(show_size):
                if i % 3 == 0:
                    draw_digit(x_test[(i/3)*show_size + j], show_size, show_size, total+1)
                elif i % 3 == 1:
                    draw_digit(x_test_input[((i - 1)/3)*show_size + j], show_size, show_size, total+1)
                else:
                    draw_digit(images[((i - 2)/3)*show_size + j], show_size, show_size, total+1)

                total += 1
        fig.suptitle(title, fontsize=35, verticalalignment='center', y=0.92)
        plt.savefig('recon_{0}_{1}_{2}_{3}.png'.format(args.dataset, model_type, args.missingness_type,
                                                       args.missingness_complexity),
                    bbox_inches='tight')
        plt.close()

    shutil.rmtree(model_dir, ignore_errors=True)
