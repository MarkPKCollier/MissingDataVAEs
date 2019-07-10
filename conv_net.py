import argparse
import logging

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
parser.add_argument('--dataset', type=str, default='mnist', help='binary_minst | mnist | cifar10 | svhn')
parser.add_argument('--marginal_ll_mc_samples', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--save_imgs', type=str2bool, default=False)
parser.add_argument('--predict_variance', type=str2bool, default=False)
parser.add_argument('--learn_variance', type=str2bool, default=False)

args = parser.parse_args()

if args.save_imgs:
    from matplotlib import pyplot as plt

# see: https://arxiv.org/pdf/1202.2745.pdf for conv architectures
# Multi-column Deep Neural Networks for Image Classification
if 'mnist' in args.dataset:
    data = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = data.load_data()
    y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)

    if args.learn_variance or args.predict_variance:
        x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)
        x_train += np.random.uniform(size=x_train.shape)
        x_test += np.random.uniform(size=x_test.shape)
        x_train = x_train/256.0
        x_test = x_test/256.0
    else:
        x_train = x_train/255.0
        x_test = x_test/255.0

    if args.dataset == 'binary_mnist':
        x_train[x_train >= 0.5] = 1.0
        x_train[x_train < 0.5] = 0.0
        x_test[x_test >= 0.5] = 1.0
        x_test[x_test < 0.5] = 0.0

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                          train_size=50000,
                                                          test_size=10000)

    # z_dim = 250
    # input_units = x_train.shape[1] * x_train.shape[2]
    # encoder_layers = [(20, 3, 1), (40, 5, 1), 250]
    # encoder_shapes = [(28, 28, 1), (14, 14, 20), (7, 7, 40)]
    # encoder_shapes_pre_pool = [(28, 28, 1), (28, 28, 20), (14, 14, 40)]
    # p_x_layers = [7*7*20,
    #               ([-1, 7, 7, 20], 40, 5, 2, tf.nn.relu),
    #               (None, 20, 3, 2, tf.nn.relu),
    #               (None, 1, 5, 1, None)]

    z_dim = 50
    input_units = x_train.shape[1] * x_train.shape[2]
    encoder_layers = [(20, 3, 2), (40, 5, 2), 250]
    encoder_shapes = [(28, 28, 1), (14, 14, 20), (7, 7, 40)]
    encoder_shapes_pre_pool = encoder_shapes
    p_x_layers = [7*7*20,
                  ([-1, 7, 7, 20], 40, 5, 2, tf.nn.relu),
                  (None, 20, 3, 2, tf.nn.relu)]
    p_x_b_z_layers = [(None, 10, 3, 1, tf.nn.relu),
                      (None, 5, 3, 1, tf.nn.relu),
                      (None, 2, 3, 1, None) if args.predict_variance else (None, 1, 3, 1, None)]
elif 'svhn' in args.dataset:
    x_train, x_test = chainer.datasets.get_svhn(withlabel=False, scale=255.0)

    x_train = np.concatenate((np.expand_dims(x_train[:, 0, :, :], -1),
                              np.expand_dims(x_train[:, 1, :, :], -1),
                              np.expand_dims(x_train[:, 2, :, :], -1)), axis=3)
    x_test = np.concatenate((np.expand_dims(x_test[:, 0, :, :], -1),
                             np.expand_dims(x_test[:, 1, :, :], -1),
                             np.expand_dims(x_test[:, 2, :, :], -1)), axis=3)

    # if args.learn_variance or args.predict_variance:
    #     x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)
    #     x_train += np.random.uniform(size=x_train.shape)
    #     x_test += np.random.uniform(size=x_test.shape)
    #     x_train = x_train/256.0
    #     x_test = x_test/256.0
    # else:
    #     x_train = x_train/255.0
    #     x_test = x_test/255.0
    x_train = x_train/255.0
    x_test = x_test/255.0

    x_train, x_valid = train_test_split(x_train, train_size=63257, test_size=10000)

    z_dim = 100
    input_units = x_train.shape[1] * x_train.shape[2]
    encoder_layers = [(20, 3, 2), (40, 5, 2), 250]
    encoder_shapes = [(32, 32, 3), (16, 16, 20), (8, 8, 40)]
    encoder_shapes_pre_pool = encoder_shapes
    p_x_layers = [8*8*20,
                  ([-1, 8, 8, 20], 40, 5, 2, tf.nn.relu),
                  (None, 20, 3, 2, tf.nn.relu)]
    p_x_b_z_layers = [(None, 10, 3, 1, tf.nn.relu),
                      (None, 5, 3, 1, tf.nn.relu),
                      (None, 6, 3, 1, None) if args.predict_variance else (None, 3, 3, 1, None)]
elif args.dataset == 'cifar10':
    data = tf.keras.datasets.cifar10

    (x_train, y_train), (x_test, y_test) = data.load_data()
    y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)

    if args.learn_variance or args.predict_variance:
        x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)
        x_train += np.random.uniform(size=x_train.shape)
        x_test += np.random.uniform(size=x_test.shape)
        x_train = x_train/256.0
        x_test = x_test/256.0
    else:
        x_train = x_train/255.0
        x_test = x_test/255.0

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                          train_size=40000,
                                                          test_size=10000)

    z_dim = 150
    input_units = x_train.shape[1] * x_train.shape[2]
    encoder_layers = [(100, 3, 1), (100, 2, 1), (100, 3, 1), (100, 2, 1), 300]
    encoder_shapes = [(32, 32, 3), (16, 16, 100), (8, 8, 100), (4, 4, 100), (2, 2, 100)]
    encoder_shapes_pre_pool = [(32, 32, 3), (32, 32, 100), (16, 16, 100),
                               (8, 8, 100), (4, 4, 100)]
    p_x_layers = [2*2*100,
                  ([-1, 2, 2, 100], 100, 2, 2, tf.nn.relu),
                  (None, 100, 3, 2, tf.nn.relu),
                  (None, 100, 2, 2, tf.nn.relu),
                  (None, 100, 3, 2, tf.nn.relu)]
    p_x_b_z_layers = [(None, 15, 3, 1, tf.nn.relu),
                      (None, 5, 3, 1, tf.nn.relu),
                      (None, 6, 3, 1, None) if args.predict_variance else (None, 3, 3, 1, None)]
    # encoder_layers = [(300, 3, 1), (300, 2, 1), (300, 3, 1), (300, 2, 1), 300]
    # encoder_shapes = [(32, 32, 3), (16, 16, 300), (8, 8, 300), (4, 4, 300), (2, 2, 300)]
    # encoder_shapes_pre_pool = [(32, 32, 3), (32, 32, 300), (16, 16, 300),
    #                            (8, 8, 300), (4, 4, 300)]
    # p_x_layers = [2*2*300,
    #               ([-1, 2, 2, 300], 300, 2, 2, tf.nn.relu),
    #               (None, 300, 3, 2, tf.nn.relu),
    #               (None, 300, 2, 2, tf.nn.relu),
    #               (None, 300, 3, 2, tf.nn.relu),
    #               (None, 6, 5, 1, None)]

max_epochs = 100
max_epoch_without_improvement = 10
img_dim = x_train.shape[1]
missingness_size = 'block' # pixel
missingness_block_size = 7
if 'mnist' in args.dataset:
    num_missing_blocks = 8
elif args.dataset == 'svhn':
    num_missing_blocks = 8
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

        net = tf.layers.conv2d_transpose(net, filters, kernel_size,
                                         strides=strides, activation=activation,
                                         padding='SAME')

        bias = tf.get_variable('p_x_b_z_layer_{0}_bias'.format(i),
                               shape=p_x_b_z_layers[i][1],
                               initializer=tf.initializers.zeros())

        net = tf.nn.bias_add(net, bias)

        if args.predict_variance:
            if 'mnist' in args.dataset:
                p = net[:, :, :, 0]
                log_var = net[:, :, :, 1]
            else:
                p = net[:, :, :, :3]
                log_var = net[:, :, :, 3:]
        else:
            p = net
            log_var = None

    return p, log_var

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

        mu, sigma, = encoder(tf.concat([x_input, tf.expand_dims(b[:, :, :, 0], axis=-1)], axis=3)
                             if '_ind' in params['model_type']
                             else x_input,
                             model_type=params['model_type'])

    q_z = tf.distributions.Normal(mu, sigma)
    
    p_z = tf.distributions.Normal(loc=np.zeros(z_dim, dtype=np.float32), scale=np.ones(z_dim, dtype=np.float32))
    
    kl = tf.reduce_mean(tf.reduce_sum(tf.distributions.kl_divergence(q_z, p_z), axis=1))

    def log_probs(z_sample):
        if args.dataset == 'binary_mnist' or args.dataset == 'mnist':
            x_logits, x_log_var = decoder(z_sample,
                                          b=b if '_decoder_b' in params['model_type'] else None)
            x_logits = tf.reshape(x_logits, [-1, img_dim, img_dim])
            x_pred = tf.nn.sigmoid(x_logits)
            if args.dataset == 'mnist' and args.predict_variance:
                x_log_var = tf.reshape(x_log_var, [-1, img_dim, img_dim])
                x_var = tf.exp(x_log_var) + 1e-2
        elif args.dataset in ('cifar10', 'svhn'):
            x_logits, x_log_var = decoder(z_sample,
                                          b=b if '_decoder_b' in params['model_type'] else None)
            x_logits = tf.reshape(x_logits, [-1, img_dim, img_dim, 3])
            # x_pred = tf.nn.sigmoid(x_logits)
            x_pred = x_logits
            if args.predict_variance:
                x_var = tf.exp(x_log_var) + 1e-2
        
        # if args.dataset == 'binary_mnist' or args.dataset == 'mnist':
        if args.dataset == 'binary_mnist':
            # valid for real valued MNIST: http://ruishu.io/2018/03/19/bernoulli-vae/
            # log_prob_full = tf.distributions.Bernoulli(logits=x_logits).log_prob(x)
            log_prob_full = -1 * tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_logits)
            log_prob = tf.reduce_mean(tf.reduce_sum(b * log_prob_full, axis=[2,1]))
            imputation_log_prob = tf.reduce_mean(tf.reduce_sum((1.0 - b) * log_prob_full, axis=[2,1]))
        elif args.predict_variance and (args.dataset in ('cifar10', 'mnist', 'svhn')):
            x_pred = x_logits
            # p_x = tf.distributions.Normal(
            #     loc=tf.contrib.layers.flatten(x_pred),
            #     scale=tf.contrib.layers.flatten(x_sigma))
            # log_prob_full = p_x.log_prob(tf.contrib.layers.flatten(x))
            # log_prob = tf.reduce_mean(
            #     tf.reduce_sum(tf.contrib.layers.flatten(b) * log_prob_full, axis=1))
            # imputation_log_prob = tf.reduce_mean(
            #     tf.reduce_sum(tf.contrib.layers.flatten(1.0 - b) * log_prob_full, axis=1))
            
            # log_prob_full = -0.5 * tf.cast(tf.log(2 * np.pi * x_var), tf.float32) -(x - x_pred)**2/(2 * x_var)
            # log_prob = tf.reduce_mean(
            #     tf.reduce_sum(b * log_prob_full,
            #                   axis=[2,1] if args.dataset == 'mnist'
            #                   else [3,2,1]))
            # imputation_log_prob = tf.reduce_mean(
            #     tf.reduce_sum((1.0 - b) * log_prob_full,
            #                   axis=[2,1] if args.dataset == 'mnist'
            #                   else [3,2,1]))
            
            centered_x = x - x_pred
            # centered_x = tf.Print(centered_x, [tf.reduce_mean(x_pred), tf.reduce_min(x_pred), tf.reduce_max(x_pred)], message='x_pred')
            # x_var = tf.Print(x_var, [tf.reduce_mean(x_var), tf.reduce_min(x_var), tf.reduce_max(x_var)], message='x_var')
            upper_in = (centered_x + (1.0/255.0))/x_var
            upper_cdf = tf.nn.sigmoid(upper_in)
            lower_in = (centered_x - (1.0/255.0))/x_var
            lower_cdf = tf.nn.sigmoid(lower_in)
            cdf_delta = upper_cdf - lower_cdf
            
            mid_in = centered_x/x_var
            log_pdf_mid = mid_in - x_log_var - 2.0 * tf.nn.softplus(mid_in)
            log_cdf_plus = upper_in - tf.nn.softplus(upper_in)
            log_one_minus_cdf_min = -tf.nn.softplus(lower_in)
            
            log_prob_full = tf.where(x < -0.999,
                                     log_cdf_plus,
                                     tf.where(x > 0.999,
                                              log_one_minus_cdf_min,
                                              tf.where(cdf_delta > 1e-5,
                                                       tf.log(tf.maximum(cdf_delta, 1e-12)),
                                                       log_pdf_mid - np.log(127.5))))

            log_prob = tf.reduce_mean(
                tf.reduce_sum(b * log_prob_full,
                              axis=[2,1] if args.dataset == 'mnist'
                              else [3,2,1]))
            imputation_log_prob = tf.reduce_mean(
                tf.reduce_sum((1.0 - b) * log_prob_full,
                              axis=[2,1] if args.dataset == 'mnist'
                              else [3,2,1]))

        elif args.learn_variance and (args.dataset in ('cifar10', 'mnist', 'svhn')):
            x_pred = x_logits
            log_var = tf.get_variable('log_var',
                                      [28, 28] if args.dataset == 'mnist' else [32, 32, 3],
                                      initializer=tf.zeros_initializer(), trainable=True)
            # log_var = tf.get_variable('log_var', [1],
            #                           initializer=tf.zeros_initializer(), trainable=True)
            var = tf.exp(log_var) + 1e-3
            # log_prob_full = -0.5 * tf.cast(tf.log(2 * np.pi), tf.float32) -0.5 * log_var -0.5 * tf.square(x - x_pred)/var
            log_prob_full = -0.5 * tf.cast(tf.log(2 * np.pi * var), tf.float32) -0.5 * tf.divide(tf.square(x - x_pred), var)
            log_prob = tf.reduce_mean(
                tf.reduce_sum(b * log_prob_full,
                              axis=[2,1] if args.dataset == 'mnist'
                              else [3,2,1]))
            imputation_log_prob = tf.reduce_mean(
                tf.reduce_sum((1.0 - b) * log_prob_full,
                              axis=[2,1] if args.dataset == 'mnist'
                              else [3,2,1]))
        elif args.dataset in ('cifar10', 'mnist', 'svhn'):
            # log_prob_full = -0.5 * tf.cast(tf.log(2 * np.pi * params['sigma_hat']), tf.float32) - tf.square(x - x_pred)/(2 * params['sigma_hat'])
            # log_prob_full = -tf.square(x - x_pred)/(2 * params['sigma_hat'])
            # log_prob_full = -tf.square(x - x_pred)/2.0

            # log_prob_full = -1 * tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_logits)
            # x_pred = tf.nn.sigmoid(x_logits)

            log_prob = tf.reduce_mean(
                tf.reduce_sum(b * log_prob_full,
                              axis=[2,1] if args.dataset == 'mnist' else [3,2,1]))
            imputation_log_prob = tf.reduce_mean(
                tf.reduce_sum((1.0 - b) * log_prob_full,
                              axis=[2,1] if args.dataset == 'mnist' else [3,2,1]))

        return x_pred, log_prob, imputation_log_prob

    def log_probs_multiple_mc_samples():
        # print '#' * 50, 'log_probs_multiple_mc_samples'
        log_prob_l = []
        imputation_log_prob_l = []

        f = (1.0/float(args.marginal_ll_mc_samples))
        for _ in range(args.marginal_ll_mc_samples):
            z_sample = q_z.sample()

            with tf.variable_scope('log_probs', reuse=tf.AUTO_REUSE):
                _, log_prob, imputation_log_prob = log_probs(z_sample)

            log_prob_l.append(f * log_prob)
            imputation_log_prob_l.append(f * imputation_log_prob)

        log_prob = tf.reduce_sum(log_prob_l)
        imputation_log_prob = tf.reduce_sum(imputation_log_prob_l)
        return log_prob, imputation_log_prob
    def log_probs_marginal_estimator():
        pass
    def log_probs_single_mc_sample():
        # print '#' * 50, 'log_probs_single_mc_sample'
        z_sample = q_z.sample()

        with tf.variable_scope('log_probs', reuse=tf.AUTO_REUSE):
            x_pred, log_prob, imputation_log_prob = log_probs(z_sample)
        return x_pred, log_prob, imputation_log_prob
    def log_probs_deterministic():
        # print '#' * 50, 'log_probs_deterministic'
        z_sample = mu

        with tf.variable_scope('log_probs', reuse=tf.AUTO_REUSE):
            x_pred, log_prob, imputation_log_prob = log_probs(z_sample)
        return x_pred, log_prob, imputation_log_prob

    # pred = tf.cast(tf.feature_column.input_layer(features, params['feature_columns'][3]), tf.bool)[0, 0]
    # x_pred, log_prob, imputation_log_prob = tf.cond(pred,
    #                                                     log_probs_deterministic,
    #                                                     log_probs_single_mc_sample)
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

# for missingness_type, p in [('dependent', 0.5), ('independent', 0.5)]:
for missingness_type, p in [('independent', 0.5)]:
    if missingness_type == 'independent' and missingness_size == 'block':
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

    elif missingness_type == 'dependent' and missingness_size == 'block':
        b_train = np.ones_like(x_train)
        b_valid = np.ones_like(x_valid)
        b_test = np.ones_like(x_test)
        for b in range(x_train.shape[0]):
            for _ in range(num_missing_blocks):
                x = np.random.choice(img_dim) + int(2 * y_train[b] - 9)
                y = np.random.choice(img_dim) + int(2 * y_train[b] - 9)
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
                x = np.random.choice(img_dim) + int(2 * y_valid[b] - 9)
                y = np.random.choice(img_dim) + int(2 * y_valid[b] - 9)
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
                x = np.random.choice(img_dim) + int(2 * y_test[b] - 9)
                y = np.random.choice(img_dim) + int(2 * y_test[b] - 9)
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
    
    # for model_type, title in [('VAE', 'Zero imputation'),
    #                    ('VAE_mean_imp', 'Mean imputation'),
    #                    ('VAE_ind', 'Zero imputation w/ indicator variables'),
    #                    ('VAE_ind_mean_imp', 'Mean imputation w/ encoder indicator variables'),
    #                    ('VAE_ind_decoder_b', 'Zero imputation w/ encoder + decoder indicator variables'),
    #                    ('VAE_ind_mean_imp_decoder_b', 'Mean imputation w/ encoder + decoder indicator variables')]:
    for model_type, title in [('VAE_ind_decoder_b', 'Zero imputation w/ encoder + decoder indicator variables')]:
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
            plt.savefig('{0}_{1}.png'.format(args.dataset), bbox_inches='tight')
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
            plt.savefig('{0}_{1}.png'.format(args.dataset, model_type), bbox_inches='tight')
            plt.close()

        feature_columns = [tf.feature_column.numeric_column(key='x', shape=[img_dim, img_dim]),
                           tf.feature_column.numeric_column(key='x_input', shape=[img_dim, img_dim]),
                           tf.feature_column.numeric_column(key='b', shape=[img_dim, img_dim]),
                           tf.feature_column.numeric_column(key='multiple_mc_samples', shape=[])]

        vae = tf.estimator.Estimator(
            model_fn=model,
            model_dir='{0}_conv_{1}_{2}_{3}'.format(args.dataset, model_type,
                                                    missingness_type, args.run_id),
            params={'feature_columns': feature_columns,
                    'model_type': model_type,
                    'sigma_hat': sigma_hat.astype(np.float32)},
            config=tf.estimator.RunConfig(
                save_summary_steps=1000,
                save_checkpoints_steps=20000,
                keep_checkpoint_max=200,
                log_step_count_steps=1000))
        
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

        best_valid_elbo = None
        best_checkpoint = None
        best_estimator = None
        epochs_since_improvement = 0
        for i in range(max_epochs):
            vae.train(steps=None, input_fn=train_input_fn)

            # eval_result = vae.evaluate(input_fn=train_input_fn)
            # logging.info('End of epoch evaluation (train set): ' + str(eval_result))

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
                plt.savefig('recon_{0}_{1}.png'.format(args.dataset, model_type), bbox_inches='tight')
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
                plt.savefig('recon_{0}_{1}.png'.format(args.dataset, model_type), bbox_inches='tight')
                plt.close()

            if best_valid_elbo is None or eval_result['elbo'] > best_valid_elbo:
                best_checkpoint = vae.latest_checkpoint()
                best_valid_elbo = eval_result['elbo']
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= max_epoch_without_improvement:
                    break

        ### missing data imputation evaluation
        
        eval_result = vae.evaluate(
            input_fn=valid_input_fn,
            checkpoint_path=best_checkpoint)
        logging.info('Valid set evaluation: {0}'.format(eval_result))
        
        eval_result = vae.evaluate(
            input_fn=test_input_fn,
            checkpoint_path=best_checkpoint)
        logging.info('Test set evaluation: {0}'.format(eval_result))

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
            plt.savefig('recon_mnist_{0}.png'.format(model_type), bbox_inches='tight')
            plt.close()
