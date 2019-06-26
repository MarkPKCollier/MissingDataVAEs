import argparse
import logging

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
parser.add_argument('--dataset', type=str, default='mnist', help='mnist | cifar10')
parser.add_argument('--marginal_ll_mc_samples', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=256)

args = parser.parse_args()

# see: https://arxiv.org/pdf/1202.2745.pdf for conv architectures
# Multi-column Deep Neural Networks for Image Classification
if args.dataset == 'mnist':
    data = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = data.load_data()
    y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)

    x_train = x_train/255.0
    x_test = x_test/255.0
    x_train[x_train >= 0.5] = 1.0
    x_train[x_train < 0.5] = 0.0
    x_test[x_test >= 0.5] = 1.0
    x_test[x_test < 0.5] = 0.0

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                          train_size=50000,
                                                          test_size=10000)

    z_dim = 250
    input_units = x_train.shape[1] * x_train.shape[2]
    encoder_layers = [(20, 3, 1), (40, 5, 1), 250]
    encoder_shapes = [(28, 28, 1), (14, 14, 20), (7, 7, 40)]
    encoder_shapes_pre_pool = [(28, 28, 1), (28, 28, 20), (14, 14, 40)]
    p_x_layers = [7*7*20,
                  ([-1, 7, 7, 20], 40, 5, 2, tf.nn.relu),
                  (None, 20, 3, 2, tf.nn.relu),
                  (None, 1, 5, 1, None)]
elif args.dataset == 'cifar10':
    data = tf.keras.datasets.cifar10

    (x_train, y_train), (x_test, y_test) = data.load_data()
    y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)

    x_train = x_train/255.0
    x_test = x_test/255.0

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                          train_size=40000,
                                                          test_size=10000)

    z_dim = 250
    input_units = x_train.shape[1] * x_train.shape[2]
    encoder_layers = [(300, 3, 1), (300, 2, 1), (300, 3, 1), (300, 2, 1), 300]
    encoder_shapes = [(32, 32, 3), (16, 16, 300), (8, 8, 300), (4, 4, 300), (2, 2, 300)]
    encoder_shapes_pre_pool = [(32, 32, 3), (32, 32, 300), (16, 16, 300),
                               (8, 8, 300), (4, 4, 300)]
    p_x_layers = [2*2*300,
                  ([-1, 2, 2, 300], 300, 2, 2, tf.nn.relu),
                  (None, 300, 3, 2, tf.nn.relu),
                  (None, 300, 2, 2, tf.nn.relu),
                  (None, 300, 3, 2, tf.nn.relu),
                  (None, 6, 5, 1, None)]

max_epochs = 100
max_epoch_without_improvement = 10
img_dim = x_train.shape[1]
missingness_size = 'block' # pixel
missingness_block_size = 7
num_missing_blocks = 8

def encoder(inputs, model_type=None):
    net = inputs

    for i, layer in enumerate(encoder_layers):
        if isinstance(layer, tuple):
            filters, kernel_size, strides = layer

            if i == 0:
                if args.dataset == 'mnist':
                    in_channels = 1
                elif args.dataset == 'cifar10':
                    in_channels = 3
                
                if '_ind' in model_type:
                    in_channels *= 2
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

            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        else:
            net = tf.contrib.layers.flatten(net)
            net = tf.contrib.layers.fully_connected(net, layer, activation_fn=tf.nn.relu)
            
    net = tf.contrib.layers.flatten(net)

    mu = tf.contrib.layers.fully_connected(net, z_dim, activation_fn=None)
    sigma = tf.contrib.layers.fully_connected(net, z_dim,
                                              activation_fn=tf.nn.softplus,
                                              biases_initializer=tf.constant_initializer(
                                                np.log(np.e - np.ones(z_dim))))
    
    return mu, sigma

def decoder(z, recon_b=False):
    net = z
    for i, layer in enumerate(p_x_layers):
        if isinstance(layer, tuple):
            input_size, filters, kernel_size, strides, activation = layer
            if input_size is not None:
                net = tf.reshape(net, input_size)

            net = tf.layers.conv2d_transpose(net, filters, kernel_size, strides=strides, activation=activation, padding='SAME')
        else:
            net = tf.contrib.layers.fully_connected(net, layer, activation_fn=tf.nn.relu)

    if isinstance(layer, tuple):
        if args.dataset == 'mnist':
            p = tf.contrib.layers.flatten(net[:, :, :, 0])
            sigma_logits = None
        else:
            p = tf.contrib.layers.flatten(net[:, :, :, :3])
            sigma_logits = tf.contrib.layers.flatten(net[:, :, :, 3:])
    elif args.dataset == 'cifar10':
        p = net[:, :, :, :3]
        sigma_logits = net[:, :, :, 3:]

    b = None
    if recon_b:
        net = z
        for i, layer in enumerate(p_x_layers):
            if isinstance(layer, tuple):
                input_size, filters, kernel_size, strides, activation = layer
                if input_size is not None:
                    net = tf.reshape(net, input_size)

                net = tf.layers.conv2d_transpose(net, filters, kernel_size, strides=strides, activation=activation, padding='SAME')
            else:
                net = tf.contrib.layers.fully_connected(net, layer, activation_fn=tf.nn.relu)

        if isinstance(layer, tuple):
            if args.dataset == 'mnist':
                b = tf.contrib.layers.flatten(net[:, :, :, 0])
            else:
                b = tf.contrib.layers.flatten(net)
    
    return p, sigma_logits, b

def model(features, labels, mode, params):
    if args.dataset == 'mnist':
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
    elif args.dataset == 'cifar10':
        x = tf.reshape(tf.feature_column.input_layer(features, params['feature_columns'][0]),
                       [-1, img_dim, img_dim, 3])
        x_input = tf.reshape(tf.feature_column.input_layer(features, params['feature_columns'][1]),
                       [-1, img_dim, img_dim, 3])
        b = tf.reshape(tf.feature_column.input_layer(features, params['feature_columns'][2]),
                       [-1, img_dim, img_dim, 3])

        mu, sigma, = encoder(tf.concat([x_input, b], axis=3)
                             if '_ind' in params['model_type']
                             else x_input,
                             model_type=params['model_type'])

    q_z = tf.distributions.Normal(mu, sigma)
    
    p_z = tf.distributions.Normal(loc=np.zeros(z_dim, dtype=np.float32), scale=np.ones(z_dim, dtype=np.float32))
    
    kl = tf.reduce_mean(tf.reduce_sum(tf.distributions.kl_divergence(q_z, p_z), axis=1))

    recon_b = 'recon_b' in params['model_type']

    def log_probs(z_sample):
        if args.dataset == 'mnist':
            x_logits, _, b_logits = decoder(z_sample, recon_b=recon_b)
            x_logits = tf.reshape(x_logits, [-1, img_dim, img_dim])
            if b_logits is not None:
                b_logits = tf.reshape(b_logits, [-1, img_dim, img_dim])
            x_pred = tf.nn.sigmoid(x_logits)
        elif args.dataset == 'cifar10':
            x_logits, x_sigma_logits, b_logits = decoder(z_sample, recon_b=recon_b)
            x_logits = tf.reshape(x_logits, [-1, img_dim, img_dim, 3])
            if b_logits is not None:
                b_logits = tf.reshape(b_logits, [-1, img_dim, img_dim, 3])
            x_pred = tf.nn.sigmoid(x_logits)
            x_sigma = tf.nn.softplus(x_sigma_logits)

        if args.dataset == 'mnist':
            log_prob = -1 * tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(b * 
                tf.nn.sigmoid_cross_entropy_with_logits(labels=b * x, logits=b * x_logits), axis=2), axis=1))
            imputation_log_prob = -1 * tf.reduce_mean(tf.reduce_sum(tf.reduce_sum((1.0 - b) * 
                tf.nn.sigmoid_cross_entropy_with_logits(labels=(1.0 - b) * x, logits=(1.0 - b) * x_logits), axis=2), axis=1))
        elif args.dataset == 'cifar10':
            p_x = tf.distributions.Normal(tf.contrib.layers.flatten(x_pred),
                                          tf.contrib.layers.flatten(x_sigma))
            log_prob_full = p_x.log_prob(tf.contrib.layers.flatten(x))
            log_prob = tf.reduce_mean(
                tf.reduce_sum(tf.contrib.layers.flatten(b) * log_prob_full, axis=1))
            imputation_log_prob = tf.reduce_mean(
                tf.reduce_sum(tf.contrib.layers.flatten(1.0 - b) * log_prob_full, axis=1))

        if recon_b:
            log_prob_b = -1 * tf.reduce_mean(tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=b, logits=b_logits), axis=[2,1]))
        else:
            log_prob_b = 0.0

        return x_pred, log_prob, imputation_log_prob, log_prob_b

    def log_probs_marginal_estimator():
        print '#' * 50, 'log_probs_marginal_estimator'
        log_prob_l = []
        imputation_log_prob_l = []
        log_prob_b_l = []

        for _ in range(args.marginal_ll_mc_samples):
            z_sample = q_z.sample()

            with tf.variable_scope('log_probs', reuse=tf.AUTO_REUSE):
                _, log_prob, imputation_log_prob, log_prob_b = log_probs(z_sample)

            importance_w = p_z.prob(z_sample)/q_z.prob(z_sample)
            f = (1.0/float(args.marginal_ll_mc_samples)) * importance_w

            log_prob_l.append(f * log_prob)
            imputation_log_prob_l.append(f * imputation_log_prob)
            log_prob_b_l.append(f * log_prob_b)

        log_prob = tf.reduce_sum(log_prob_l)
        imputation_log_prob = tf.reduce_sum(imputation_log_prob_l)
        log_prob_b = tf.reduce_sum(log_prob_b_l)
        return log_prob, imputation_log_prob, log_prob_b
    def log_probs_mc_sample():
        print '#' * 50, 'log_probs_mc_sample'
        z_sample = q_z.sample()

        with tf.variable_scope('log_probs', reuse=tf.AUTO_REUSE):
            x_pred, log_prob, imputation_log_prob, log_prob_b = log_probs(z_sample)
        return log_prob, imputation_log_prob, log_prob_b

    pred = tf.cast(tf.feature_column.input_layer(features, params['feature_columns'][3]), tf.bool)[0, 0]
    print 'CHECK', pred
    log_prob, imputation_log_prob, log_prob_b = tf.cond(pred,
                                                        log_probs_marginal_estimator,
                                                        log_probs_mc_sample)
        
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode,
            predictions={'x': x_pred, 'z': mu},
            export_outputs={'y': tf.estimator.export.ClassificationOutput(scores=x_pred)})
        
    if recon_b:
        elbo = log_prob + log_prob_b - kl
    else:
        elbo = log_prob - kl
    
    loss = -elbo

    tf.summary.scalar('elbo', elbo)
    tf.summary.scalar('kl', kl)
    tf.summary.scalar('log_prob_x', log_prob)
    tf.summary.scalar('loss', loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=
                                          {'log_prob_x': tf.metrics.mean(log_prob),
                                           'imputation_log_prob':  tf.metrics.mean(
                                               imputation_log_prob),
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
    for model_type in ['VAE', 'VAE_mean_imp', 'VAE_ind', 'VAE_ind_mean_imp']:
        if missingness_type == 'independent' and 'recon_b' in model_type:
            continue
        if missingness_type == 'independent' and missingness_size == 'block':
            b_train = np.ones_like(x_train)
            b_valid = np.ones_like(x_valid)
            b_test = np.ones_like(x_test)
            for b in range(x_train.shape[0]):
                for _ in range(num_missing_blocks):
                    x = np.random.choice(img_dim)
                    y = np.random.choice(img_dim)
                    if args.dataset == 'mnist':
                        b_train[b,
                                max(x - missingness_block_size/2, 0):min(x + missingness_block_size/2, img_dim),
                                max(y - missingness_block_size/2, 0):min(y + missingness_block_size/2, img_dim)] = 0.0
                    elif args.dataset == 'cifar10':
                        b_train[b,
                                max(x - missingness_block_size/2, 0):min(x + missingness_block_size/2, img_dim),
                                max(y - missingness_block_size/2, 0):min(y + missingness_block_size/2, img_dim),
                                :] = 0.0
            for b in range(x_valid.shape[0]):
                for _ in range(num_missing_blocks):
                    x = np.random.choice(img_dim)
                    y = np.random.choice(img_dim)
                    if args.dataset == 'mnist':
                        b_valid[b,
                                max(x - missingness_block_size/2, 0):min(x + missingness_block_size/2, img_dim),
                                max(y - missingness_block_size/2, 0):min(y + missingness_block_size/2, img_dim)] = 0.0
                    elif args.dataset == 'cifar10':
                        b_valid[b,
                                max(x - missingness_block_size/2, 0):min(x + missingness_block_size/2, img_dim),
                                max(y - missingness_block_size/2, 0):min(y + missingness_block_size/2, img_dim),
                                :] = 0.0
            for b in range(x_test.shape[0]):
                for _ in range(num_missing_blocks):
                    x = np.random.choice(img_dim)
                    y = np.random.choice(img_dim)
                    if args.dataset == 'mnist':
                        b_test[b,
                                max(x - missingness_block_size/2, 0):min(x + missingness_block_size/2, img_dim),
                                max(y - missingness_block_size/2, 0):min(y + missingness_block_size/2, img_dim)] = 0.0
                    elif args.dataset == 'cifar10':
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
                    if args.dataset == 'mnist':
                        b_train[b,
                                max(x - missingness_block_size/2, 0):min(x + missingness_block_size/2, img_dim),
                                max(y - missingness_block_size/2, 0):min(y + missingness_block_size/2, img_dim)] = 0.0
                    elif args.dataset == 'cifar10':
                        b_train[b,
                                max(x - missingness_block_size/2, 0):min(x + missingness_block_size/2, img_dim),
                                max(y - missingness_block_size/2, 0):min(y + missingness_block_size/2, img_dim),
                                :] = 0.0
            for b in range(x_valid.shape[0]):
                for _ in range(num_missing_blocks):
                    x = np.random.choice(img_dim) + int(2 * y_valid[b] - 9)
                    y = np.random.choice(img_dim) + int(2 * y_valid[b] - 9)
                    if args.dataset == 'mnist':
                        b_valid[b,
                                max(x - missingness_block_size/2, 0):min(x + missingness_block_size/2, img_dim),
                                max(y - missingness_block_size/2, 0):min(y + missingness_block_size/2, img_dim)] = 0.0
                    elif args.dataset == 'cifar10':
                        b_valid[b,
                                max(x - missingness_block_size/2, 0):min(x + missingness_block_size/2, img_dim),
                                max(y - missingness_block_size/2, 0):min(y + missingness_block_size/2, img_dim),
                                :] = 0.0
            for b in range(x_test.shape[0]):
                for _ in range(num_missing_blocks):
                    x = np.random.choice(img_dim) + int(2 * y_test[b] - 9)
                    y = np.random.choice(img_dim) + int(2 * y_test[b] - 9)
                    if args.dataset == 'mnist':
                        b_test[b,
                               max(x - missingness_block_size/2, 0):min(x + missingness_block_size/2, img_dim),
                               max(y - missingness_block_size/2, 0):min(y + missingness_block_size/2, img_dim)] = 0.0
                    elif args.dataset == 'cifar10':
                        b_test[b,
                               max(x - missingness_block_size/2, 0):min(x + missingness_block_size/2, img_dim),
                               max(y - missingness_block_size/2, 0):min(y + missingness_block_size/2, img_dim),
                                :] = 0.0

        print 'Train fraction of pixels missing:', 1.0 - np.sum(b_train)/np.sum(np.ones_like(b_train))
        print 'Valid fraction of pixels missing:', 1.0 - np.sum(b_valid)/np.sum(np.ones_like(b_valid))
        print 'Test fraction of pixels missing:', 1.0 - np.sum(b_test)/np.sum(np.ones_like(b_test))

        if '_mean_imp' in model_type:
            x_mu = np.true_divide(np.sum(b_train * x_train, axis=0), np.sum(b_train, axis=0))
            x_train_input = b_train * x_train + (1.0 - b_train) * x_mu
            x_valid_input = b_valid * x_valid + (1.0 - b_valid) * x_mu
            x_test_input = b_test * x_test + (1.0 - b_test) * x_mu
        else:
            x_train_input = b_train * x_train
            x_valid_input = b_valid * x_valid
            x_test_input = b_test * x_test

        feature_columns = [tf.feature_column.numeric_column(key='x', shape=[img_dim, img_dim]),
                           tf.feature_column.numeric_column(key='x_input', shape=[img_dim, img_dim]),
                           tf.feature_column.numeric_column(key='b', shape=[img_dim, img_dim]),
                           tf.feature_column.numeric_column(key='estimate_log_p_x', shape=[])]

        vae = tf.estimator.Estimator(
            model_fn=model,
            model_dir='{0}_conv_{1}_{2}_{3}'.format(args.dataset, model_type,
                                                    missingness_type, args.run_id),
            params={'feature_columns': feature_columns, 'model_type': model_type},
            config=tf.estimator.RunConfig(
                save_summary_steps=1000,
                save_checkpoints_steps=20000,
                keep_checkpoint_max=200,
                log_step_count_steps=1000))
        
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            {'x': x_train, 'x_input': x_train_input, 'b': b_train,
             'estimate_log_p_x': np.zeros(x_train.shape[0])},
            shuffle=True, batch_size=args.batch_size)

        valid_input_fn = tf.estimator.inputs.numpy_input_fn(
            {'x': x_valid, 'x_input': x_valid_input, 'b': b_valid,
             'estimate_log_p_x': np.ones(x_valid.shape[0])},
            shuffle=False, batch_size=args.batch_size)
        
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            {'x': x_test, 'x_input': x_test_input, 'b': b_test,
            'estimate_log_p_x': np.ones(x_test.shape[0])},
            shuffle=False, batch_size=args.batch_size)

        best_valid_elbo = None
        best_checkpoint = None
        best_estimator = None
        epochs_since_improvement = 0
        for _ in range(max_epochs):
            vae.train(steps=None, input_fn=train_input_fn)

            # eval_result = vae.evaluate(input_fn=train_input_fn)
            # logging.info('End of epoch evaluation (train set): ' + str(eval_result))

            eval_result = vae.evaluate(input_fn=valid_input_fn)
            logging.info('End of epoch evaluation (valid set): ' + str(eval_result))

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

