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

args = parser.parse_args()

if args.dataset == 'mnist':
    data = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = data.load_data()
    y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)

    x_train = x_train/255.0
    x_test = x_test/255.0
    # x_train[x_train >= 0.5] = 1.0
    # x_train[x_train < 0.5] = 0.0
    # x_test[x_test >= 0.5] = 1.0
    # x_test[x_test < 0.5] = 0.0

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                          train_size=50000,
                                                          test_size=10000)
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
layer_units = [1500, 1000, 500]
# see: https://arxiv.org/pdf/1003.0358.pdf 
# Deep Big Simple Neural Nets Excel on Handwritten Digit Recognition

max_epochs = 100
max_epoch_without_improvement = 10
img_dim = x_train.shape[1]
missingness_size = 'block' # pixel
missingness_block_size = 7
num_missing_blocks = 8

def encoder(inputs, model_type=None):
    net = inputs
    
    for i, units in enumerate(layer_units):
        if (i == 0) and ('_ind' in model_type):
            d = 2 * input_units
        elif i == 0:
            d = input_units
        else:
            d = layer_units[i-1]

        w = tf.get_variable('layer_{0}_w'.format(i),
                            shape=(d, units),
                            initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('layer_{0}_bias'.format(i),
                               shape=units,
                               initializer=tf.initializers.zeros())

        net = tf.nn.relu(tf.matmul(net, w) + bias)
    
    mu = tf.contrib.layers.fully_connected(net, z_dim, activation_fn=None)
    sigma = tf.contrib.layers.fully_connected(net, z_dim,
                                              activation_fn=tf.nn.softplus,
                                              biases_initializer=tf.constant_initializer(
                                                np.log(np.e - np.ones(z_dim))))
    
    return mu, sigma

def decoder(z, recon_b=False):
    net = z
    for units in reversed(layer_units):
        net = tf.contrib.layers.fully_connected(net, units, activation_fn=tf.nn.relu)
    
    p = tf.contrib.layers.fully_connected(net, input_units, activation_fn=None)
    if recon_b:
        net = z
        for units in reversed(layer_units):
            net = tf.contrib.layers.fully_connected(net, units, activation_fn=tf.nn.relu)
        
        b = tf.contrib.layers.fully_connected(net, input_units, activation_fn=None)
        return p, b
    
    return p, None

def model(features, labels, mode, params):
    x = tf.feature_column.input_layer(features, params['feature_columns'][0])
    x_input = tf.feature_column.input_layer(features, params['feature_columns'][1])
    b = tf.feature_column.input_layer(features, params['feature_columns'][2])
    recon_b = 'recon_b' in params['model_type']
    
    mu, sigma = encoder(tf.concat([x_input, b], 1)
                        if '_ind' in params['model_type']
                        else x_input,
                        model_type=params['model_type'])
    
    q_z = tf.distributions.Normal(mu, sigma)
    
    p_z = tf.distributions.Normal(loc=np.zeros(z_dim, dtype=np.float32),
                                  scale=np.ones(z_dim, dtype=np.float32))
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        decoder_inputs = z_sample = q_z.sample()
    else:
        decoder_inputs = z_sample = mu
    
    if 'dec_cond_b' in params['model_type']:
        decoder_inputs = tf.concat([z_sample, b], axis=1)
    x_logits, b_logits = decoder(decoder_inputs, recon_b=recon_b)
    x_pred = tf.nn.sigmoid(x_logits)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode,
            predictions={'x': x_pred, 'z': mu},
            export_outputs={'y': tf.estimator.export.ClassificationOutput(scores=x_pred)})
        
    kl = tf.reduce_mean(tf.reduce_sum(tf.distributions.kl_divergence(q_z, p_z), axis=1))
    log_prob = -1 * tf.reduce_mean(tf.reduce_sum(b * 
        tf.nn.sigmoid_cross_entropy_with_logits(labels=b * x, logits=b * x_logits), axis=1))
    imputation_log_prob = -1 * tf.reduce_mean(tf.reduce_sum((1.0 - b) *
        tf.nn.sigmoid_cross_entropy_with_logits(labels=(1.0 - b) * x, logits=(1.0 - b) * x_logits), axis=1))

    if recon_b:
        log_prob_b = -1 * tf.reduce_mean(tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=b, logits=b_logits), axis=1))
        
        elbo = log_prob + log_prob_b - kl
    else:
        elbo = log_prob - kl

    loss = -elbo

    tf.summary.scalar('kl', kl)
    tf.summary.scalar('log_prob_x', log_prob)
    tf.summary.scalar('elbo', elbo)
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
                    b_train[b,
                            max(x - missingness_block_size/2, 0):min(x + missingness_block_size/2, img_dim),
                            max(y - missingness_block_size/2, 0):min(y + missingness_block_size/2, img_dim)] = 0.0
            for b in range(x_valid.shape[0]):
                for _ in range(num_missing_blocks):
                    x = np.random.choice(img_dim)
                    y = np.random.choice(img_dim)
                    b_valid[b,
                            max(x - missingness_block_size/2, 0):min(x + missingness_block_size/2, img_dim),
                            max(y - missingness_block_size/2, 0):min(y + missingness_block_size/2, img_dim)] = 0.0
            for b in range(x_test.shape[0]):
                for _ in range(num_missing_blocks):
                    x = np.random.choice(img_dim)
                    y = np.random.choice(img_dim)
                    b_test[b,
                           max(x - missingness_block_size/2, 0):min(x + missingness_block_size/2, img_dim),
                           max(y - missingness_block_size/2, 0):min(y + missingness_block_size/2, img_dim)] = 0.0
        elif missingness_type == 'dependent' and missingness_size == 'block':
            b_train = np.ones_like(x_train)
            b_valid = np.ones_like(x_valid)
            b_test = np.ones_like(x_test)
            for b in range(x_train.shape[0]):
                for _ in range(num_missing_blocks):
                    x = np.random.choice(img_dim) + int(2 * y_train[b] - 9)
                    y = np.random.choice(img_dim) + int(2 * y_train[b] - 9)
                    b_train[b,
                            max(x - missingness_block_size/2, 0):min(x + missingness_block_size/2, img_dim),
                            max(y - missingness_block_size/2, 0):min(y + missingness_block_size/2, img_dim)] = 0.0
            for b in range(x_valid.shape[0]):
                for _ in range(num_missing_blocks):
                    x = np.random.choice(img_dim) + int(2 * y_valid[b] - 9)
                    y = np.random.choice(img_dim) + int(2 * y_valid[b] - 9)
                    b_valid[b,
                            max(x - missingness_block_size/2, 0):min(x + missingness_block_size/2, img_dim),
                            max(y - missingness_block_size/2, 0):min(y + missingness_block_size/2, img_dim)] = 0.0
            for b in range(x_test.shape[0]):
                for _ in range(num_missing_blocks):
                    x = np.random.choice(img_dim) + int(2 * y_test[b] - 9)
                    y = np.random.choice(img_dim) + int(2 * y_test[b] - 9)
                    b_test[b,
                           max(x - missingness_block_size/2, 0):min(x + missingness_block_size/2, img_dim),
                           max(y - missingness_block_size/2, 0):min(y + missingness_block_size/2, img_dim)] = 0.0
        elif missingness_type == 'independent':
            b_train = np.random.uniform(size=x_train.shape) > p
            b_valid = np.random.uniform(size=x_valid.shape) > p
            b_test = np.random.uniform(size=x_test.shape) > p
        elif missingness_type == 'dependent':
            b_train = np.random.uniform(size=x_train.shape) > np.expand_dims(
                np.expand_dims(p * (1 + y_train)/10.0, axis=1), axis=2)
            b_valid = np.random.uniform(size=x_valid.shape) > np.expand_dims(
                np.expand_dims(p * (1 + y_valid)/10.0, axis=1), axis=2)
            b_test = np.random.uniform(size=x_test.shape) > np.expand_dims(
                np.expand_dims(p * (1 + y_test)/10.0, axis=1), axis=2)

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
                           tf.feature_column.numeric_column(key='b', shape=[img_dim, img_dim])]

        vae = tf.estimator.Estimator(
            model_fn=model,
            model_dir='{0}_{1}_{2}_{3}'.format(args.dataset, model_type, missingness_type, args.run_id),
            params={'feature_columns': feature_columns, 'model_type': model_type},
            config=tf.estimator.RunConfig(
                save_summary_steps=1000,
                save_checkpoints_steps=6000,
                keep_checkpoint_max=1000,
                log_step_count_steps=1000))
        
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            {'x': x_train, 'x_input': x_train_input, 'b': b_train},
            shuffle=True)

        valid_input_fn = tf.estimator.inputs.numpy_input_fn(
            {'x': x_valid, 'x_input': x_valid_input, 'b': b_valid},
            shuffle=False)
        
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            {'x': x_test, 'x_input': x_test_input, 'b': b_test},
            shuffle=False)

        best_valid_elbo = None
        best_checkpoint = None
        best_estimator = None
        epochs_since_improvement = 0
        for _ in range(max_epochs):
            vae.train(steps=None, input_fn=train_input_fn)

            eval_result = vae.evaluate(input_fn=train_input_fn)
            logging.info('End of epoch evaluation (train set): ' + str(eval_result))

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
