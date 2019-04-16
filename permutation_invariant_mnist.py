import logging

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.propagate = False

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=50000, test_size=10000)

z_dim = 128
input_units = x_train.shape[1] * x_train.shape[2]
layer_units = [1024, 512, 256, 256, 256]
max_epochs = 100
max_epoch_without_improvement = 15
mnist_dim = x_train.shape[1]

def encoder(inputs, b=None):
    net = inputs
    b_net = b
    for units in layer_units:
        net = tf.contrib.layers.fully_connected(net, units, activation_fn=tf.nn.relu)
        if b is not None:
            b_net = tf.contrib.layers.fully_connected(b_net, units, activation_fn=tf.nn.sigmoid)
            net = net * b_net
    
    mu = tf.contrib.layers.fully_connected(net, z_dim, activation_fn=None)
    sigma = tf.contrib.layers.fully_connected(net, z_dim, activation_fn=tf.nn.softplus)
    
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
    b = tf.feature_column.input_layer(features, params['feature_columns'][1])
    recon_b = 'recon_b' in params['model_type']
    
#     mu, sigma = encoder(tf.concat([x, b, 1.0 - b], 1) if '_ind' in params['model_type'] else x)
    mu, sigma = encoder(tf.concat([x * b, b], 1) if '_ind' in params['model_type'] else x,
                        b=b if 'self_dropout' in params['model_type'] else None)
    
    q_z = tf.distributions.Normal(mu, sigma)
    
    p_z = tf.distributions.Normal(loc=np.zeros(z_dim, dtype=np.float32), scale=np.ones(z_dim, dtype=np.float32))
    
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
        tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_logits), axis=1))

    if recon_b:
        log_prob_b = -1 * tf.reduce_mean(tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=b, logits=b_logits), axis=1))
        
        loss = -1 * log_prob - log_prob_b + kl
    else:
        loss = -1 * log_prob + kl

    tf.summary.scalar('kl', kl)
    tf.summary.scalar('log_prob', log_prob)
    tf.summary.scalar('loss', loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=
                                          {'log_prob': tf.metrics.mean(log_prob),
                                           'imputation_log_prob':  tf.metrics.mean(
                                               imputation_log_prob),
                                           'kl': tf.metrics.mean(kl)})

    optimizer = tf.train.AdamOptimizer(learning_rate=0.002)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss,
            global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

for missingness_type, p in [('dependent', 0.2), ('independent', 0.2)]:
    # for model_type in ['VAE_ind_dec_cond_b', 'VAE_ind_recon_b', 'VAE', 'VAE_ind']:
    for model_type in ['VAE_ind_self_dropout', 'VAE_ind']:
        if missingness_type == 'independent':
            b_train = np.random.uniform(size=x_train.shape) > p
#             x_train_ = b_train * x_train
            b_valid = np.random.uniform(size=x_valid.shape) > p
#             x_valid_ = b_valid * x_valid
            b_test = np.random.uniform(size=x_test.shape) > p
#             x_test_ = b_test * x_test
        elif missingness_type == 'dependent':
            b_train = np.random.uniform(size=x_train.shape) > np.expand_dims(
                np.expand_dims(p * (1 + y_train)/10.0, axis=1), axis=2)
#             x_train_ = b_train * x_train
            b_valid = np.random.uniform(size=x_valid.shape) > np.expand_dims(
                np.expand_dims(p * (1 + y_valid)/10.0, axis=1), axis=2)
#             x_valid_ = b_valid * x_valid
            b_test = np.random.uniform(size=x_test.shape) > np.expand_dims(
                np.expand_dims(p * (1 + y_test)/10.0, axis=1), axis=2)
#             x_test_ = b_test * x_test

        feature_columns = [tf.feature_column.numeric_column(key='x', shape=[mnist_dim, mnist_dim]),
                           tf.feature_column.numeric_column(key='b', shape=[mnist_dim, mnist_dim])]

        vae = tf.estimator.Estimator(
            model_fn=model,
            model_dir='mnist_{0}_{1}'.format(model_type, missingness_type),
            params={'feature_columns': feature_columns, 'model_type': model_type},
            config=tf.estimator.RunConfig(
                save_summary_steps=1000,
                save_checkpoints_steps=6000,
                keep_checkpoint_max=1000,
                log_step_count_steps=1000))
        
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            {'x': x_train, 'b': b_train},
            shuffle=True)

        valid_input_fn = tf.estimator.inputs.numpy_input_fn(
            {'x': x_valid, 'b': b_valid},
            shuffle=False)
        
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            {'x': x_test, 'b': b_test},
            shuffle=False)

        best_valid_error = None
        best_checkpoint = None
        best_estimator = None
        epochs_since_improvement = 0
        for _ in range(max_epochs):
            vae.train(steps=None, input_fn=train_input_fn)

            eval_result = vae.evaluate(input_fn=train_input_fn)
            logging.info('End of epoch evaluation (train set): ' + str(eval_result))

            eval_result = vae.evaluate(input_fn=valid_input_fn)
            logging.info('End of epoch evaluation (valid set): ' + str(eval_result))

            if best_valid_error is None or eval_result['loss'] < best_valid_error:
                best_checkpoint = vae.latest_checkpoint()
                best_valid_error = eval_result['loss']
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
        
        ### representation learning evaluation (train linear classifier on learned z)
        
        input_fn = tf.estimator.inputs.numpy_input_fn({'x': x_train, 'b': b_train}, shuffle=False)
        z_train = np.asarray([pred['z'] for pred in vae.predict(input_fn=input_fn,
                                                                       checkpoint_path=best_checkpoint)])
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            {'z': z_train}, y=y_train, shuffle=True)
        
        input_fn = tf.estimator.inputs.numpy_input_fn({'x': x_valid, 'b': b_valid}, shuffle=False)
        z_valid = np.asarray([pred['z'] for pred in vae.predict(input_fn=input_fn,
                                                                       checkpoint_path=best_checkpoint)])
        valid_input_fn = tf.estimator.inputs.numpy_input_fn(
            {'z': z_valid}, y=y_valid, shuffle=False)
        
        input_fn = tf.estimator.inputs.numpy_input_fn({'x': x_test, 'b': b_test}, shuffle=False)
        z_test = np.asarray([pred['z'] for pred in vae.predict(input_fn=input_fn,
                                                                       checkpoint_path=best_checkpoint)])
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            {'z': z_test}, y=y_test, shuffle=False)
        
        classifier = tf.estimator.LinearClassifier(
            feature_columns=[tf.feature_column.numeric_column(key='z', shape=[z_dim])],
            model_dir='mnist_{0}_{1}_classifier'.format(model_type, missingness_type),
            n_classes=10,
            config=tf.estimator.RunConfig(
                save_summary_steps=1000,
                save_checkpoints_steps=6000,
                keep_checkpoint_max=1000,
                log_step_count_steps=1000))
        
        best_valid_error = None
        best_checkpoint = None
        best_estimator = None
        epochs_since_improvement = 0
        for _ in range(max_epochs):
            classifier.train(steps=None, input_fn=train_input_fn)

            eval_result = classifier.evaluate(input_fn=train_input_fn)
            logging.info('End of epoch evaluation (train set): ' + str(eval_result))

            eval_result = classifier.evaluate(input_fn=valid_input_fn)
            logging.info('End of epoch evaluation (valid set): ' + str(eval_result))

            if best_valid_error is None or eval_result['loss'] < best_valid_error:
                best_checkpoint = classifier.latest_checkpoint()
                best_valid_error = eval_result['loss']
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= max_epoch_without_improvement:
                    break

        eval_result = classifier.evaluate(
            input_fn=valid_input_fn,
            checkpoint_path=best_checkpoint)
        logging.info('Valid set evaluation: {0}'.format(eval_result))
        
        eval_result = classifier.evaluate(
            input_fn=test_input_fn,
            checkpoint_path=best_checkpoint)
        logging.info('Test set evaluation: {0}'.format(eval_result))