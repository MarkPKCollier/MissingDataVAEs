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


x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                      train_size=50000,
                                                      test_size=10000)

# z_dim = 128
# input_units = x_train.shape[1] * x_train.shape[2]
# encoder_layers = [(32, 5, 1), (64, 5, 1), 256]
# decoder_layers = [256, 7*7*64,
#                   ([-1, 7, 7, 64], [5, 5, 32, 64], 32, [1, 2, 2, 1], [-1, 14, 14, 32], tf.nn.relu),
#                   (None, [5, 5, 2, 32], 2, [1, 2, 2, 1], [-1, 28, 28, None], None)]
# max_epochs = 75
# max_epoch_without_improvement = 10
# mnist_dim = x_train.shape[1]

z_dim = 256
input_units = x_train.shape[1] * x_train.shape[2]
encoder_layers = [(10, 5, 1), (20, 5, 1), 512]
# encoder_shapes = [28*28*1, 14*14*10, 7*7*20]
encoder_shapes = [(28, 28, 1), (14, 14, 10), (7, 7, 20)]
encoder_shapes_pre_pool = [(28, 28, 1), (28, 28, 10), (14, 14, 20)]
# decoder_layers = [256, 7*7*16,
#                   ([-1, 7, 7, 16], [5, 5, 8, 16], 8, [1, 2, 2, 1], [-1, 14, 14, 8], tf.nn.relu),
#                   (None, [5, 5, None, 8], 2, [1, 2, 2, 1], [-1, 28, 28, None], None)]
# decoder_layers = [256, 7*7*20,
#                   ([-1, 7, 7, 20], [5, 5, 10, 20], 10, [1, 2, 2, 1], [-1, 14, 14, 10], tf.nn.relu),
#                   (None, [5, 5, None, 10], 2, [1, 2, 2, 1], [-1, 28, 28, None], None)]
p_x_layers = [7*7*10,
              ([-1, 7, 7, 10], 20, 5, 2, tf.nn.relu),
              (None, 10, 5, 2, tf.nn.relu),
              (None, 1, 5, 1, None)]
max_epochs = 60
max_epoch_without_improvement = 5
mnist_dim = x_train.shape[1]
missingness_size = 'block' # pixel

def encoder(inputs, b=None, model_type=None):
    net = inputs
    b_net = b
    if b is not None:
        u = tf.get_variable('input_layer_mean',
                            shape=encoder_shapes_pre_pool[0],
                            initializer=tf.initializers.zeros())

        layers = [(net, b_net, u)]
    else:
        layers = [net]

    for i, layer in enumerate(encoder_layers):
        if isinstance(layer, tuple):
            filters, kernel_size, strides = layer

            if i == 0:
                if '_ind' in model_type and 'self_dropout' not in model_type:
                    in_channels = 2
                else:
                    in_channels = 1
            else:
                in_channels = encoder_layers[i-1][0]

            w = tf.get_variable('layer_{0}_w'.format(i),
                                shape=(kernel_size, kernel_size, in_channels, filters),
                                initializer=tf.contrib.layers.xavier_initializer())

            if b is not None:
                b_net_ = b_net
                u_ = u

                b_net = tf.nn.conv2d(b_net_, tf.abs(w),
                                     strides=(1, strides, strides, 1), padding='SAME')

                bias = tf.get_variable('layer_{0}_bias'.format(i),
                                       shape=encoder_shapes_pre_pool[i+1][2],
                                       initializer=tf.initializers.zeros())

                bias1 = tf.get_variable('layer_{0}_bias_b_net'.format(i),
                                       shape=encoder_shapes_pre_pool[i+1][2],
                                       initializer=tf.initializers.zeros())

                gamma = tf.get_variable('layer_{0}_gamma_b_net'.format(i),
                                       shape=encoder_shapes_pre_pool[i+1][2],
                                       initializer=tf.initializers.ones())

                u = tf.get_variable('layer_{0}_mean'.format(i),
                                    shape=encoder_shapes_pre_pool[i+1],
                                    initializer=tf.initializers.zeros())

                b_net = tf.stop_gradient(b_net)
                b_net = tf.nn.bias_add(b_net, bias1)
                b_net = tf.nn.sigmoid(gamma * b_net)
                
                net = (tf.nn.conv2d(net, w, strides=(1, strides, strides, 1), padding='SAME') +
                       tf.nn.conv2d((1.0 - b_net_) * u_, w, strides=(1, strides, strides, 1), padding='SAME'))

                net = tf.nn.bias_add(net, bias)
                net = b_net * net
                net = tf.nn.relu(net)

                layers.append((net, b_net, u))

                net, indicies = tf.nn.max_pool_with_argmax(net,
                                                           ksize=[1, 2, 2, 1],
                                                           strides=[1, 2, 2, 1],
                                                           padding='SAME')

                b_net = tf.gather(tf.reshape(b_net, [-1]), indicies)
                b_net = tf.reshape(b_net, tf.shape(net))

                multiples = [tf.shape(net)[0], 1, 1, 1]
                u = tf.gather(
                        tf.reshape(tf.tile(tf.expand_dims(u, axis=0), multiples), [-1]),
                        indicies)
                u = tf.reshape(u, tf.shape(net))
            else:
                net = tf.nn.conv2d(net, w, strides=(1, strides, strides, 1), padding='SAME')

                bias = tf.get_variable('layer_{0}_bias'.format(i),
                                       shape=encoder_shapes_pre_pool[i+1][2],
                                       initializer=tf.initializers.zeros())

                net = tf.nn.bias_add(net, bias)
                net = tf.nn.relu(net)
                layers.append(net)

                net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        else:
            net = tf.contrib.layers.flatten(net)

            w = tf.get_variable('layer_{0}_w'.format(i),
                                shape=(7*7*20, layer),
                                initializer=tf.contrib.layers.xavier_initializer())
            
            bias = tf.get_variable('layer_{0}_bias'.format(i),
                                   shape=layer,
                                   initializer=tf.initializers.zeros())
            if b is not None:
                b_net = tf.contrib.layers.flatten(b_net)
                u = tf.contrib.layers.flatten(u)
                b_net_ = b_net
                u_ = u

                bias1 = tf.get_variable('layer_{0}_bias_b_net'.format(i),
                                        shape=layer,
                                        initializer=tf.initializers.zeros())

                gamma = tf.get_variable('layer_{0}_gamma_b_net'.format(i),
                                        shape=layer,
                                        initializer=tf.initializers.ones())

                u = tf.get_variable('layer_{0}_mean'.format(i),
                                    shape=layer,
                                    initializer=tf.initializers.zeros())

                b_net_transformed = tf.matmul(b_net, tf.abs(w))
                b_net_transformed = tf.stop_gradient(b_net_transformed)
                b_net = tf.nn.sigmoid(gamma * (b_net_transformed + bias1))

                net = tf.nn.relu(b_net *
                                 (tf.matmul(net, w) +  tf.matmul((1.0 - b_net_) * u_, w) + bias))

                layers.append((net, b_net, u))
            else:
                net = tf.nn.relu(tf.matmul(net, w) + bias)
                layers.append(net)
            
    net = tf.contrib.layers.flatten(net)
    mu = tf.contrib.layers.fully_connected(net, z_dim, activation_fn=None)
    sigma = tf.contrib.layers.fully_connected(net, z_dim, activation_fn=tf.nn.softplus)
    
    return mu, sigma, layers

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
        p = tf.contrib.layers.flatten(net[:, :, :, 0])

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
            b = tf.contrib.layers.flatten(net[:, :, :, 0])

        return p, b
    
    return p, None

# def decoder(z, recon_b=False):
#     net = z
#     for i, layer in enumerate(decoder_layers):
#         if isinstance(layer, tuple):
#             input_size, weights, bias, strides, output_shape, activation = layer
#             if input_size is not None:
#                 net = tf.reshape(net, input_size)
#             output_shape = [tf.shape(net)[0]] + output_shape[1:]
#             if output_shape[-1] == None:
#                 output_shape[-1] = 2 if recon_b else 1
#                 weights[-2] = 2 if recon_b else 1
            
#             w = tf.get_variable('decoder_weights_{0}'.format(i), weights,
#                                 initializer=tf.contrib.layers.xavier_initializer())
#             b = tf.get_variable('decoder_bias_{0}'.format(i), bias,
#                                 initializer=tf.constant_initializer(0.0))
            
#             net = tf.nn.conv2d_transpose(net, w, output_shape=output_shape, strides=strides)
#             net = net + b
#             if activation is not None:
#                 net = activation(net)
#         else:
#             net = tf.contrib.layers.fully_connected(net, layer, activation_fn=tf.nn.relu)
    
#     p_net = tf.contrib.layers.flatten(net[:, :, :, 0])
#     p = tf.contrib.layers.fully_connected(p_net, mnist_dim * mnist_dim, activation_fn=None)
    
#     if recon_b:
#         net = z
#         for i, layer in enumerate(decoder_layers):
#             if isinstance(layer, tuple):
#                 input_size, weights, bias, strides, output_shape, activation = layer
#                 if input_size is not None:
#                     net = tf.reshape(net, input_size)
#                 output_shape = [tf.shape(net)[0]] + output_shape[1:]
#                 if output_shape[-1] == None:
#                     output_shape[-1] = 2 if recon_b else 1
#                     weights[-2] = 2 if recon_b else 1
                
#                 w = tf.get_variable('decoder_recon_b_weights_{0}'.format(i), weights,
#                                     initializer=tf.contrib.layers.xavier_initializer())
#                 b = tf.get_variable('decoder_recon_b_bias_{0}'.format(i), bias,
#                                     initializer=tf.constant_initializer(0.0))
                
#                 net = tf.nn.conv2d_transpose(net, w, output_shape=output_shape, strides=strides)
#                 net = net + b
#                 if activation is not None:
#                     net = activation(net)
#             else:
#                 net = tf.contrib.layers.fully_connected(net, layer, activation_fn=tf.nn.relu)

#             b = tf.contrib.layers.fully_connected(net, mnist_dim * mnist_dim, activation_fn=None)
#             return p, b
    
#     return p, None

def model(features, labels, mode, params):
    x = tf.reshape(tf.feature_column.input_layer(features, params['feature_columns'][0]),
                   [-1, mnist_dim, mnist_dim])
    b = tf.reshape(tf.feature_column.input_layer(features, params['feature_columns'][1]),
                   [-1, mnist_dim, mnist_dim])
    recon_b = 'recon_b' in params['model_type']
    
    # mu, sigma = encoder(tf.stack([x, b], axis=3) if '_ind' in params['model_type'] else tf.expand_dims(x, -1),
    #                     b=tf.expand_dims(b, 3) if 'self_dropout' in params['model_type'] else None)
    
    mu, sigma, layers = encoder(tf.stack([x * b, b], axis=3) if '_ind' in params['model_type'] and 'self_dropout' not in params['model_type'] else tf.expand_dims(x * b, -1),
                                b=tf.expand_dims(b, 3) if 'self_dropout' in params['model_type'] else None,
                                model_type=params['model_type'])

    q_z = tf.distributions.Normal(mu, sigma)
    
    p_z = tf.distributions.Normal(loc=np.zeros(z_dim, dtype=np.float32), scale=np.ones(z_dim, dtype=np.float32))
        
    decoder_inputs = z_sample = q_z.sample()
    if 'dec_cond_b' in params['model_type']:
        decoder_inputs = tf.concat([z_sample, tf.contrib.layers.flatten(b)], axis=1)
    x_logits, b_logits = decoder(decoder_inputs, recon_b=recon_b)
    x_logits = tf.reshape(x_logits, [-1, mnist_dim, mnist_dim])
    if b_logits is not None:
        b_logits = tf.reshape(b_logits, [-1, mnist_dim, mnist_dim])
    x_pred = tf.nn.sigmoid(x_logits)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode,
            predictions={'x': x_pred, 'z': mu},
            export_outputs={'y': tf.estimator.export.ClassificationOutput(scores=x_pred)})
    
#     p_x_z = tf.distributions.Bernoulli(probs=x_mu)
    
    kl = tf.reduce_mean(tf.reduce_sum(tf.distributions.kl_divergence(q_z, p_z), axis=1))
    # log_prob = -1 * tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(
    #     b * tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_logits), axis=2), axis=1))
#     log_prob = tf.reduce_mean(tf.reduce_sum(p_x_z.log_prob(inputs), axis=1))

    log_prob = -1 * tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(b * 
        tf.nn.sigmoid_cross_entropy_with_logits(labels=b * x, logits=b * x_logits), axis=2), axis=1))
    imputation_log_prob = -1 * tf.reduce_mean(tf.reduce_sum(tf.reduce_sum((1.0 - b) * 
        tf.nn.sigmoid_cross_entropy_with_logits(labels=(1.0 - b) * x, logits=(1.0 - b) * x_logits), axis=2), axis=1))
    if recon_b:
        log_prob_b = -1 * tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=b, logits=b_logits), axis=2), axis=1))
        
        elbo = log_prob + log_prob_b - kl
    else:
        elbo = log_prob - kl

    if 'self_dropout' in params['model_type']:
        loss = -elbo
        for i, (net, b_net, u) in enumerate(layers):
            fixed_net = tf.stop_gradient(net)
            if i == 0:
                fixed_b_net = tf.stop_gradient(b_net)
                loss += tf.reduce_mean(tf.reduce_sum((fixed_b_net * (fixed_net - u))**2, axis=1))
            elif isinstance(encoder_layers[i-1], tuple):
                loss += tf.reduce_mean(tf.reduce_sum(
                    tf.reduce_sum((fixed_net - u)**2, axis=1), axis=2))
            else:
                loss += tf.reduce_mean(tf.reduce_sum((fixed_net - u)**2, axis=1))
    else:
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

    optimizer = tf.train.AdamOptimizer(learning_rate=0.002)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss,
            global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

# for missingness_type, p in [('dependent', 0.5), ('independent', 0.5)]:
for missingness_type, p in [('independent', 0.5)]:
    # for model_type in ['VAE_ind_dec_cond_b', 'VAE_ind_recon_b', 'VAE', 'VAE_ind']:
    # for model_type in ['VAE', 'VAE_ind']:
    for model_type in ['VAE_ind_self_dropout_recon_b', 'VAE_ind_self_dropout', 'VAE_ind', 'VAE', 'VAE_ind_recon_b']:
        if missingness_type == 'independent' and 'recon_b' in model_type:
            continue
        if missingness_type == 'independent' and missingness_size == 'block':
            b_train = np.ones_like(x_train)
            b_valid = np.ones_like(x_valid)
            b_test = np.ones_like(x_test)
            for b in range(x_train.shape[0]):
                x = np.random.choice(mnist_dim)
                y = np.random.choice(mnist_dim)
                b_train[b, max(x - 7, 0):min(x + 7, mnist_dim), max(y - 7, 0):min(y + 7, mnist_dim)] = 0.0
            for b in range(x_valid.shape[0]):
                x = np.random.choice(mnist_dim)
                y = np.random.choice(mnist_dim)
                b_valid[b, max(x - 7, 0):min(x + 7, mnist_dim), max(y - 7, 0):min(y + 7, mnist_dim)] = 0.0
            for b in range(x_test.shape[0]):
                x = np.random.choice(mnist_dim)
                y = np.random.choice(mnist_dim)
                b_test[b, max(x - 7, 0):min(x + 7, mnist_dim), max(y - 7, 0):min(y + 7, mnist_dim)] = 0.0
        elif missingness_type == 'independent':
            b_train = np.random.uniform(size=x_train.shape) > p
            # x_train_ = b_train * x_train
            b_valid = np.random.uniform(size=x_valid.shape) > p
            # x_valid_ = b_valid * x_valid
            b_test = np.random.uniform(size=x_test.shape) > p
            # x_test_ = b_test * x_test
        elif missingness_type == 'dependent':
            b_train = np.random.uniform(size=x_train.shape) > np.expand_dims(
                np.expand_dims(p * (1 + y_train)/10.0, axis=1), axis=2)
            # x_train_ = b_train * x_train
            b_valid = np.random.uniform(size=x_valid.shape) > np.expand_dims(
                np.expand_dims(p * (1 + y_valid)/10.0, axis=1), axis=2)
            # x_valid_ = b_valid * x_valid
            b_test = np.random.uniform(size=x_test.shape) > np.expand_dims(
                np.expand_dims(p * (1 + y_test)/10.0, axis=1), axis=2)
            # x_test_ = b_test * x_test

        feature_columns = [tf.feature_column.numeric_column(key='x', shape=[mnist_dim, mnist_dim]),
                           tf.feature_column.numeric_column(key='b', shape=[mnist_dim, mnist_dim])]

        vae = tf.estimator.Estimator(
            model_fn=model,
            model_dir='mnist_conv_{0}_{1}'.format(model_type, missingness_type),
            params={'feature_columns': feature_columns, 'model_type': model_type},
            config=tf.estimator.RunConfig(
                save_summary_steps=1000,
                save_checkpoints_steps=20000,
                keep_checkpoint_max=200,
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
        
        ### representation learning evaluation (train linear classifier on learned z)
        
        # input_fn = tf.estimator.inputs.numpy_input_fn({'x': x_train_, 'b': b_train}, shuffle=False)
        # z_train = np.asarray([pred['z'] for pred in vae.predict(input_fn=input_fn,
        #                                                                checkpoint_path=best_checkpoint)])
        # train_input_fn = tf.estimator.inputs.numpy_input_fn(
        #     {'z': z_train}, y=y_train, shuffle=True)
        
        # input_fn = tf.estimator.inputs.numpy_input_fn({'x': x_valid_, 'b': b_valid}, shuffle=False)
        # z_valid = np.asarray([pred['z'] for pred in vae.predict(input_fn=input_fn,
        #                                                                checkpoint_path=best_checkpoint)])
        # valid_input_fn = tf.estimator.inputs.numpy_input_fn(
        #     {'z': z_valid}, y=y_valid, shuffle=False)
        
        # input_fn = tf.estimator.inputs.numpy_input_fn({'x': x_test_, 'b': b_test}, shuffle=False)
        # z_test = np.asarray([pred['z'] for pred in vae.predict(input_fn=input_fn,
        #                                                                checkpoint_path=best_checkpoint)])
        # test_input_fn = tf.estimator.inputs.numpy_input_fn(
        #     {'z': z_test}, y=y_test, shuffle=False)
        
        # classifier = tf.estimator.LinearClassifier(
        #     feature_columns=[tf.feature_column.numeric_column(key='z', shape=[z_dim])],
        #     model_dir='mnist_conv_{0}_{1}_classifier'.format(model_type, missingness_type),
        #     n_classes=10,
        #     config=tf.estimator.RunConfig(
        #         save_summary_steps=1000,
        #         save_checkpoints_steps=6000,
        #         keep_checkpoint_max=1000,
        #         log_step_count_steps=1000))
        
        # best_valid_error = None
        # best_checkpoint = None
        # best_estimator = None
        # epochs_since_improvement = 0
        # for _ in range(max_epochs):
        #     classifier.train(steps=None, input_fn=train_input_fn)

        #     eval_result = classifier.evaluate(input_fn=train_input_fn)
        #     logging.info('End of epoch evaluation (train set): ' + str(eval_result))

        #     eval_result = classifier.evaluate(input_fn=valid_input_fn)
        #     logging.info('End of epoch evaluation (valid set): ' + str(eval_result))

        #     if best_valid_error is None or eval_result['loss'] < best_valid_error:
        #         best_checkpoint = classifier.latest_checkpoint()
        #         best_valid_error = eval_result['loss']
        #         epochs_since_improvement = 0
        #     else:
        #         epochs_since_improvement += 1
        #         if epochs_since_improvement >= max_epoch_without_improvement:
        #             break

        # eval_result = classifier.evaluate(
        #     input_fn=valid_input_fn,
        #     checkpoint_path=best_checkpoint)
        # logging.info('Valid set evaluation: {0}'.format(eval_result))
        
        # eval_result = classifier.evaluate(
        #     input_fn=test_input_fn,
        #     checkpoint_path=best_checkpoint)
        # logging.info('Test set evaluation: {0}'.format(eval_result))
