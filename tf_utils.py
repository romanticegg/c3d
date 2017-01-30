"""
A set of helper functions for tensorflow
"""
import tensorflow as tf
import os


def print_tensor_shape(tensor, name=None):
    if name is None:
        name = tensor.op.name
    print '{:s} : [{:s}]'.format(name, ', '.join(map(str, tensor.get_shape().as_list())))


def get_all_names():
    '''
    get all the names of a graph
    :return: a list of names of all the nodes in current graph
    '''
    return [n.name for n in tf.get_default_graph().as_graph_def().node]


#create a variable on CPU
def variable_on_cpu(name, shape, initializer=None, trainable=True):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name=name, shape=shape, initializer=initializer, trainable=trainable)
    return var


#create a variable with weight decay
#fixme: this should be simplified as _variable_on_cpu
def variable_with_weight_decay(name, shape, initializer=None, wd=None):#, wd=None, stddev=None):
    # if not stddev:
    #     stddev = 5e-2
    #
    # if not initializer:
    #     initializer = tf.truncated_normal_initializer(stddev=stddev)

    var = variable_on_cpu(name=name, shape=shape, initializer=initializer)

    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


#batch normalization
def bn(x, isTraining=True, name=None, use_bias=False, moving_average_decay=0.9999, bn_epsilon=0.001):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]
    if not name:
        name= "bn"

    # returns a simple bias add operation
    if use_bias:
        bias = variable_on_cpu('bias_{:s}'.format(name), params_shape,
                               initializer=tf.zeros_initializer)
        return tf.nn.bias_add(x, bias=bias)


    axis = list(range(len(x_shape) - 1))

    beta = variable_on_cpu('beta_{:s}'.format(name),
                           params_shape,
                           initializer=tf.constant_initializer(0.0))
    gamma = variable_on_cpu('gamma_{:s}'.format(name),
                            params_shape,
                            initializer=tf.constant_initializer(1.0))

    moving_mean = variable_on_cpu('moving_mean_{:s}'.format(name),
                                  params_shape,
                                  initializer=tf.constant_initializer(0.0),
                                  trainable=False)
    moving_variance = variable_on_cpu('moving_variance_{:s}'.format(name),
                                      params_shape,
                                      initializer=tf.constant_initializer(0.0),
                                      trainable=False)

    if isTraining:
        mean, variance = tf.nn.moments(x, axis)
        train_mean = tf.assign(moving_mean,
                               moving_mean * moving_average_decay + mean * (1 - moving_average_decay))
        train_var = tf.assign(moving_variance,
                              moving_variance * moving_average_decay + variance * (1 - moving_average_decay))
        with tf.control_dependencies([train_mean, train_var]):
                x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, bn_epsilon, name=name)
    # else:
    #
    #     #update 1: default case: using the weighted train mean and var
    #     # fixme: result is not good
    #     x = tf.nn.batch_normalization(x, moving_mean, moving_variance, beta, gamma, bn_epsilon, name=name)
    else:

        #update 2: ignoring the learned variance and so on, only using the learned beta and gamma
        #fixme:
        # mean, variance = tf.nn.moments(x, axis)
        x = tf.nn.batch_normalization(x, moving_mean, moving_variance, beta, gamma, bn_epsilon, name=name)
    return x


# add histogram and sparisty summary
def activation_summary(x, tensor_name=None):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
    x: Tensor
    Returns:
    nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    if not tensor_name:
        #fixme: too complex, see if you can improve this later
        # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


#gpu_config:
def gpu_config(gpu_id=None):
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.6
        config.allow_soft_placement = True
    else:
        config = tf.ConfigProto()

    return config


def print_layer_info(layername, kernel=None, stride=None, reslt=None):

    print 'Layer {:s}'.format(layername)
    if kernel:
        print 'Kernel size [{:s}]'.format(', '.join(map(str, kernel)))
    if stride:
        print 'Stride size [{:s}]'.format(', '.join(map(str, stride)))
    if reslt:
        print 'Result size [{:s}]'.format(', '.join(map(str, reslt)))
    print '-' * 32