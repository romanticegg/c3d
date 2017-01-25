"""
A set of helper functions for tensorflow
"""
import tensorflow as tf
import os


def print_tensor_shape(tensor, name=None):
    if name is None:
        name = ' '
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