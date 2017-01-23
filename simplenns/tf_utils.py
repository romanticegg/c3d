"""
A set of helper functions for tensorflow
"""
import tensorflow as tf

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
def _variable_on_cpu(name, shape, initializer=None):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var


#create a variable with weight decay
#fixme: this should be simplified as _variable_on_cpu
def _variable_with_weight_decay(name, shape, initializer=None, wd=None):#, wd=None, stddev=None):
    # if not stddev:
    #     stddev = 5e-2
    #
    # if not initializer:
    #     initializer = tf.truncated_normal_initializer(stddev=stddev)

    var = _variable_on_cpu(name=name, shape=shape, initializer=initializer)

    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var
