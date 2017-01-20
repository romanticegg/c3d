"""
A set of helper functions for tensorflow
"""
import tensorflow as tf

def print_tensor_shape(tensor, name=None):
    if name is None:
        name = ' '
    print '{:s} : [{:s}]'.format(name, ', '.join(map(str, tensor.get_shape().as_list())))
