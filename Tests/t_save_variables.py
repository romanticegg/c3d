# NOTE: very important functions for data save and load for specific variables
import tensorflow as tf
import numpy as np


# save the list of vars in to a list and then save to some readable format
def save_vars_to_dict(sess, var_list):
    val_dicts = {}
    for i in range(len(var_list)):
        name = var_list[i].op.name
        val = sess.run(var_list[i])
        val_dicts[name] = val

    return val_dicts


# update: a very important function to initialize
def custom_load(self, data_path, session, ignore_missing=False):
    data_dict = np.load(data_path).item()
    for key in data_dict:
        with tf.variable_scope(key, reuse=True):
            for subkey in data_dict[key]:
                try:
                    var = tf.get_variable(subkey)
                    session.run(var.assign(data_dict[key][subkey]))
                    print "assign pretrain model " + subkey + " to " + key
                except ValueError:
                    print "ignore " + key
                    if not ignore_missing:
                        raise