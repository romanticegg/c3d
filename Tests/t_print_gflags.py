# check if there is a way to print the parameters out
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/Users/zijwei/Dev/datasets/UCF-101-g16/train', 'directory to save training data[/Users/zijwei/Dev/datasets]')
flags.DEFINE_string("save_name", None, "Directory in which to save output of this run[Currentdate such as 2017-01...]")
flags.DEFINE_integer('batch_size', 12, 'batch size[12]')
flags.DEFINE_boolean('rewrite', False, 'If rewrite training logs to save_name[False]')
flags.DEFINE_integer('max_steps', 100000, 'Number of training steps[100000]')
flags.DEFINE_integer('gpu_id', None, 'GPU ID [None]')
flags.DEFINE_float('init_lr', 0.05, 'initial learning rate[0.05]')
flags.DEFINE_float('lr_decay_rate', 0.5, 'Decay learning rate by [0.5]')
flags.DEFINE_integer('num_epoch_per_decay', 4, 'decay of learning rate every [4] epoches')
flags.DEFINE_float('weight_decay_conv', 0.0005, 'weight decay for convolutional layers [0.0005]')
flags.DEFINE_float('weight_decay_fc', 0.0005, 'weight decay for fully connected (fully convoluted) layers [0.0005]')
flags.DEFINE_float('dropout', 0.5, 'dropuout ratio[0.5]')

FLAGS = flags.FLAGS
def main(argv=None):

    for name, value in FLAGS.__flags.iteritems():
        print ' ', name, ':\t', value
    # print 'Hello World'


if __name__ == '__main__':
    tf.app.run()