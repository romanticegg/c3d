"""
Creates a set of subdirectories to save the generated files when running tensorflow
Dependencies: utils
"""
import os
import utils


class tf_easy_dir:
    def __init__(self, save_dir):
        self.save_dir = utils.get_dir(save_dir)

        self.model_save_dir = utils.get_dir(os.path.join(self.save_dir, 'models'))
        self.summary_save_dir = utils.get_dir(os.path.join(self.save_dir, 'summaries'))
        self.image_save_dir = utils.get_dir(os.path.join(self.save_dir, 'images'))
        self.log_save_dir = utils.get_dir(os.path.join(self.save_dir, 'logs'))

    def clear_save_name(self):
        """
        Clears all saved content for SAVE_NAME.
        """
        utils.clear_dir(self.model_save_dir)
        utils.clear_dir(self.summary_save_dir)
        utils.clear_dir(self.log_save_dir)
        utils.clear_dir(self.image_save_dir)
        print ('Clear stuff in {}'.format(os.path.join(self.save_dir)))

