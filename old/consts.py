import os
import utils


IMAGE_FMT = 'jpg'
VIDEO_FMT = 'avi'
RGB = True

# Network structure related
NUM_CLASSES = 101
# Images are cropped to (CROP_SIZE, CROP_SIZE)
CROP_SIZE = 112
CHANNELS = 3
# Number of frames per video clip
NUM_FRAMES_PER_CLIP = 16

# root directory for all saved content
SAVE_DIR = utils.get_dir('./Save/')

# inner directory to differentiate between runs
SAVE_NAME = None
# directory for saved models
# MODEL_SAVE_DIR = utils.get_dir(os.path.join(SAVE_DIR, SAVE_NAME, 'Models'))
# # directory for saved TensorBoard summaries
# SUMMARY_SAVE_DIR = utils.get_dir(os.path.join(SAVE_DIR, SAVE_NAME, 'Summaries'))
# # directory for saved images
# IMG_SAVE_DIR = utils.get_dir(os.path.join(SAVE_DIR, SAVE_NAME, 'Images'))
MODEL_SAVE_DIR = None
# directory for saved TensorBoard summaries
SUMMARY_SAVE_DIR = None
# directory for saved images
IMG_SAVE_DIR = None

def set_save_name(name):
    """
    Edits all constants dependent on SAVE_NAME.

    @param name: The new save name.
    """
    global SAVE_NAME, MODEL_SAVE_DIR, SUMMARY_SAVE_DIR, IMG_SAVE_DIR

    SAVE_NAME = name
    MODEL_SAVE_DIR = utils.get_dir(os.path.join(SAVE_DIR, SAVE_NAME, 'Models'))
    SUMMARY_SAVE_DIR = utils.get_dir(os.path.join(SAVE_DIR, SAVE_NAME, 'Summaries'))
    IMG_SAVE_DIR = utils.get_dir(os.path.join(SAVE_DIR, SAVE_NAME, 'Images'))
    print ('Set save dir to {}'.format(os.path.join(SAVE_DIR, SAVE_NAME)))

def clear_save_name():
    """
    Clears all saved content for SAVE_NAME.
    """
    utils.clear_dir(MODEL_SAVE_DIR)
    utils.clear_dir(SUMMARY_SAVE_DIR)
    utils.clear_dir(IMG_SAVE_DIR)
    print ('Clear stuff in {}'.format(os.path.join(SAVE_DIR, SAVE_NAME)))