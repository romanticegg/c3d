import tensorflow as tf
from scipy.ndimage import imread
import PIL.Image as Image
import numpy as np

def test_reading_methods():
    # method 1:
    imagefile = '/Users/zijwei/Dev/datasets/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/00001.jpg'
    image1 = Image.open(imagefile)
    # to the test following the reading process in get_frame_data function
    # method 2:
    frame = imread(imagefile, mode='RGB')
    print ('Conclusion: using cv2 and scipy image for image reading and so on')


if __name__ == "__main__":
    test_reading_methods()