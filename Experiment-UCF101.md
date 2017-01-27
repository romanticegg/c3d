# Experiment sets
## Datasets:
  UCF 101
  Train/Test 01: 9537 training/3783 testing 

## experiment 1:
for the original paper, the spatial size of the video is resized to 128*171,  a random 112*112 region is cropped for training the network.

In our case, we set the crop size to be 128 * 128 in order to do fully convolution. We resize the videos to 150 * 200 accordingly.

**As mentioned in Apdix A of the paper, 128 by 128 seems to be way better than 64 by 64 **
