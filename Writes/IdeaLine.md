# Idea line

## The beginning:

Starting: https://arxiv.org/pdf/1511.05440v6.pdf (LeCun beyond Mean Square error)

Problem:
1.  Images too small, only generate near future
2.  What if we add important region : more than L1, L2, GAN and GDL errors, we give more weights on the regions that will be looked at (Benefits:  a>remove errors on unimportant regions,  b>produce less number of possible futures)

[Some results: using 4 frames to predict the future 8 frames. Image size: 64 by 64](/Users/zijwei/Dev/Adversarial_Video_Generation/Save/ServerData/2016-12-29-07-26-51-sz-64/Images/Test/Step_0000177600)


## Future important region prediction

Based on the second idea, we want to do the following:
Given a set of images, we want to predict the future "important regions" based on previous frames.(a.k.a, video saliency prediction)
Some proof:

[Fully Conv Networks for Semantic prediction](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
And a bunch of work on single image saliency prediction using DNN structures

## Current Video Saliency prediction

Exactly the same structure, but the output is becoming current frames saliency

## What I'm doing right now:

Setting up the basic video structure from [C3D Model](https://arxiv.org/abs/1412.0767)

Some experiments have already been deployed.

What I'm working on recently
1. Tuning parameters for a base model. Difference between Training and Testing is large (Using more regularizations/Training-testing inconsistant?)

Some observations:

1. Video is always very long (~100 frames at least at 5 FPS per video, but structure only take 16 frames)
2. Dense sampling does not work well

**May be this is the perfect situation to apply Sparsity + Diversity to frame selection**

## To elaborate this idea:
1. [This paper](https://arxiv.org/pdf/1604.04494v1.pdf) mentioned:
>Breaking this structure into short clips and aggregating video-level
information by the simple average of clip scores or more sophisticated
schemes such as as LSTMs  is likely to be suboptimal

Take home message from this paper:
1. Either spatial or temporal, adding more frames/spatial content will give better performances (Fig 4)
