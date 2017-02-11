# Writing arguments

## About picking evaluation metrics

Different from , We only use the video level not only because as indicated in they are coherent with each other, but also the clip based evaluation is not sufficient since a large portion of them do not contain any relevant action.

## About not very well using the whole video data
If you compare the Video acc and table acc in Table 2 of paper [], you will find there is not a large improvement whereas more data are used.


## From the Fig 4 some observations:
1. The larger spatial and temporal the better
2. For some classes it is not a good idea to use all...


## Compare to others' work
1. Compare to ECCV2016, arbitary length of video can be used
2. Compare to Fully Convolution Layer (Conv --to replace max-pooling) more spatial variances
