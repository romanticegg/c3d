# C3D-tensorflow

## Disclaimer:
This set of code is adapted from https://github.com/hx173149/C3D-tensorflow with modifications. I should have forked it but I'm too lazy and stupid.

If there is any copyright violation, please notify zijwei@cs.stonybrook.edu to delete

## Modifications:
1. The data loading process
2. Image loaders
3. options on RGB or BGR Image
4. In order to fix the uncertainties, no random sampling for frames was used during testing and training. the starting frame is (#allframes-samplelength)/2

## Introduction
This code is trying to improve the performance reported by @hx173149. Here I'm using the provided model without any modification. The performance, however, on UCF101 is as below:

|training (~9K)  | Testing (~3.7K)   |  
|---|---|
|   0.93| 0.91   |

This is higher than both the results posed on https://github.com/hx173149/C3D-tensorflow (~.72) and in the C3D paper (.85).

I double checked without noticing any bugs. But if you do find one, please let me know.

## Dependences
1. tensorflow
2. opencv (conda install opencv)
3. scipy.ndimage
4. [PIL] is **not** necessary


## How to run:
1. Prepare data like list/train.list and list/test.list, make sure the paths stored in are correct
2. `python train_c3d_ucf101.py --max_steps 0 --gpu_id 1`   Here 0 means only test

----------------------------------------------------------------------------------
**Below are from the original post:**

## Requirements:

1. You must have installed the following two python libs:
a) [tensorflow][1]
b) [Pillow][2]
2. You must have downloaded the [UCF101][3] (Action Recognition Data Set)
3. Each single avi file is decoded with 5FPS (it's depend your decision) in a single directory.
    - you can use the `./list/convert_video_to_images.sh` script to decode the ucf101 video files
    - run `./list/convert_video_to_images.sh .../UCF101 5`
4. Generate {train,test}.list files in `list` directory. Each line corresponds to "image directory" and a class (zero-based). For example:
    - you can use the `./list/convert_images_to_list.sh` script to generate the {train,test}.list for the dataset
    - run `./list/convert_images_to_list.sh .../dataset_images dataset_train`, this will generate `dataset_train.list` file inside the root folder

```
database/ucf101/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01 0
database/ucf101/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c02 0
database/ucf101/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c03 0
database/ucf101/train/ApplyLipstick/v_ApplyLipstick_g01_c01 1
database/ucf101/train/ApplyLipstick/v_ApplyLipstick_g01_c02 1
database/ucf101/train/ApplyLipstick/v_ApplyLipstick_g01_c03 1
database/ucf101/train/Archery/v_Archery_g01_c01 2
database/ucf101/train/Archery/v_Archery_g01_c02 2
database/ucf101/train/Archery/v_Archery_g01_c03 2
database/ucf101/train/Archery/v_Archery_g01_c04 2
database/ucf101/train/BabyCrawling/v_BabyCrawling_g01_c01 3
database/ucf101/train/BabyCrawling/v_BabyCrawling_g01_c02 3
database/ucf101/train/BabyCrawling/v_BabyCrawling_g01_c03 3
database/ucf101/train/BabyCrawling/v_BabyCrawling_g01_c04 3
database/ucf101/train/BalanceBeam/v_BalanceBeam_g01_c01 4
database/ucf101/train/BalanceBeam/v_BalanceBeam_g01_c02 4
database/ucf101/train/BalanceBeam/v_BalanceBeam_g01_c03 4
database/ucf101/train/BalanceBeam/v_BalanceBeam_g01_c04 4
...
```

5. If you want to test my pre-trained model, you need to download my model from here: https://www.dropbox.com/sh/8wcjrcadx4r31ux/AAAkz3dQ706pPO8ZavrztRCca?dl=0

## Run command:

1. `python train_c3d_ucf101.py` will train C3D model. The trained model will saved in `models` directory.
2. `python predict_c3d_ucf101.py` will test C3D model on a validation data set.

## Experiment result:

Top-1 accuracy of 72.6% should be achieved for the validation dataset with this code and pre-trained from the sports1M model. You can download my pretrained UCF101 model and mean file from here:
https://www.dropbox.com/sh/8wcjrcadx4r31ux/AAAkz3dQ706pPO8ZavrztRCca?dl=0

## References:

- Thanks the author [Du tran][4]'s code: [C3D-caffe][5]
- [C3D: Generic Features for Video Analysis][6]


[1]: https://www.tensorflow.org/
[2]: http://pillow.readthedocs.io/en/3.1.x/reference/Image.html
[3]: http://crcv.ucf.edu/data/UCF101.php
[4]: https://github.com/dutran
[5]: https://github.com/facebook/C3D
[6]: http://vlg.cs.dartmouth.edu/c3d/
