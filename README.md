# YOLOv2 in PyTorch
This is a [PyTorch](https://github.com/pytorch/pytorch)
implementation of YOLOv2.
This project is forked from (https://github.com/longcw/yolo2-pytorch), but not compatible with origin version.

Currently, I train this model for [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/). So you 

For details about YOLO and YOLOv2 please refer to their [project page](https://pjreddie.com/darknet/yolo/) 
and the [paper](https://arxiv.org/abs/1612.08242):
YOLO9000: Better, Faster, Stronger by Joseph Redmon and Ali Farhadi.


### Installation and demo
1. Clone this repository
    ```bash
    git clone git@github.com:cory8249/yolo2-pytorch.git
    ```

2. Build the reorg layer ([`tf.extract_image_patches`](https://www.tensorflow.org/api_docs/python/tf/extract_image_patches))
    ```bash
    cd yolo2-pytorch
    ./make.sh
    ```
3. Download the trained model [kitti_baseline_v3_100.h5](https://drive.google.com/file/d/0B3IzhcU-mEUsWnBIcW00aUsteTQ) 
and set the model path in `yolo_detect.py`
4. Run demo `python3 yolo_detect.py`. 

