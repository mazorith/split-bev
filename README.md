# split-bev

### Install

This repo is split version of the BEV project from mmdet3d found here:

https://github.com/open-mmlab/mmdetection3d/tree/main

First set up your either your conda or pip env with pytorch>=2.0.0 and python>=3.9 

clone the repo and then install the mm libs here 

``` 
pip install mmengine
pip install mmdet
pip install mmcv
```

after installing these run the setup.py script located in your cloned mmdet3d repo. 

```python3 setup.py install```

or

```python3 setup.py install --user```

Some of the depences between these can be weird. If installing one dependancy reinstalls antoher mm lib version such that it overwrites the version you already installed try to reinstall it afterward. The core idea is that you need all mm libs compatable with the mmdet3d repo version you are cloning. This will take a bit of figuring out for your system as these repos have custom compiled code for pytorch and CUDA version.

### USE

You can replace the existing folder from the repo with the new one in mmdetectiond3d/projects/ directory

You can follow the instructions from the project folders README.md or at:
https://github.com/open-mmlab/mmdetection3d/tree/dev-1.x/projects/BEVFusion
to train and demo the model. Just use the split___.py config files instead of the default bevfusion ones.

I had to develop and create the bev_head.py that currently exists in the repo and it's functionality is difficult to work with. For the time being it is recommend that we use the default object detection model provided from the orginal repo to test input and outputs with carla as well as to start developing the visualization scripts which can be found at https://github.com/open-mmlab/mmdetection3d/tree/dev-1.x

There are some more scripts for visualization inside the mmdet3d repo but this demo script should be a good start for the that rabbit hole. 

Good luck and I'll talk to yall soon :)
