# Driver-Intention-Prediction
A framework predicts driver's maneuver behaviors.

Pytorch implementation of the paper ["Driver Intention Anticipation Based on In-Cabin and Driving Scene Monitoring"](https://arxiv.org/pdf/2006.11557.pdf).

Here is the demo of our framework:
![](demo.gif)

In this demo, the prediction is made after every second. If the prediction is correct, there is a ✓. 


## Dataset Preparation

The dataset used is downloaded from [Brain4cars](https://github.com/asheshjain399/ICCV2015_Brain4Cars).

1. Videos are extracted into images with the fps=25 under each directory. The file name format is e.g. "image-0001.png".

   You can use our script ``extract_frames.py`` in ``datasets/annotation`` to extract images: Copy this file to directory of "face_camera", and then run this script.

2. We split the dataset using 5-fold cross validaton. Run script ``n_fold_Brain4cars.py`` in directory ``datasets/annotation`` to split.

   You can use the five ``.csv`` files in ``datasets/annotation`` and skip this step.


## Train/Evaluate 3D-ResNet50 with inside videos


The network, 3D-ResNet 50 and its pretrained model is downloaded from [3D ResNets](https://github.com/kenshohara/3D-ResNets-PyTorch). 

We thank this project :)


Before running the run-3DResnet.sh script. Please give the path to: 
1. ``root_path``: path to this project.

2. ``annotation_path`` : path to annotation directory in this project.

3. ``video_path``: path to image frames of driver videos.

4. ``pretrain_path``: path to the pretrained 3D ResNet 50 model.


**Notes**:

1. ``n_fold``: is the number of the fold. Here, n_fold is from 0 to 4.

2. ``sample_duration``: the length of input vidoes. Here, 16 frames.

3. ``end_second``: the second before the maneuver. Here, end_second is from 1 to 5.

   E.g. end_second = 3, frames which are 3 seconds (including the third second) before maneuver are given as input.

More details about other args, please refer to the ``opt.py``.


The trained model using our script can be found under this [link](https://tumde-my.sharepoint.com/:f:/r/personal/yao_rong_tum_de/Documents/Driver-Intention-Prediction?csf=1&web=1&e=18wyqp). The model name is "save_best_3DResNet50.pth".


## Train/Evaluate ConvLSTM with outside videos

We used [FlowNet 2.0](https://github.com/NVIDIA/flownet2-pytorch) to extract the optical flow of all outside images.

You could also find these optical flow images under this [link](https://tumde-my.sharepoint.com/:f:/r/personal/yao_rong_tum_de/Documents/Driver-Intention-Prediction?csf=1&web=1&e=18wyqp).

We adapted our ConvLSTM network from this [repo](https://github.com/automan000/Convolutional_LSTM_PyTorch).

We thank those two projects :)


Before running the run-ConvLSTM.sh script. Please give the path to: 
1. ``root_path``: path to this project.

2. ``annotation_path`` : path to annotation directory in this project.

3. ``video_path``: path to image frames of optical flow images.


**Notes**:

1. ``n_fold``: is the number of the fold. Here, n_fold is from 0 to 4.

2. ``sample_duration``: the length of input vidoes. Here, 5 frames.

3. ``interval``: this is the interval of two frames inside the input clip. It should be between 5 to 30. 
                  More details can be found in the Section IV.B of the [paper](https://arxiv.org/pdf/2006.11557.pdf). 

4. ``end_second``: the second before the maneuver. Here, end_second is from 1 to 5.



If you find this work useful, please cite as follows:

```
@INPROCEEDINGS{9294181,
  author={Rong, Yao and Akata, Zeynep and Kasneci, Enkelejda},
  booktitle={2020 IEEE 23rd International Conference on Intelligent Transportation Systems (ITSC)}, 
  title={Driver Intention Anticipation Based on In-Cabin and Driving Scene Monitoring}, 
  year={2020},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/ITSC45102.2020.9294181}}
```
