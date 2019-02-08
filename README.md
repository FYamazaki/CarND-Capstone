This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

### Environment
This is an environment which I used.
* Ubuntu 16.04 Xenial Xerus. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
  * 4 CPU (2.4 GHz)
  * 8 GB system memory
  * Geforce GTX 660
  * 25 GB of free hard drive space
  * python 2.7.12
  * keras 2.0.8, tensorflow 1.3, CUDA 9
  * Tensorflow Object Detection API (https://github.com/tensorflow/models/tree/289a2f99a7df528f6193a5ab3ee284ff3112b731/object_detection)
* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).


### Traffic Light Detection
I used the traffic light detection [object_detection.py](https://github.com/FYamazaki/CarND-Capstone/blob/master/ros/src/tl_detector/light_classification/object_detector.py) first, then I used the traffic light classification [tl_classifier.py](https://github.com/FYamazaki/CarND-Capstone/blob/master/ros/src/tl_detector/light_classification/tl_classifier.py).
* The traffic light detection uses SSD(tensorflow object detection API).
This API detects the traffic light nicely.  
Detected Image: 
![alt text](https://github.com/FYamazaki/CarND-Capstone/blob/master/pictures/original_image.png "Original Image and Detected Box")
And I collected Traffic Light Images by this Object Detection.  
Collected Traffic Light Image:![alt text](https://github.com/FYamazaki/CarND-Capstone/blob/master/pictures/only_traffic_signal.png "Detected Traffic Light")
* I used V in HSV.
I resized to (60, 160), converted to HSV and only used V, because this looks more clear than gray scale and faster than original image.  
Converted Image:![alt text](https://github.com/FYamazaki/CarND-Capstone/blob/master/pictures/traffic_signal.png "Converted Image")
* The traffic light classification uses LeNet [TL_Detection.ipynb](https://github.com/FYamazaki/CarND-Capstone/blob/master/DeepLearning/TL_Detection.ipynb).  
Learning Curv:![alt text](https://github.com/FYamazaki/CarND-Capstone/blob/master/pictures/learning_curvV0203.png "Learning Curv")

The below is the network, I used.

Layer (type)                  | Output Shape     |Param # |  
------------------------------|-----------------|--------:|  
lambda_3 (Lambda)             | (None, 160, 60) | 0           
conv1d_3 (Conv1D)             | (None, 156, 5)  | 1505        
max_pooling1d_3 (MaxPooling1) | (None, 78, 5)   | 0           
dropout_7 (Dropout)           | (None, 78, 5)   | 0           
conv1d_4 (Conv1D)             | (None, 74, 5)   | 130         
max_pooling1d_4 (MaxPooling1) | (None, 37, 5)   | 0         
dropout_8 (Dropout)           | (None, 37, 5)   | 0         
flatten_3 (Flatten)           | (None, 185)     | 0         
dense_5 (Dense)               | (None, 256)     | 47616     
dropout_9 (Dropout)           | (None, 256)     |  0         
dense_6 (Dense)               | (None, 3)       | 771       

### The result
This is the final result.
[![Final Result](https://github.com/FYamazaki/CarND-Capstone/blob/master/pictures/simulator.png)](https://youtu.be/p2BWPz7WfZo)

### Discussion
I originally started VM on windows10 with simulator.  But when I turn on Camera, the car started going off.  So I switched to native Ubuntu 16.0. Then I started working on traffic light detection.  At the beginning, I tried to train whole image (800x600) by LeNet, but it didn't learn well.  So I decided to use object detection first.  I compared YOLO with SSD(Tensorflow Object Detection API), and YOLO (darknet) is slower than SSD. Then I decided to use SSD.  Now it works.  But this process is heavey on my poor laptop.  So, I call traffic light detection ony every 5th camera image.  If  I increase more, then car cannot stop at red light.  I wanted to work on more powerfull machine.  Now it barely works, but if the car runs more track and CPU get hotter, it eventually goes of the track.  I want to retrain SSD, but I gave up, because I need to create training data.

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images
