#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)  

[image0]: ./examples/simulator.png "Simulator" 
[image1]: ./examples/center_sample.jpg "Center"  
[image2]: ./examples/left_sample.jpg "Left"  
[image3]: ./examples/right_sample.jpg "Right"  
[image4]: ./examples/recov1.jpg "Recovery Image"  
[image5]: ./examples/recov2.jpg "Recovery Image"  
[image6]: ./examples/recov3.jpg "Recovery Image"  
[image7]: ./examples/recov4.jpg "Recovery Image"  
[image8]: ./examples/recov5.jpg "Recovery Image"  
[image9]: ./examples/original.jpg "Original Image"  
[image10]: ./examples/cropped.jpg "Cropped Image"  
[image11]: ./examples/history.png "History"  

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

![alt text][image0]  

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model architecture corresponds to the Nvidia architecture described in the paper: End to End Learning for Self-Driving Cars  
https://arxiv.org/pdf/1604.07316.pdf  

Model based on Nvidia's end-to-end architecture:  
model = Sequential()  
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))  
model.add(Cropping2D(cropping=((CROP_TOP, CROP_BOTTOM), (0,0))))  
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))  
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))  
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))  
model.add(Convolution2D(64,3,3, activation='relu'))  
model.add(Convolution2D(64,3,3, activation='relu'))  
model.add(Flatten())  
model.add(Dense(100, activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(50, activation='relu'))  
model.add(Dense(10, activation='relu'))  
model.add(Dense( 1))  

The total number of parameters is: 348219. 

My model consists of a convolution neural network with hierarchical 5x5 and 3x3 filter sizes followed by fully connected layers. 

The model includes RELU layers to introduce nonlinearity  and the data is normalized in the model using a Keras lambda layer.
Additionaly cropping is integrated into the neural network so that this operation is performed on GPU: which is faster.    

In the context of this project a smaller network would probably have been good enough but I wanted to experiment with a network architecture that is suitable for larger scale and more realistic environments as reported in the scientific publication by Nvidia. This architecture and pipeline should be the basis for further testing based on real camera inputs taken from a car. 

To overcome potential memory issues when dealing with big data sets, Keras fit_generator is used.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. Moreover early stop is being used to prevent overfitting. 

![alt text][image11]  

Model produced at epoch 7 is elected based on lowest validation loss.  

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 

I used a combination of center lane driving: .
load_data('./driving_data/track1_drive/', 'driving_log.csv', samples)

and recovering from the left and right sides of the road:
load_data('./driving_data/track1_recovery/', 'driving_log.csv', samples)

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to follow the Nvidia paper: which is a reference on this topic.  

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.  
80% of the data was used for the training set and 20% for the valisation set.  


To combat the overfitting, I modified the model so that it used dropouts after the biggest fully connected layer.


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I added recovery recordings.

At the end of the process, the vehicle is able to drive autonomously, at full speed 30 mph, around the track without leaving the road, endlessly, on track1.

For track2, I still need to record good training data (I have some hard times trying to drive it manually).  

####2. Final Model Architecture

The final model architecture corresponds to  Nvidia End to End Learning for Self-Driving Cars + dropouts.


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of: 
+ center lane driving:   
![alt text][image1]  
+ associated left camera:  
![alt text][image2]
+ associated right camera:  
![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image4]  
![alt text][image5]  
![alt text][image6]  
![alt text][image7]  
![alt text][image8]  

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would helps the model to generalize.  



After the collection process, I had 7874 images corresponding to a "normal track" + 1458 images corresponding to "recovery operatios". 

I  randomly shuffled the data set and put 20% of the data into a validation set. 

To summarize:
- Left and right cameras are used to improve recovery: angle +/- 0.2 are used for left/right cameras images vs center camera (idealy this should be derived based on trigonometry and cinematics of the vehicle)  
- all images are being flipped: and -angle is being used for the flipped image  
So we end up with 6ximages for 1 center camera image.  
- 90% of the images corresponding to 0 degree steering angle are discarded to ensure this class of angles is not over-represented during training.  
- Top and bottom parts of the images are cropped: to make sure irrelevant parts for steering angle prediction, of the images are excluded.  
- to prevent overfitting: dropout, validation set and early stop are being used.  
- for scalability and to prevent memory issues: keras fit_generator is used.   
- for scalability and to prepare more real life scenarios, a powerfull Nvidia's end-to-end architecture is used. 
- for efficiency: normalization and cropping are performed as part of the Keras Neural Network model. So this will be handled by the GPU.    
- On track1, the model runs smoothly over and over without driving outside the road. I also tried that while making some small manual modifications, going left or right, when coming back to the autonomous mode, the car recovers and come back to the center of the road.    

This pipeline could now be used in a context similar to Udacity open source challenge 2 with real car camera images.  


