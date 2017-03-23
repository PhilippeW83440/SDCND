#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy and pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is: 34799  
* The size of validation set is: 4410 (a bit more than 10% of the data used during training for cross-validation) 
* The size of test set is: 12630
* The shape of a traffic sign image is: 32x32x3 (3 RGB channels)
* The number of unique classes/labels in the data set is: 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set.   
It is a bar chart showing how the data is actually unbalanced:  
- 17 classes have more than 1000 training samples, whith peaks at 2000 training samples.
- 19 classes have less than 500 training samples  
  
Ideally we would like to have the training, validation and test sets to be well balanced. So typically this could be a topic for data augmentation to make sure we are dealing with balanced classes.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

I am using a per image normalization: per image and per channel, I am computing the mean and standard deviation and changing pixels value to (x - mu) / stddev. Using a per image normalization rather than a global normalization like (x - 128) / 128 type of normalization (assuming all pixels are in a [0, 255] range of values) enabled to improve results significantly.  
I am not converting the images to grayscale: which could have the benefit of faster training times. I tried but got slightly better results with colored images and as the training is pretty fast (around 10 minutes with a GPU 980 TI card), I am sticking to colored images. I have also tried histogram equalization but as per my experiments so far, the key point was doing a **per image normalization**.  


Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

During training, a bit more than 10% of data is being used for cross-validation.  
The cross-validation accuracy is used as a trigger for storing the best performing model. So typically the training lasts 50 epochs, but **a new model is being stored and qualified only when cross-validation accuracy is improved**.  
At the beginning of every epoch the training set is shuffled: this is very important.

Also I am using one form of data augmentation: at every epoch I have the ability to derive from the training set a companion training set with modified images. The perturbations used are geometric: rotations, translation, scaling and perspective transforms. So actually **the training set is 2x the original training set, with 50% of the samples, the augmented ones, being geometric transformations of the original ones**. 
My best performing model, is make a perturbated copy of the original training set and using this perturbated copy during 8 epochs, before generating a new perturbated copy. And so on during training.  
This enabled me to get a Validation Accuracy above 99% (while I was at 98% without such, simple and limited data augmenation).  
I have also noticed that this improved the ability to generalize better and improved results with random traffic sign images retrieved on the web.  


The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because ... To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

The starting point was the LeNet model that was improved by:
- adding droput after the non-linearities (RELU) of the fully connected layers. This is a regularization feature to prevent overfitting and enable better generalization.  
- increasing the number of convolution filters: I started with 6 and 16 and gradualy multiplied by 2 as long as I got accuracy imporvements. Good choices are (48 filters for conv1 and 128 filters for conv2)
- I changed the random initialization of the weights by using a smaller standard deviation and it realy helped improved the accuracy. Weights initialization is a key point when dealing with Neural Networks.

The things I have tried and finally rejected were:
- Batch normalization before the relu non linearities. No improvement. Maybe because the per image mean var normalization is already providing benefits and the network here is relativelly small.  

I have not considered adding a 3rd convolutional layer as the input here is small 32x32 and the size out of the 2nd convolution layer is already down to 5x5. I think much more convolutional layers would make a lot of sense with bigger input images.   

The things I would like to try further:
- using DenseNet or  multi-scale features, which means that convolutional layers’ output is not only forwarded into subsequent layer, but is also branched off and fed together into the first fully connected layer. My first trials were not successful but  with more data augmentation and deeper network this could or should help.  

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 48 filters: 1x1 stride, valid padding, outputs 28x28x48 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x48 				|
| Convolution 5x5     	| 128 filters: 1x1 stride, valid padding, outputs 10x10x128 	|  
| RELU					|	non linearities											|  
| Max pooling	      	| 2x2 stride,  outputs 5x5x128 	|  
| Flatten | the 3D shape (feature maps) into a flat vector |
| Fully connected		| 120 neurons        									|  
| RELU					|	non linearities											|  
| DROPOUT					|	50% drop during training											|  
| Fully connected		| 84 neurons        									|  
| RELU					|												|
| DROPOUT					|	50% drop during training											|
| Softmax				| 43 classes       									|


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an ....

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
