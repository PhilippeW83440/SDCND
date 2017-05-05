##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


[//]: # (Image References)
[image1]: ./writeup/car_notcar.png
[image2]: ./writeup/convnet_acc.png
[image3]: ./writeup/convnet_loss.png
[image4]: ./writeup/convnet_hotmap.png
[image5]: ./writeup/convnet_hotmap_gray.png
[image6]: ./writeup/convnet_augmented.png
[image7]: ./writeup/heatmap.png
[image8]: ./writeup/pipeline_result.png
[image9]: ./writeup/test5_result.png
[video1]: ./writeup/project_video_out1.mp4
[video2]: ./writeup/p5_video_out1.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README


![alt text][image9]


```python
heatmaps.clear()
output = 'project_video_out1.mp4'
clip1 = VideoFileClip("project_video.mp4")
test_clip = clip1.fl_image(process_image)
%time test_clip.write_videofile(output, audio=False)

#[MoviePy] >>>> Building video project_video_out1.mp4
#[MoviePy] Writing video project_video_out1.mp4
#100%|█████████▉| 1270/1271 [00:53<00:00, 23.63it/s]
#[MoviePy] Done.
#[MoviePy] >>>> Video ready: project_video_out1.mp4 
```

Here's a [link to my video result](https://www.youtube.com/watch?v=9iieJO-0upU)

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  


The approach I have chosen is a Deep Learning approach that is not relying on existing DL object detection frameworks (no Yolo, no SSD, no SqueezeDet re-use here).  
The implementation is fully self-contained here, done in python and Keras and actually maches pretty well with the pure Computer Vision steps expected in the below rubric.  

The steps used are the following:  
- A 64x64 image car classifier is built and trained on the GTI/KITTI data provided. **This classifier is a fully convolutional neural network**    
- Then this 64x64 car classifier is used as the building block of a sliding window and **multi scales** detector applied on 720x1280 images  
- Heatmaps, thresholding and filtering over a few consecutive frames are used **to remove false positive detections** and track vehicles over consecutive frames  

The implementation is fast: around 25 fps (on my PC: iCore7 + GPU GTX 980 TI). So much faster than a typical HOG+SVM CV approach while enabling to detect vehicules in both directions and is comparable to existing DL detection frameworks in terms of speed and ability to detect vehicles at different scales.  

In terms of write up, I am following the proposed rubric points template. Even if it was initially targetted for the HOG+SVM CV approach, appart for some specific details, eg the use of HOG features, in terms of steps and project breakdowns and implmentation there is a 1-1 correspondance between the HOG+SVM CV approach and the Deep Learning CNN detection approach presented here.  
  

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.



This is one of the interest of DL approaches, you do not have to hand-design specific features. You can usually deal with raw pixels. The Neural Network has the ability to learn a good representation of the raw input pixels. So the input here is a standard RGB image/pixels. Apart from mean-var normalization no specific pre-processing or features extraction is performed.  


The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]



####2. Explain how you settled on your final choice of HOG parameters.

Raw RGB pixels are being used as input. Mean-var normalization is integrated in the fully convolutionnal classifier: so it will be handled on the GPU as well.  
So the features extraction step is fast and minimalist.  

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a fully convolutional Neural Network using Keras framework.  
  
The keyword here is **fully** convolutional Neural Network. Traditionnaly DL classifiers are made up of several convolutional layers folowed by a few dense layers. Here the dense layers are replaced by convolutionnal layers as mentionned in the paper referred in (1) in the credits: Fully convolutionnal networks for Semantic Segmentation, https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf. 

This approach is typically used as the basis of segmentation, to build heatmaps, and indeed this what we are going to do. But instead of performing segmentation we will use the heatmap built by this fully convolutionnal network, to detect objects, on the heat areas.  
  
So the classifier is a simple CNN network with the 2 final dense layers replaced by 1x1 convolutional filters where we used to have dense connections.  
  
The nice property of a convolutional filter or a fully convolutional neural network (built initialy for classifying 64x64 images) is that it will naturally slide over bigger images (720x1280 in our case) to create outputs that are no more 1 dimension (car not car for a 64x64 image) but a feature map or heatmap that will tell us for every 64x64 possible positions within the 720x1280 image what the car probability will be for a specific location.  

```python
def convnet(input_shape=(64,64,3), filename=None):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=input_shape, output_shape=input_shape))
    model.add(Convolution2D(48, 3, 3, activation='relu', name='conv1',input_shape=input_shape, border_mode="same"))
    model.add(Convolution2D(48, 3, 3, activation='relu', name='conv2',border_mode="same"))
    model.add(MaxPooling2D(pool_size=(8,8)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(128,8,8,activation="relu",name="dense1")) # This was Dense(128)
    model.add(Dropout(0.5))
    model.add(Convolution2D(1,1,1,name="dense2", activation="tanh")) # This was Dense(1)
    if filename:
        model.load_weights(filename)        
    return model

model = convnet()
model.add(Flatten()) # (None, 1, 1, 1) -> (None, 1)
model.summary()
```

Total params: 415601

91.95 Seconds to train the model...  
Test score: 0.00289260983378  
Test accuracy: 0.996621621622  

The amount of parameters here is relatively low for a Neural Network classifier. The training is fast and the accuracy on the validation set is 99.6%.  
90% of the provided data has been used for the training set and 10% has been used as a validation set.  

![alt text][image2]
![alt text][image3]

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Two techniques are being used here:  
-  the fully convolutionnal neural network 64x64 classifier will naturally slide over an image that is much bigger than 64x64 to produce a feature map (or heatmap) ouput. As represented below.


By sliding with our 64x64 classifier over (720, 1280, 3) images we get an output that is (83, 153, 1); mainly dividing size by 8 as we are using an 8x8 pooling and the filters have been defined so the borders reamain the same (before pooling). So we get a 2D ouput where the (x,y) pixel value corresponds to a number between -1 and 1. The closer the pixel value to 1, the highest the probability that there is a car located within the m-rectangle (mx=8*x, my=8*y, mw=64, mh=64) in the original input image.  

**This m-rectangle is 1 of the candidate bounding box for object detection.  There will be many of them, most of the time overlapping. We will perform a grouping of these overlapping bounding boxes to extract 1 bounding box per detected object.**  



![alt text][image4]  

A thresholding is applied to get a binary output.  

![alt text][image5]  

- 3 different scales are being handled here (0.5, 1.2 and 2)

First of all the image is split or cropped into 4 different areas:
- upper part: corresponding to sky and trees mainly, is excluded from the search window.
- the area from 400:550 y-corrdinates: is searched at a scale of 0.5.   
- the are from 450:550 y-coordinates: is searched at a scale of 1.2.  
- the area from 500:660 y-coordinates: is searched at a scale of 2.   

The rationale is that cars at the bottom of the image will appear bigger than the ones at upper parts.  
It will also help the detection to run faster by not sliding over the full image for every possible scales.  

Note that to search at a scale of eg 2, we resize the original image by making it 2 times smaller and then run the 64x64 classifier over it.  
Similarly to search at a scale of eg 0.5, we resize the original image by making it 2 times bigger and then run the 64x64 classifier over it.  

The image processing pipeline is the following:
- SCALE0 0.5 search on 400:550 y-coordinates area.
- SCALE1 1.2 search on 400:550 y-coordinates area
- SCALE2 2.0 search on 500:660 y-coordinates.

Candidate bounding boxes are captured:
- boxes_front: for cars driving in the opposite direction typically
- boxes: for cars driving in the same direction than our car

Then as explained in the lectures, overlapping boxes are grouped together and overlayed over the detected object.  

```python
HOTMAP_THRES = 0.999 #0.99
MH = 64
MW = 64
CROP_YMIN = 400
CROP_YMAX = 660
N_FRAMES_FILTER = 3 # 3

SCALE0 = 0.5
SCALE1 = 1.2 # 1.2
SCALE2 = 2 #1.8

heatmodel = convnet(input_shape=(None, None, 3), filename="convnet.h5")

import collections
heatmaps = collections.deque(maxlen=N_FRAMES_FILTER)

from scipy.ndimage.measurements import label

def process_image(img):
    orig_img = img.copy()
    # We crop the image to 400-660px in the vertical direction from 720x1280
    #crop_img = orig_img[CROP_YMIN:CROP_YMAX, :] # TODO MAKE THAT DYNAMIC based on img.shape
    
    boxes_front = []
    
    crop_img = orig_img[400:550, 0:400]
    dimx = int(crop_img.shape[0]/SCALE0)
    dimy = int(crop_img.shape[1]/SCALE0)
    resized_img = cv2.resize(crop_img, (dimy, dimx), interpolation=cv2.INTER_AREA)
    hotmap0 = heatmodel.predict(resized_img.reshape(1, resized_img.shape[0], resized_img.shape[1], resized_img.shape[2]))
    find_boxes_front(hotmap0, boxes_front, scale=SCALE0, crop_ymin=400)
    
    for (mx,my,mw,mh) in boxes_front:
        cv2.rectangle(orig_img, (mx, my), (mx+mw, my+mh), (255,0,0), 5)
    
    boxes = []
    
    crop_img = orig_img[CROP_YMIN:550, 150::]
    dimx = int(crop_img.shape[0]/SCALE1)
    dimy = int(crop_img.shape[1]/SCALE1)
    resized_img = cv2.resize(crop_img, (dimy, dimx), interpolation=cv2.INTER_AREA)
    hotmap1 = heatmodel.predict(resized_img.reshape(1, resized_img.shape[0], resized_img.shape[1], resized_img.shape[2]))
    find_boxes(hotmap1, boxes, scale=SCALE1, crop_ymin=CROP_YMIN, crop_xmin=150)
    
    crop_img = orig_img[500:CROP_YMAX, 150::]
    dimx = int(crop_img.shape[0]/SCALE2)
    dimy = int(crop_img.shape[1]/SCALE2)
    resized_img = cv2.resize(crop_img, (dimy, dimx), interpolation=cv2.INTER_AREA)
    hotmap2 = heatmodel.predict(resized_img.reshape(1, resized_img.shape[0], resized_img.shape[1], resized_img.shape[2]))
    find_boxes(hotmap2, boxes, scale=SCALE2, crop_ymin=500, crop_xmin=150)  
    
    #for (mx,my,mw,mh) in boxes:
    #    cv2.rectangle(orig_img, (mx, my), (mx+mw, my+mh), (0,0,255), 5)
    #return orig_img

    heat = np.zeros_like(orig_img[:,:,0]).astype(np.float)
    
    # Add heat to each box in box list
    heat = add_heat(heat, boxes)
    heatmaps.append(heat)
    heatmaps_sum = sum(heatmaps)
    
    # Apply threshold to help remove false positives
    heatmaps_sum = apply_threshold(heatmaps_sum, N_FRAMES_FILTER*2) #*3

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heatmaps_sum, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(orig_img, labels)
    
    return draw_img
```



####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

 Here are some example images:

![alt text][image8]
![alt text][image9]

To otimize the accuracy of the classifier I used:
- DL CNN classifier  
- 3 different scales (0.5, 1.2 and 2)  

To optimize the speed of the classifier I used:
- DL CNN classifier leveraging on GPU capabilities  
- cropping image for different limited searches. Upper parts of the image look for smaller cars, and bottom parts of the image look for bigger cars basically.    
  
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://www.youtube.com/watch?v=9iieJO-0upU)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:



![alt text][image6]
![alt text][image7]
![alt text][image8]  

Note that I have used different settings:
- for cars driving in the same direction than our car: 3 consecutive frames are being used and their heatmaps sumed. Then a threshold of 3*3 overlaps is used to identify areas where we will drow a bounding box.
- for cars driving in the opposite direction than our car: the relative speed is much higher so no consecutive frames are being used. Otherwise we would filter most of these detections.  

Different colors are used:
- for cars coming from the opposite direction: red  
- for cars driving in the same direction than our car: blue  
  

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

First of all I highly recommend reading the 3 links provided below in the credits. These are the roots of this implementation.  

Here I have used a Deep Learning approach but it has many things in common with the traditional HOG+SVM CV approach.  
The key step here, using fully convolutional networks to build heatmaps, is usually applied in a context of Images segmentation. Here it has been used to derive object detections. I am pretty happy with the result both in terms of accuracy (for vehichles driving in both directions, even if the training data was provided for cars seen only from behind !!!) and in terms of speed: around 25 fps.   
I have also re-used many things from the Udacity Computer Vision lectures to handle the grouping of candidate bounding boxes and filtering of false positives.  
I probably could improve the stability of the drawing of the bounding boxes from frame to frame by using a 1st order low pass filter (something like y[n]=0.8*y[n-1]+0.2*x[n] for the bounding boxes coordinates)  

In terms of next steps and improvements I really would like to study in details the SquezzeDet framework and architecture: 
https://arxiv.org/abs/1612.01051   
**Unified, Small, Low Power Fully Convolutional Neural Networks for Real-Time Object Detection for Autonomous Driving.**


**Credits:**  
(1) Fully convolutionnal networks for Semantic Segmentation:     
    https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf  
(2) heatmaps with convnets:   
    from a French startup Heuritech https://github.com/heuritech/convnets-keras   
(3) heatmaps with convnets:  
    from a fellow Udacity SDCND student https://medium.com/@tuennermann/   
