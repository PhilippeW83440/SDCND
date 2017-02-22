#**Finding Lane Lines on the Road** 

---

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./writeup_images/raw.png "Raw"
[image2]: ./writeup_images/white_and_yellow.png "White and yellow"
[image3]: ./writeup_images/grayscale.png "Grayscale"
[image4]: ./writeup_images/gaussian_blur.png "Gaussian Blur"
[image5]: ./writeup_images/canny.png "Canny"
[image6]: ./writeup_images/roi.png "Region Of Interest"
[image7]: ./writeup_images/raw_with_lane_lines_detected.png "Raw with detected lane lines"
[image8]: ./writeup_images/extra.gif "Extra"


---

### Reflection

###1. Pipeline description.
  
Let's consider an example with an image extracted from the extra challenge.  
Starting with this raw image, that has some curvature and shadows, let's follow up step by step the processing pipeline.  

![alt text][image1]  
  
My pipeline consisted of following steps steps:  
  
**1. Color selection**  
New Function: select_white_yellow  
White and Yellow pixels are extracted in HLS space: cf https://en.wikipedia.org/wiki/HSL_and_HSV  
White is identified with high L (Light) values: above 200.    
Yellow is identified with a H (Hue) value in between 10 and 40 and a S (Saturation) value above 100.  
  
![alt text][image2]  
  
**2. Grayscale**    
Existing function: grayscale  
Then the image is converted to grayscale which is more suitable for edges detection.  
  
![alt text][image3]  
  
**3. Gaussian Blur**   
Existing function: gaussian_blur  
Before performing edges detection, Gaussian Blur filter is applied with a kernel size of 15. 
Note that lower kernel sizes values are more CPU friendly.  
* kernel-size: in the Gaussian Filter will remove the noise leaving the most distinguishable parts. Must be an odd number (5, 7...)  
  
![alt text][image4]  
  
**4. Canny Edges detection**    
Existing function: canny  
The Canny algorithm detects edges by looking at gradients: corresponding to pixel intensities changes.  
Lower threshold used is 50 while higher threshold is 150.  
We have a ratio of 1:3 (recommendation from J. Canny is a ratio between 1:2 and 1:3).  
  
![alt text][image5]  
  
**5. Region Of Interest**    
Existing function: region_of_interest  
A trapezoidal region corresponding to the lower part of the camera is delimited in a generic way.  
By using ratio of image dimension.  
bottom_left  = [cols x 0.1, rows x 0.99]  
top_left     = [cols x 0.4, rows x 0.6]  
bottom_right = [cols x 0.9, rows x 0.99]  
top_right    = [cols x 0.6, rows x 0.6]   
  
![alt text][image6]  
  
**6. Hough Lines detection**   
Existing function: hough_lines  
Probabilistic Hough Line detection is being used (cv2.HoughLineP) with the folowing parameters:  
rho=1, theta=np.pi/180, threshold=20, min_line_len=50, max_line_gap=300  
  
* min_line_len: is the minimum length of a line (in pixels) that we will accept in the output.  
* max_line_gap: is the maximum distance(in pixels) between segments that will be allowed to connect into a single line.  
* Increasing min_line_len and max_line_gap (~100 and above) for Hough Transform will make lines longer and will have less number of breaks. This will make the solid annotated line longer in the output.  
* Increasing max_line_gap will allow points that are farther away from each other to be connected with a single line.  
* threshold: increasing(~50-60) aims to rule out spurious lines. It defines the minimum number of intersections in a given grid cell that are required to choose a line.  
* rho: value of 2 or 1 is recommended. It gives distance resolution in pixels of the Hough grid.  

  
**7. Left and Right line detection**    
Existing function: draw_lines (with most of the modifications being done here)  
Thanks to houghLineP, opencv typically detects a few hundreds of lines more or less relevant to our purpose.  
Our objective is to determine a best fit for the left and right lanes.  
A single slope, intercept and length value is determined for left (negative slope) and right (positive slope) lanes.  
The key characteristics used are:  
* a weighted average is computed for slope and intercept: depending on individual lines length. So that bigger lines contribute more to our slope and intercept values.  
* a simple outliers rejection criteria is defined: a lane has to intersect with the bottom part of the image.  
* a 1st order low pass filtering is applied for the values derived for the current frame (slope, intercept and line length):  
  * y[n] = alpha x[n] + (1 - alpha) y[n-1] with a default alpha value of 0.2  
  
Thanks to these 3 features the estimation of the lanes appears to be pretty accurate and stable on the images and videos provided for this project.  
  
**8. Weighted image construction**    
The estimated left and right lines are superimposed on the original image.  
  
![alt text][image7]  
  
And here is the result on the extra challenge:  
  
[extra.mp4](https://youtu.be/zExBAaFdJjQ)  
    
###2. Identify potential shortcomings with your current pipeline


The main shortcoming is that it is strictly dealing with lines. There is no estimation of curvature.  
Moreover, more conditions should be tested: by night, under rain, with roads going up and down ...  
Also when the car is moving from a lane line to another.  
Or when other cars just in front of us are moving from one lane line to the other.


###3. Suggest possible improvements to your pipeline

The way I implemented the 1st order low pass filtering could be cleaner: I am using global variable. It should be implement in an Object Oriented way with encapsulated variables.  

In terms of additional techniques:
* curvature will be dealt by P4 advanced lane lines detection
* More advanced filtering like Kalman filtering could be used
* More advanced outliers rejection algorithms could be used: maybe like RANSAC algorithm
* The ROI could be adapted depending on road orientation: car going up or down
* One keypoint is related to realtime optimizations. This has not been investigated here (for example using smaller kernel sizes would diminish processing requirements)  
* [Robust and real time detection of curvy lanes for driving assistance and autonomous vehicles](http://airccj.org/CSCP/vol5/csit53211.pdf) 

