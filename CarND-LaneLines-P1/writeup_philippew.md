#**Finding Lane Lines on the Road** 

---

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

###1. Pipeline description.

My pipeline consisted of following steps steps:  
  
**1. Color selection** 
New Function: select_white_yellow  
White and Yellow pixels are extracted in HLS space: cf https://en.wikipedia.org/wiki/HSL_and_HSV  
White is identified with high L (Light) values: above 200.    
Yellow is identified with a H (Hue) value in between 10 and 40 and a S (Saturation) value above 100.  
  
**2. Grayscale**  
Existing function: grayscale  
Then the image is converted to grayscale which is more suitable for edges detection.  
  
**3. Gaussian Blur** 
Existing function: gaussian_blur  
Before performing edges detection, Gaussian Blur filter is applied with a kernel size of 15. 
Note that lower kernel sizes values are more CPU friendly.  
  
**4. Canny Edges detection**  
Existing function: canny  
Lower threshold used is 50 while higher threshold is 150.  
We have a ratio of 1:3 (recommendation from J. Canny is a ratio between 1:2 and 1:3).  
  
**5. Region Of Interest**  
Existing function: region_of_interest  
A trapezoidal region corresponding to the lower part of the camera is delimited in a generic way.  
By using ratio of image dimension.  
bottom_left  = [cols*0.1, rows*0.99]  
top_left     = [cols*0.4, rows*0.6]  
bottom_right = [cols*0.9, rows*0.99]  
top_right    = [cols*0.6, rows*0.6]   
  
**6. Hough Lines detection** 
Existing function: hough_lines  
Probabilistic Hough Line detection is being used (cv2.HoughLineP) with the folowing parameters:  
rho=1, theta=np.pi/180, threshold=20, min_line_len=50, max_line_gap=300  
  
**7. Left and Right line detection**  
Existing function: draw_lines (with most of the modification being done here)  
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


If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


###2. Identify potential shortcomings with your current pipeline


The main shortcoming is that it is strictly dealing with lines. There is no estimation of curvature.  


###3. Suggest possible improvements to your pipeline

The way I implemented the 1st order low pass filtering could be cleaner: I am using global variable. It should be implement in an Object Oriented way.
