# Lane-Detection---Curve-Estimation-Method

## The following the steps of image processing and analysis involved in 'lane.py',
1. Apply a perspective transform to rectify binary image ("birds-eye view").
2. Use binary transforms., to create a thresholded binary image.
3. Apply median blur filter for removing the thresholded image noise.
4. Split the image for the left and right lane.
5. Determine the center, top and bottom points of the left and right lanes.

![3-Point Estimation](/images/1.png)

6. Fit a second degree polynomial curve through the points.
7. Determine the curvature of the lane and vehicle position with respect to center.
8. Project the detected lane boundaries back onto the original image.
9. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle offset position.

![](/images/2.png)

10. The left lane curve coefficients, right lane coefficients and  the offset from the center of the lane are determined.  

![](/images/3.png)

## Required Python Libraries
1. imutils
2. cv2 [opencv]
3. numpy

## Code Execution 
`<cd Lane-Detection---Curve-Estimation-Method>` 

`<python lane-detection.py>` 
