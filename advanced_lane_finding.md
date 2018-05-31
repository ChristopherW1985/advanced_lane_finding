
# Project 4: Advanced Lane Finding

This project aims to develop a pipeline which allows for the detection lane lines in a video. In contrast to the similar project _Finding Lane Lines on the Road_ this project makes use of a couple of new methods which allow to detect lane lines more robustly. Our pipeline consists of the following steps
* camera calibration
* color and gradient thresholds for lane line selection
* perspective transformation
* lane line fitting
* inverse transformation

Each of these steps is discussed in detail in the next sections.

## 1. Camera Calibration

Modern cameras suffer from distortion effects i.e. objects appear to have a different sizes and shapes than they actual have. Fortunatley cameras can be easily calibrated. Calibration removes most of the distortion effects. 

For a reliable calibration a set of approximately 20 images of the same object is needed. This object needs to have a number of well defined points with known relative distances. A good choice is a chess board. Its corner points can be easily detected automatically and the distance between them is always the same.  

Using the functionality of opencv one can compute distortion coefficients and the camera matrix based the location of these corner points.


```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
%matplotlib inline

def display_images(images, figsize = (15, 10), title=None, fontsize=20):
    "Display images in two columns. Choose gray-scale colorbar if image is two-dimensional."
    ncols = 2
    nrows = np.ceil(len(images)/2)
    
    fig = plt.figure(figsize = figsize)
    for i, image in enumerate(images, start = 1):
        ax = fig.add_subplot(nrows, ncols, i)
        if title is not None:
            ax.set_title(title[i-1], fontsize=fontsize)
        plt.imshow(image) if len(image.shape) == 3 else plt.imshow(image, cmap='gray')
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()
    
# load images for camera calibration
images_cal = [mpimg.imread(image) for image in glob.glob("camera_cal/*.jpg")]

display_images(images_cal[:2])
```


![png](./images/output_4_0.png)



```python
def calibration_points(images_cal, n_rows, n_cols):
    """Determine object and image points for camera calibration.
    
    Assumes that images_cal is a list of images of a chessboard taken with the same 
    camera but from different perspectives. Algorithm detects corners and uses them as 
    images points.
    
    Args:
        images_cal -- Images of a chessboard
        n_rows     -- number of rows
        n_cols     -- number of columns
    """
    # prepare object points: (0,0,0), (1,0,0), ...
    objp = np.zeros((n_rows*n_cols,3), np.float32)
    objp[:,:2] = np.mgrid[0:n_rows,0:n_cols].T.reshape(-1,2) # python trick
    objpoints = [] # 3D real-world points    
    imgpoints = [] # 2D image points
    images_corner = [] 
    # iterate over images and determine object and image points 
    for image in images_cal:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (n_rows, n_cols), None)
        # if pattern was successfully found in image add object and image points
        if ret:
            objpoints.append(objp)
            # refine corners (optional)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            # create image with found corners drawn onto it for assessing the quality of the method
            img = np.copy(image)       
            images_corner.append(cv2.drawChessboardCorners(img, (n_rows,n_cols), corners, ret))
    return objpoints, imgpoints, images_corner   
    
    
# determine object and image points
objpoints, imgpoints, images_corner = calibration_points(images_cal, 9, 6)

# check if corners where detected correctly
display_images(images_corner[:2])
```


![png](./images/output_5_0.png)



```python
def calibrate_camera(objpoints, imgpoints, shape):
    """Calibrate camera using object and image points and return camera matrix and distortion coefficients."""
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
    return mtx, dist

def undistort(img, mtx, dist):
    """Undistort image.
    
    Args:
        img  -- distorted image
        mtx  -- camera matrix
        dist -- distortion coefficients
    """
    return cv2.undistort(img, mtx, dist, None, mtx)


height, width, _ = images_cal[0].shape
mtx, dist = calibrate_camera(objpoints, imgpoints, (width, height))
```


```python
images_cal_undis = [undistort(image, mtx, dist) for image in images_cal]

display_images([images_cal[6], images_cal_undis[6]], title=['distorted', 'undistorted'], fontsize=20) 
```


![png](./images/output_7_0.png)



```python
# read test images
images_orig = [mpimg.imread(image) for image in glob.glob("test_images/*.jpg")]
   
images_undis = [undistort(image, mtx, dist) for image in images_orig]

display_images([images_orig[0], images_undis[0]], title=['distorted', 'undistorted'], fontsize=20) 
```


![png](./images/output_8_0.png)


## 2. Color and Gradient Threshold

The original images contain many objects and structures which are not related to lane lines. For conducting a successful fit of a polynom to potential lane line pixels one has to suppress unrelated pixels as much as possible. The course presented various methods how this can be achieved e.g. through thresholds on properties of the gradient. We don't use properties of the gradient and restrict ourselves to a simple color and region selection. 

Since  lane lines are located in approximately the same region of the image one can choose a polygon of fixed dimensions to select the region of interest. Lane lines are typically either yellow or white. We transform the image from RGB colorspace to HLS colorspace and select white and yellow colors. This filter is more robust than conducting the selection process directly in RGB colorspace. 


```python
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh = (0, 255)):
    """Zero all pixels with a gradient smaller than a given threshold.
    
    Args:
        img          -- original image
        orient       -- direction of the gradient
        sobel_kernel -- size of the kernel
        thresh       -- lower and upper threshold
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient=='x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)   
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)         
    magn = np.absolute(sobel)
    magn = np.uint(255*magn/np.max(magn))
    binary_output = np.zeros_like(magn, dtype=np.uint8)
    binary_output[(magn > thresh[0]) & (magn < thresh[1])] = 1   
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(50, 255)):   
    """Zero all pixels with a magnitude of the gradient smaller than a given threshold.
    
    Args:
        img          -- original image
        sobel_kernel -- size of the kernel
        thresh       -- lower and upper threshold
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)   
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)         
    magn = np.sqrt(sobelx**2 + sobely**2)    
    magn = np.uint(255*magn/np.max(magn))
    binary_output = np.zeros_like(magn, dtype=np.uint8)
    binary_output[(magn > mag_thresh[0]) & (magn < mag_thresh[1])] = 1
    return binary_output   

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """Zero all pixels with direction of the gradient not in range defined by thresh.
    
    Args:
        img          -- original image
        sobel_kernel -- size of the kernel
        thresh       -- lower and upper threshold of direction (in radiants)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx)) 
    binary_output = np.zeros_like(absgraddir, dtype=np.uint8)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1    
    return binary_output    

def region_of_interest(img):
    """Zero all pixels expect pixels in region defined by vertices."""
    mask = np.zeros_like(img)   
    if len(img.shape) > 2:
        ignore_mask_color = (255,) * img.shape[2] 
    else:
        ignore_mask_color = 255
    bottom_left = (0.10, 0.97)
    top_left = (0.41, 0.64)
    top_right = (0.59, 0.65)
    bottom_right = (0.96, 0.96)
    xdim, ydim  = img.shape
    bl_y, bl_x = bottom_left
    tl_y, tl_x = top_left
    tr_y, tr_x = top_right
    br_y, br_x = bottom_right
    v1 = int(bl_y*ydim), int(bl_x*xdim)
    v2 = int(tl_y*ydim), int(tl_x*xdim)
    v3 = int(tr_y*ydim), int(tr_x*xdim)
    v4 = int(br_y*ydim), int(br_x*xdim)
    vertices = np.array([[v1, v2, v3, v4]])   
    # set pixel intensities to ignore_mask_color inside region defined by vertices
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # zero all pixel outside of region defined by vertices
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def select_white_yellow(image):
    """Select white and yellow lane lines."""
    converted = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # white color mask
    #lower = np.uint8([200, 200, 200])
    #upper = np.uint8([255, 255, 255])
    #white_mask_1 = cv2.inRange(image, lower, upper)
    lower = np.uint8([0,   200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask_2 = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 60, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    # combine  masks
    mask = cv2.bitwise_or(white_mask_2, yellow_mask)
    color_binary = np.zeros_like(mask)
    color_binary[mask==255] = 1
    return color_binary

def threshold(img):
    """Apply color and region selection to img and return a binary image."""
    color_binary = select_white_yellow(img)
    return region_of_interest(color_binary)

```


```python
images_thresh = [threshold(image) for image in images_undis]

display_images([images_undis[3], images_thresh[3]], title=['undistorted', 'binary'], fontsize=20)
```


![png](./images/output_12_0.png)


## 3. Perspective Transformation

In order to facilitate a reliable fit of a polynom to lane line pixels we need to perform a perspective transformation. This is also requiered if we want to determine the curvature of the lane lines. The most appropriate perspective is the birds-eye perspective. In this perspective straight lane lines should also appear straight in the image.

The transformation matrix can be inferred from a set of four points in the original image (source points) and a set of four points in the transformed image (destination points). The transformation matrix maps the four source points to the four destination points. 

We need one original image with straight lane lines and choose the four source points such that they are directly on top of the lane line markings. The destination points have to form a rectangle.


```python
# read images with straight lane lines 
images_straight = [mpimg.imread(image) for image in glob.glob("test_images/straight_lines*.jpg")]

# undistort images of straight lane lines
images_undis_straight = [undistort(image, mtx, dist) for image in images_straight]

# draw image with rectangle
img_poly = np.copy(images_undis_straight[0])
src = np.array([[1120, 720], [189, 720], [563, 470], [720, 470]], dtype=np.int32)
src = src.reshape((-1,1,2))
cv2.polylines(img_poly, [src], True, (255,0,0), thickness=2)

fig = plt.figure(figsize = (25, 20))
plt.imshow(img_poly);
```


![png](./images/output_15_0.png)



```python
 def unwarp_matrix():
    """Return transformation matrix for birds-eye view and its inverse."""
    src = np.array([[1110, 720],  # bottom right 
                    [189, 720],   # bottom left
                    [563, 470],   # top left
                    [720, 470]],  # top right 
                    dtype=np.float32)
    dst = np.array([[920, 720], [320,720], [320, 1], [920, 1]], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv    
        
def change_persepctive(img, M):
    """Apply transformation M to image img for changing perspective."""
    (height, width) = img.shape[:2]
    dist = cv2.warpPerspective(img, M, (width, height))
    return dist


M, Minv = unwarp_matrix()
```


```python
img_size = (img_poly[0][1], img_poly[0][0])
images_straight_unwarp = [change_persepctive(image, M) for image in (img_poly, images_undis_straight[1])]

display_images(images_undis_straight, title=['original', 'original'])
display_images(images_straight_unwarp, title=['birds-eye', 'birds-eye'])
```


![png](./images/output_17_0.png)



![png](./images/output_17_1.png)



```python
images_unwarp = [change_persepctive(image, M) for image in images_thresh]

display_images([images_undis[2], images_unwarp[2]], title=['original', 'birds-eye'])
```


![png](./images/output_18_0.png)


## 4. Fitting lane lines

In the next step we have to decide which pixels of the binary image are actual lane line pixel. The algorithm we employ makes use of the observation that lane lines should have a higher intensity than background object. If we create a histogram of the entire image along the x-axis the two lane lines are assumed to have the highest peaks. The location of of these peaks give a first indication of the location of the actual lane lines.  


```python
fig = plt.figure(figsize = (10, 5))
histogram = np.sum(images_unwarp[0], axis=0)
plt.plot(histogram);
```


![png](./images/output_21_0.png)


The algorithm to decide which pixels are lane line pixels has the following steps:
* divide image along x-axis in two regions (for left and right lane line)
* for each region create histogram with appropriate binning and assume location of peaks correspond approximately to location of lane lines
* divide image along y-axis in nine regions
* for first region choose a 200 pixel wide window centered at these approximate lane line locations
* all pixels in both windows are assumed to be lane line pixels
* compute more precise location of lane lines by computing the mean position of all pixels within each window
* use this updated lane line location as starting point for the next one of the nine windows
* iterate over all nine windows repeating the previous steps
* updating the approximate lane line location is only performed if the 200 pixel wide window contains at least 50 non-zero pixels 
    
After determining the lane line pixels they are fitted with a second order polynom of the form
$$lane(y) = a\cdot y^2 + b\cdot y + c.$$
The polynom is written in dependence of $y$ due to the small changes in $x$.


```python
def fit_lane_lines(binary_unwarped):
    """Fit lane line pixels with a second order polynom and return result.
    
    Return None if fit failed.
    
    Args:
        binary_unwarped -- A binary image transformed to birds-eye view 
    Returns:
        left_fit  -- coefficients of polynom fitting left lane line
        right_fit -- coefficients of polynom fitting left lane line
        out_img   -- image where left and right lane line points are highlighted in red and green
                   respectively
    """
    # stacks three binary_unwarped images on top of eachother
    # allows to draw e.g. a colored rectangle
    out_img = np.dstack((binary_unwarped, binary_unwarped, binary_unwarped))

    # find midpoint between left and right halves of image
    midpoint = binary_unwarped.shape[1]//2

    # midpoints and margins of the regions where we look for lane line pixels
    leftx_base = None
    rightx_base = None
    margin = 100 

    # number of windows 
    nwindows = 9
    # window_height (in y direction), note that window_width = 2*margin)
    window_height = binary_unwarped.shape[0]//nwindows

    # identify the x and y indices of all non-zero pixels in the whole image
    # python trick: why interchange x and y?
    nonzerox = binary_unwarped.nonzero()[1]
    nonzeroy = binary_unwarped.nonzero()[0]
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        # y coordinates of the sliding window
        win_y_high = binary_unwarped.shape[0] - window*window_height
        win_y_low = win_y_high - window_height

        # in case of first window: search for maximum peaks in right and left halves of the image
        # and use these points as center points of each window
        if window==0:
            histogram = np.sum(binary_unwarped, axis=0)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = midpoint + np.argmax(histogram[midpoint:])
                
        # x coordinates of the left and right window 
        win_xleft_low = leftx_base - margin
        win_xleft_high = leftx_base + margin
        win_xright_low = rightx_base - margin
        win_xright_high = rightx_base + margin    
    
        # draw window regions on image for visualization
        #left lane line
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        # right lane line
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
    
        # all points within in the windows are considered as lane line pixels:
        # get the x, y-indices of these pixels (note that e.g. the first entry of nonzerox and nonzeroy
        # refer to the same point, false is interpreted as 0)
        # left lane line
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        # right lane line
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]    
    
        # append indices to lists (python trick)
        # note that these indices refer to the index of the lane line pixels
        # in nonzerox and nonzeroy arrays
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
    
        # udate leftx_base and rightx_base for next window
        minpix = 50
        if len(good_left_inds) > minpix:
            leftx_base = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_base = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    if (leftx.size==0) or (lefty.size==0) or (rightx.size==0) or (righty.size==0):
       return None
    
    # Fit a second order polynomial to each line
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # draw imageout_img
    ploty = np.linspace(0, binary_unwarped.shape[0]-1, binary_unwarped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]   
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    return left_fit, right_fit, out_img


def points_for_drawing(image, left_fit, right_fit):
    """Generate lane line points for plotting.
    
    Args:
        image     -- image to draw lane lines on 
        left_fit  -- coefficients of polynom fitting left lane line
        right_fit --
    Returns:
        left_fitx  -- x coordinates of points of left lane line (for plotting)
        right_fitx -- x coordinates of points of right lane line (for plotting)
        ploty      -- y coordinates of lane line pixls (for plotting)    
    """
    ydim = image.shape[0]
    ploty = np.linspace(0, ydim-1, ydim)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2] 
    return left_fitx, right_fitx, ploty


fit_results = [fit_lane_lines(image) for image in images_unwarp]
```


```python
left_fit, right_fit, out_img = fit_results[2]
left_fitx, right_fitx, ploty = points_for_drawing(out_img, left_fit, right_fit)

    
fig = plt.figure(figsize = (17, 10))
ax = fig.add_subplot(1, 2, 1)
ax.set_title("original", fontsize=15)
plt.imshow(images_undis[2])
ax = fig.add_subplot(1, 2, 2)
ax.set_title("lane lines", fontsize=15)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.title("lane lines", fontsize=20)
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.imshow(out_img);
```


![png](./images/output_24_0.png)


## 5. Inverse Transform

After determining the coefficients of the second order polynom describing the lane lines we have to draw it on the original image. First we draw the lane lines on a blank image. At this point the perspective is still in *birds-eye view*. Afterward we apply the inverse perspective transformation to transform the perspective of the image back to the one of the original image. Then we are able to combine the resulting image with the original image. Instead of highlighting the lane lines in red as in the first project we use a green area to highlight the area where the road is located.


```python
def draw_lines(undist, unwarped, Minv, left_fitx, right_fitx, ploty):
    """Draw lane lines on original image.
    
    Args:
        undist     -- original image
        unwarped   -- binary warped image
        Minv       -- transform matrix to transform birds-eye view to original image
        left_fitx  -- x-coordinates of left lane line
        right_fitx -- x-coordinates of right lane line
        ploty      -- y-coordinates of lane line
    Return:
        Original image with lane lines drawn on.
    """
    
    # create mage to draw lines on
    unwarped = np.zeros_like(unwarped).astype(np.uint8)
    color_unwarp = np.dstack((unwarped, unwarped, unwarped))

    # transform x and y points to appropriate format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # draw the lane on the blank image
    cv2.fillPoly(color_unwarp, np.int_([pts]), (0, 255, 0))
    
    # Warp the blank back to original image space using Minv
    newwarp = cv2.warpPerspective(color_unwarp, Minv, (undist.shape[1], undist.shape[0])) 
    # combine new image with original one
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result
```


```python
final_image = []
for idx in range(0, len(fit_results)):
    left_fit, right_fit, out_img  = fit_results[idx]
    left_fitx, right_fitx, ploty = points_for_drawing(out_img, left_fit, right_fit)
    undis = images_undis[idx]
    unwarped = images_unwarp[idx]
    final_image.append(draw_lines(undis, unwarped, Minv, left_fitx, right_fitx, ploty))
    
display_images(final_image)
```


![png](./images/output_28_0.png)


## 6. Radius of curvature and center offset

For determining an appropriate steering angle the car needs to know the offset from the center of the lane line and the curvature of the lane line. These two quantities can be determined in a straightforward way from the fit results. The radius of the curvature is given as follows:

$$R_{curv}(y) = \frac{(1 + (2Ay + B)^2)^{3/2}}{|2A|}.$$

$A$ and $B$ are the second and first order coefficients of the polynom respectively. The offset of the center of the car from the center of the lane line can be computed with

$$d_{off} = x_{left} + \frac{x_{left} - x_{right}}{2} - x_{center}.$$

$x_{center}$ is the center of the car (assumed to be the center of the image i.e. the camera is mounted in the center of the car). $x_{left} + \frac{x_{left} - x_{right}}{2}$ is the center of the lane line. $d_{off}$ is negative if the car is too far on the right with respect to the center of the lane line and positive otherwise.


```python
def radius_of_curvature(fit_res, y_eval):
    """Computes radius of curvature in meters.
    
    Args:
        fit_res -- coefficients of polynom
        y_eval  -- y-position at which curvature is evaluated
    """
    ym_per_pix = 30/720 
    xm_per_pix = 3.7/700
    A = fit_res[0]*xm_per_pix/(ym_per_pix**2)
    B = fit_res[1]*xm_per_pix/ym_per_pix  
    return round(((1 + (2*A*y_eval*ym_per_pix + B)**2)**1.5) / np.absolute(2*A), 3)

def offset_center(img, left_fit, right_fit):
    """Computes offset from center of lane (center of car = center of image).
    
    Positive: car is on the left of lane center
    Negative: car is on the right of lane center
    """
    xm_per_pix = 3.7/700
    height, width = img.shape[:2]
    x_center = width // 2
    left_fitx = left_fit[0]*height**2 + left_fit[1]*height + left_fit[2]
    right_fitx = right_fit[0]*height**2 + right_fit[1]*height + right_fit[2] 
    center_lane = (right_fitx - left_fitx) / 2 + left_fitx
    return round((center_lane - x_center)*xm_per_pix, 2)
    

left_fit, right_fit, out_img = fit_results[0]
print("left lane curvature: ", radius_of_curvature(left_fit,  720), "m")
print("right lane curvature:", radius_of_curvature(right_fit, 720), "m")
print("offset from lane center:", offset_center(out_img, left_fit, right_fit), "m")
```

    left lane curvature:  1495.728 m
    right lane curvature: 1385.127 m
    offset from lane center: 0.29 m


## 7. Full Pipeline

In order to process a video (i.e. a stream of images) we implement a class which executes the full pipline
* undistort camera image
* apply thresholds to create a binary image where pixels which are not part of the lane line are suppressed
* change perspective to birds-eye view
* fit lane lines with a second order polynom
* check if fit was successful and is not an outlier
* draw lane lines on blank image
* transform  perspective of image with lane line to original perspective and combine result with original image
* add curvature and offset information to image.

In addition we implemented an averaging of detected lane lines to prevent it from jumping around from image to image. Our algorithm considers the last nine successfully fitted lane line and averages them with the fit result of the current image. The averaging is done for each of the three coefficients of the second order polynom (e.g. all second order coefficients are averaged). This procedure helps to reduce the jumping of lane lines significantly. It is not advised to consider more than previous ten successful fits for averaging because the algorithms wouldn't be able to follow changes in lane line curvature appropriately. 

Sometimes no valid fit was possible. In this case the corresponding image is disgarded and the averaged lane lines of the previous ten successful fits are plotted. These lane lines are identical to the one plotted on the previous video frame.

In other cases a valid fit was possible but the result is not meaningful. This can happen if besides the actual lane lines background pixels are present in the binary image. The algorithms tries to fit these pixels as well, which might result a unreasonable lines which are strongly curved. This would influence not only the current image but also at least the ten next frames. The best way to prevent this effect is to analyse the root cause and improve the *threshold*-function such that pixels that are not part of the lane line are suppressed sufficiently. A complementary approach is to conduct an outlier detection to filter fits that exhibit unreasonable coefficients. In our implementation each fit result is checked for unreasonable large coefficients of the qudratic term of the polynom (which determines the curvature of the line). In addition we check if the right and left lane line are curved in the same direction and if the curvature doesn't deviate too much from the average of the previous ten successful fits. The latter two points are only considered if the lane line is not straight.

If the current fit is an outlier it is disgarded and the average of the previous fits is plotted on the current image (same treatment as for the not successful fit). 


```python
import collections

class FindLaneLines:
    def __init__(self, mtx, dist, M, Minv):
        # container to save previously determined lane lines
        self.memory = 10
        self.left = collections.deque(maxlen=self.memory)
        self.right = collections.deque(maxlen=self.memory)
        # average
        self.left_avg = None
        self.right_avg = None
        # matrices for perspective transformation, distortion coefficients and undistort matrix
        self.mtx = mtx
        self.dist = dist
        self.M = M
        self.Minv = Minv
        
    def process_image(self, image):
        """Process image with pipeline and return image with detected lane lines drawn on it.""" 
        image_orig = np.copy(image)
        # undistort image
        undist = undistort(image_orig, self.mtx, self.dist) 
        # apply thresholds
        thresh = threshold(undist)
        # warp to birds-eye view
        warped = change_persepctive(thresh, self.M)
        # fit lane lines
        result = fit_lane_lines(warped)
        
        # if fit not successful return previous lane line
        if result is None:
            left_fitx, right_fitx, ploty = points_for_drawing(image, self.left_avg, self.right_avg)
            image = draw_lines(undist, warped, self.Minv, left_fitx, right_fitx, ploty)
            return image   
        left_fit, right_fit, out_img = result
         
        def outlier_radius(curr_left, avg_left, curr_right, avg_right):
            """Check if current lane line is outlier."""
            # in case image is first frame of video
            if avg_left is None:
                return True
            # filter out lines with strong curvature
            cond1 = np.abs(curr_left[0]) < 0.001
            cond2 = np.abs(curr_right[0]) < 0.001
            # do not apply filter if lines are straight
            cond3 = True
            if np.abs(curr_left[0])>0.0002 or np.abs(curr_right[0])>0.0002:   
                # lane lines should be curved in opposite directions
                cond4 = np.sign(curr_left[0]*curr_right[0])==1
                # current lane lines should not deviate too much from average of previous lane lines 
                cond5 = np.abs(avg_left[0]-curr_left[0]) < 2*curr_left[0]
                cond6 = np.abs(avg_left[0]-curr_left[0]) < 2*curr_left[0]
                cond3 = (not cond4) and (not cond5) and (not cond6)            
            return cond1 and cond2 and cond3
                        
        # if lane line of current image is no outlier update left_avg and right_avg values
        # otherwise return previous lane line
        no_outlier = outlier_radius(left_fit, self.left_avg, right_fit, self.right_avg)
        if no_outlier: 
            self.left.append(left_fit)
            self.right.append(right_fit)
            # compute new average (smoothing)
            self.left_avg = np.mean(self.left, axis=0)
            self.right_avg = np.mean(self.right, axis=0)
           
        # draw lane lines on image
        left_fitx, right_fitx, ploty = points_for_drawing(image, self.left_avg, self.right_avg)
        image = draw_lines(undist, warped, self.Minv, left_fitx, right_fitx, ploty)
        
        # add text to image about center offset and curvature to image
        r_avg_left = radius_of_curvature(self.left_avg, np.max(ploty)) 
        r_avg_right = radius_of_curvature(self.right_avg, np.max(ploty)) 
        offset = offset_center(img, left_fit, right_fit)
        cv2.putText(image, 'curvature (left) : ' + str(round(r_avg_left, 2)) + 'm',  (80, 90),  cv2.FONT_HERSHEY_SIMPLEX, 
                    1.3, (255,255,255), thickness=2)
        cv2.putText(image, 'curvature (right): ' + str(round(r_avg_right, 2)) + 'm', (80, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.3, (255,255,255), thickness=2) 
        cv2.putText(image, 'offset: ' + str(offset) + 'm', (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.3, (255,255,255), thickness=2)  
        return image
    
```

Let's test the full pipline on the six test images.


```python
pic = []
for img in images_orig:
    detector = FindLaneLines(mtx, dist, M, Minv)
    pic.append(detector.process_image(img))
    
display_images(pic, figsize = (15, 15))
```


![png](./images/output_36_0.png)


## 8. Video stream

As a last step we have to apply the full pipeline to the video stream. The video is processed in a couple of minutes, although we don't make use of the more efficient method described in the course to find lane lines (based on the fit results of the previous frame to guide the search in the current frame). 


```python
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import os


def process_video(video_input, video_output):
    detector = FindLaneLines(mtx, dist, M, Minv)

    clip = VideoFileClip(os.path.join('test_videos', video_input))
    processed = clip.fl_image(detector.process_image)
    processed.write_videofile(os.path.join('test_videos_output', video_output), audio=False)
```


```python
%time process_video('project_video.mp4', 'project_video.mp4') 
```

    [MoviePy] >>>> Building video test_videos_output/project_video.mp4
    [MoviePy] Writing video test_videos_output/project_video.mp4


    
      0%|          | 0/1261 [00:00<?, ?it/s][A
      0%|          | 2/1261 [00:00<02:13,  9.42it/s][A
      0%|          | 3/1261 [00:00<02:17,  9.17it/s][A
      0%|          | 4/1261 [00:00<02:36,  8.03it/s][A
      0%|          | 5/1261 [00:00<02:44,  7.63it/s][A
      0%|          | 6/1261 [00:00<02:52,  7.27it/s][A
      1%|          | 7/1261 [00:00<02:50,  7.37it/s][A
      1%|          | 8/1261 [00:01<02:51,  7.30it/s][A
      1%|          | 9/1261 [00:01<02:53,  7.20it/s][A
      1%|          | 10/1261 [00:01<02:48,  7.43it/s][A
      1%|          | 11/1261 [00:01<02:43,  7.67it/s][A
      1%|          | 12/1261 [00:01<02:40,  7.80it/s][A
      1%|          | 13/1261 [00:01<02:50,  7.31it/s][A
      1%|          | 14/1261 [00:01<03:14,  6.43it/s][A
      1%|          | 15/1261 [00:02<03:10,  6.54it/s][A
      1%|▏         | 16/1261 [00:02<03:02,  6.83it/s][A
      1%|▏         | 17/1261 [00:02<03:02,  6.81it/s][A
      1%|▏         | 18/1261 [00:02<03:04,  6.73it/s][A
      2%|▏         | 19/1261 [00:02<03:06,  6.66it/s][A
      2%|▏         | 20/1261 [00:02<03:05,  6.69it/s][A
      2%|▏         | 21/1261 [00:02<02:54,  7.10it/s][A
      2%|▏         | 22/1261 [00:03<03:38,  5.67it/s][A
      2%|▏         | 23/1261 [00:03<04:27,  4.63it/s][A
      2%|▏         | 24/1261 [00:03<04:06,  5.02it/s][A
      2%|▏         | 25/1261 [00:03<04:13,  4.88it/s][A
      2%|▏         | 26/1261 [00:04<04:08,  4.98it/s][A
      2%|▏         | 27/1261 [00:04<03:42,  5.54it/s][A
      2%|▏         | 28/1261 [00:04<03:16,  6.28it/s][A
      2%|▏         | 29/1261 [00:04<03:36,  5.70it/s][A
      2%|▏         | 30/1261 [00:04<03:44,  5.48it/s][A
      2%|▏         | 31/1261 [00:04<03:25,  5.98it/s][A
      3%|▎         | 32/1261 [00:04<03:11,  6.41it/s][A
      3%|▎         | 33/1261 [00:05<03:00,  6.82it/s][A
      3%|▎         | 34/1261 [00:05<02:51,  7.15it/s][A
      3%|▎         | 35/1261 [00:05<02:44,  7.43it/s][A
      3%|▎         | 36/1261 [00:05<02:58,  6.86it/s][A
      3%|▎         | 37/1261 [00:05<02:51,  7.16it/s][A
      3%|▎         | 38/1261 [00:05<02:49,  7.21it/s][A
      3%|▎         | 39/1261 [00:05<02:47,  7.31it/s][A
      3%|▎         | 40/1261 [00:06<02:35,  7.84it/s][A
      3%|▎         | 42/1261 [00:06<02:26,  8.32it/s][A
      3%|▎         | 43/1261 [00:06<02:28,  8.23it/s][A
      3%|▎         | 44/1261 [00:06<02:34,  7.87it/s][A
      4%|▎         | 45/1261 [00:06<03:33,  5.69it/s][A
      4%|▎         | 46/1261 [00:06<03:14,  6.24it/s][A
      4%|▎         | 47/1261 [00:07<03:02,  6.64it/s][A
      4%|▍         | 48/1261 [00:07<03:10,  6.37it/s][A
      4%|▍         | 49/1261 [00:07<03:07,  6.48it/s][A
      4%|▍         | 50/1261 [00:07<02:54,  6.95it/s][A
      4%|▍         | 51/1261 [00:07<02:57,  6.83it/s][A
      4%|▍         | 52/1261 [00:07<03:12,  6.28it/s][A
      4%|▍         | 53/1261 [00:07<03:10,  6.33it/s][A
      4%|▍         | 54/1261 [00:08<04:11,  4.79it/s][A
      4%|▍         | 55/1261 [00:08<04:26,  4.52it/s][A
      4%|▍         | 56/1261 [00:08<04:08,  4.85it/s][A
      5%|▍         | 57/1261 [00:08<04:05,  4.91it/s][A
      5%|▍         | 58/1261 [00:09<04:07,  4.86it/s][A
      5%|▍         | 59/1261 [00:09<04:23,  4.57it/s][A
      5%|▍         | 60/1261 [00:09<03:50,  5.21it/s][A
      5%|▍         | 61/1261 [00:09<04:06,  4.87it/s][A
      5%|▍         | 62/1261 [00:09<04:13,  4.72it/s][A
      5%|▍         | 63/1261 [00:10<04:06,  4.86it/s][A
      5%|▌         | 64/1261 [00:10<03:47,  5.25it/s][A
      5%|▌         | 65/1261 [00:10<04:14,  4.70it/s][A
      5%|▌         | 66/1261 [00:10<04:07,  4.82it/s][A
      5%|▌         | 67/1261 [00:11<04:53,  4.07it/s][A
      5%|▌         | 68/1261 [00:11<04:36,  4.32it/s][A
      5%|▌         | 69/1261 [00:11<03:53,  5.10it/s][A
      6%|▌         | 70/1261 [00:11<03:25,  5.81it/s][A
      6%|▌         | 71/1261 [00:11<03:09,  6.29it/s][A
      6%|▌         | 72/1261 [00:11<03:01,  6.53it/s][A
      6%|▌         | 73/1261 [00:11<02:57,  6.70it/s][A
      6%|▌         | 74/1261 [00:12<03:07,  6.33it/s][A
      6%|▌         | 75/1261 [00:12<03:01,  6.53it/s][A
      6%|▌         | 76/1261 [00:12<04:14,  4.65it/s][A
      6%|▌         | 77/1261 [00:12<03:51,  5.11it/s][A
      6%|▌         | 78/1261 [00:13<04:02,  4.87it/s][A
      6%|▋         | 79/1261 [00:13<03:56,  4.99it/s][A
      6%|▋         | 80/1261 [00:13<03:46,  5.22it/s][A
      6%|▋         | 81/1261 [00:13<03:57,  4.98it/s][A
      7%|▋         | 82/1261 [00:13<03:30,  5.61it/s][A
      7%|▋         | 83/1261 [00:13<03:43,  5.26it/s][A
      7%|▋         | 84/1261 [00:14<03:26,  5.71it/s][A
      7%|▋         | 85/1261 [00:14<04:01,  4.86it/s][A
      7%|▋         | 86/1261 [00:14<03:35,  5.45it/s][A
      7%|▋         | 87/1261 [00:14<03:11,  6.14it/s][A
      7%|▋         | 88/1261 [00:14<03:11,  6.13it/s][A
      7%|▋         | 89/1261 [00:14<03:05,  6.30it/s][A
      7%|▋         | 90/1261 [00:15<03:04,  6.33it/s][A
      7%|▋         | 91/1261 [00:15<03:36,  5.40it/s][A
      7%|▋         | 92/1261 [00:15<04:01,  4.84it/s][A
      7%|▋         | 93/1261 [00:15<04:25,  4.39it/s][A
      7%|▋         | 94/1261 [00:16<04:20,  4.47it/s][A
      8%|▊         | 95/1261 [00:16<03:58,  4.89it/s][A
      8%|▊         | 96/1261 [00:16<04:35,  4.23it/s][A
      8%|▊         | 97/1261 [00:16<04:04,  4.77it/s][A
      8%|▊         | 98/1261 [00:16<04:17,  4.51it/s][A
      8%|▊         | 99/1261 [00:17<04:06,  4.71it/s][A
      8%|▊         | 100/1261 [00:17<04:51,  3.98it/s][A
      8%|▊         | 101/1261 [00:17<05:21,  3.60it/s][A
      8%|▊         | 102/1261 [00:17<04:51,  3.98it/s][A
      8%|▊         | 103/1261 [00:18<04:19,  4.46it/s][A
      8%|▊         | 104/1261 [00:18<04:45,  4.06it/s][A
      8%|▊         | 105/1261 [00:18<05:01,  3.83it/s][A
      8%|▊         | 106/1261 [00:18<04:29,  4.28it/s][A
      8%|▊         | 107/1261 [00:19<04:18,  4.46it/s][A
      9%|▊         | 108/1261 [00:19<04:05,  4.70it/s][A
      9%|▊         | 109/1261 [00:19<03:44,  5.14it/s][A
      9%|▊         | 110/1261 [00:19<03:49,  5.02it/s][A
      9%|▉         | 111/1261 [00:19<03:43,  5.14it/s][A
      9%|▉         | 112/1261 [00:19<03:23,  5.64it/s][A
      9%|▉         | 113/1261 [00:20<03:11,  5.99it/s][A
      9%|▉         | 114/1261 [00:20<03:06,  6.15it/s][A
      9%|▉         | 115/1261 [00:20<03:06,  6.14it/s][A
      9%|▉         | 116/1261 [00:20<03:12,  5.95it/s][A
      9%|▉         | 117/1261 [00:20<03:25,  5.55it/s][A
      9%|▉         | 118/1261 [00:21<03:29,  5.45it/s][A
      9%|▉         | 119/1261 [00:21<03:36,  5.27it/s][A
     10%|▉         | 120/1261 [00:21<03:22,  5.64it/s][A
     10%|▉         | 121/1261 [00:21<03:13,  5.89it/s][A
     10%|▉         | 122/1261 [00:21<03:17,  5.76it/s][A
     10%|▉         | 123/1261 [00:21<03:16,  5.80it/s][A
     10%|▉         | 124/1261 [00:22<05:06,  3.71it/s][A
     10%|▉         | 125/1261 [00:22<04:59,  3.79it/s][A
     10%|▉         | 126/1261 [00:22<04:18,  4.40it/s][A
     10%|█         | 127/1261 [00:22<03:55,  4.82it/s][A
     10%|█         | 128/1261 [00:23<03:36,  5.23it/s][A
     10%|█         | 129/1261 [00:23<03:30,  5.39it/s][A
     10%|█         | 130/1261 [00:23<03:14,  5.81it/s][A
     10%|█         | 131/1261 [00:23<03:19,  5.65it/s][A
     10%|█         | 132/1261 [00:23<03:38,  5.17it/s][A
     11%|█         | 133/1261 [00:23<03:24,  5.51it/s][A
     11%|█         | 134/1261 [00:24<03:29,  5.38it/s][A
     11%|█         | 135/1261 [00:24<03:15,  5.75it/s][A
     11%|█         | 136/1261 [00:24<03:06,  6.02it/s][A
     11%|█         | 137/1261 [00:24<03:20,  5.61it/s][A
     11%|█         | 138/1261 [00:24<03:08,  5.97it/s][A
     11%|█         | 139/1261 [00:25<03:37,  5.15it/s][A
     11%|█         | 140/1261 [00:25<03:18,  5.63it/s][A
     11%|█         | 141/1261 [00:25<03:32,  5.26it/s][A
     11%|█▏        | 142/1261 [00:25<03:33,  5.24it/s][A
     11%|█▏        | 143/1261 [00:25<03:49,  4.86it/s][A
     11%|█▏        | 144/1261 [00:26<04:11,  4.43it/s][A
     11%|█▏        | 145/1261 [00:26<04:04,  4.56it/s][A
     12%|█▏        | 146/1261 [00:26<03:53,  4.78it/s][A
     12%|█▏        | 147/1261 [00:26<03:24,  5.46it/s][A
     12%|█▏        | 148/1261 [00:26<04:07,  4.50it/s][A
     12%|█▏        | 149/1261 [00:27<04:26,  4.17it/s][A
     12%|█▏        | 150/1261 [00:27<04:34,  4.05it/s][A
     12%|█▏        | 151/1261 [00:27<04:13,  4.38it/s][A
     12%|█▏        | 152/1261 [00:27<04:09,  4.45it/s][A
     12%|█▏        | 153/1261 [00:28<03:29,  5.30it/s][A
     12%|█▏        | 154/1261 [00:28<03:14,  5.70it/s][A
     12%|█▏        | 155/1261 [00:28<02:59,  6.15it/s][A
     12%|█▏        | 156/1261 [00:28<03:16,  5.62it/s][A
     12%|█▏        | 157/1261 [00:28<03:21,  5.49it/s][A
     13%|█▎        | 158/1261 [00:28<03:21,  5.49it/s][A
     13%|█▎        | 159/1261 [00:29<03:03,  5.99it/s][A
     13%|█▎        | 160/1261 [00:29<02:58,  6.17it/s][A
     13%|█▎        | 161/1261 [00:29<02:45,  6.63it/s][A
     13%|█▎        | 162/1261 [00:29<03:04,  5.97it/s][A
     13%|█▎        | 163/1261 [00:29<03:48,  4.80it/s][A
     13%|█▎        | 164/1261 [00:30<04:40,  3.91it/s][A
     13%|█▎        | 165/1261 [00:30<04:16,  4.28it/s][A
     13%|█▎        | 166/1261 [00:30<03:51,  4.73it/s][A
     13%|█▎        | 167/1261 [00:30<03:45,  4.86it/s][A
     13%|█▎        | 168/1261 [00:30<03:39,  4.99it/s][A
     13%|█▎        | 169/1261 [00:31<03:54,  4.66it/s][A
     13%|█▎        | 170/1261 [00:31<03:55,  4.62it/s][A
     14%|█▎        | 171/1261 [00:31<03:25,  5.29it/s][A
     14%|█▎        | 172/1261 [00:31<03:09,  5.74it/s][A
     14%|█▎        | 173/1261 [00:31<03:20,  5.42it/s][A
     14%|█▍        | 174/1261 [00:31<03:18,  5.47it/s][A
     14%|█▍        | 175/1261 [00:32<03:16,  5.54it/s][A
     14%|█▍        | 176/1261 [00:32<03:29,  5.19it/s][A
     14%|█▍        | 177/1261 [00:32<03:13,  5.60it/s][A
     14%|█▍        | 178/1261 [00:32<03:07,  5.77it/s][A
     14%|█▍        | 179/1261 [00:32<02:57,  6.08it/s][A
     14%|█▍        | 180/1261 [00:33<03:08,  5.75it/s][A
     14%|█▍        | 181/1261 [00:33<03:02,  5.90it/s][A
     14%|█▍        | 182/1261 [00:33<03:01,  5.93it/s][A
     15%|█▍        | 183/1261 [00:33<02:52,  6.26it/s][A
     15%|█▍        | 184/1261 [00:33<02:56,  6.12it/s][A
     15%|█▍        | 185/1261 [00:33<03:10,  5.64it/s][A
     15%|█▍        | 186/1261 [00:34<03:12,  5.59it/s][A
     15%|█▍        | 187/1261 [00:34<03:14,  5.51it/s][A
     15%|█▍        | 188/1261 [00:34<03:03,  5.86it/s][A
     15%|█▍        | 189/1261 [00:34<03:00,  5.94it/s][A
     15%|█▌        | 190/1261 [00:34<03:13,  5.55it/s][A
     15%|█▌        | 191/1261 [00:35<03:56,  4.52it/s][A
     15%|█▌        | 192/1261 [00:35<03:45,  4.73it/s][A
     15%|█▌        | 193/1261 [00:35<03:37,  4.90it/s][A
     15%|█▌        | 194/1261 [00:35<03:35,  4.95it/s][A
     15%|█▌        | 195/1261 [00:36<04:26,  4.00it/s][A
     16%|█▌        | 196/1261 [00:36<04:01,  4.41it/s][A
     16%|█▌        | 197/1261 [00:36<03:32,  5.02it/s][A
     16%|█▌        | 198/1261 [00:36<03:41,  4.79it/s][A
     16%|█▌        | 199/1261 [00:36<03:29,  5.06it/s][A
     16%|█▌        | 200/1261 [00:36<03:12,  5.52it/s][A
     16%|█▌        | 201/1261 [00:37<03:22,  5.23it/s][A
     16%|█▌        | 202/1261 [00:37<03:10,  5.56it/s][A
     16%|█▌        | 203/1261 [00:37<03:25,  5.15it/s][A
     16%|█▌        | 204/1261 [00:37<03:15,  5.41it/s][A
     16%|█▋        | 205/1261 [00:37<03:01,  5.82it/s][A
     16%|█▋        | 206/1261 [00:37<03:08,  5.60it/s][A
     16%|█▋        | 207/1261 [00:38<04:05,  4.30it/s][A
     16%|█▋        | 208/1261 [00:38<03:58,  4.42it/s][A
     17%|█▋        | 209/1261 [00:38<03:30,  4.99it/s][A
     17%|█▋        | 210/1261 [00:38<03:04,  5.68it/s][A
     17%|█▋        | 211/1261 [00:38<02:54,  6.00it/s][A
     17%|█▋        | 212/1261 [00:39<02:44,  6.39it/s][A
     17%|█▋        | 213/1261 [00:39<02:52,  6.08it/s][A
     17%|█▋        | 214/1261 [00:39<02:39,  6.55it/s][A
     17%|█▋        | 215/1261 [00:39<03:12,  5.44it/s][A
     17%|█▋        | 216/1261 [00:39<03:00,  5.79it/s][A
     17%|█▋        | 217/1261 [00:39<02:58,  5.84it/s][A
     17%|█▋        | 218/1261 [00:40<02:41,  6.47it/s][A
     17%|█▋        | 219/1261 [00:40<03:15,  5.33it/s][A
     17%|█▋        | 220/1261 [00:40<03:47,  4.57it/s][A
     18%|█▊        | 221/1261 [00:40<03:47,  4.58it/s][A
     18%|█▊        | 222/1261 [00:41<03:38,  4.76it/s][A
     18%|█▊        | 223/1261 [00:41<03:42,  4.68it/s][A
     18%|█▊        | 224/1261 [00:41<04:03,  4.25it/s][A
     18%|█▊        | 225/1261 [00:41<03:51,  4.48it/s][A
     18%|█▊        | 226/1261 [00:41<03:18,  5.21it/s][A
     18%|█▊        | 227/1261 [00:42<03:33,  4.84it/s][A
     18%|█▊        | 228/1261 [00:42<03:31,  4.88it/s][A
     18%|█▊        | 229/1261 [00:42<03:10,  5.43it/s][A
     18%|█▊        | 230/1261 [00:42<02:47,  6.15it/s][A
     18%|█▊        | 231/1261 [00:42<02:36,  6.57it/s][A
     18%|█▊        | 232/1261 [00:42<02:37,  6.52it/s][A
     18%|█▊        | 233/1261 [00:42<02:38,  6.50it/s][A
     19%|█▊        | 234/1261 [00:43<02:54,  5.87it/s][A
     19%|█▊        | 235/1261 [00:43<02:42,  6.31it/s][A
     19%|█▊        | 236/1261 [00:43<02:33,  6.68it/s][A
     19%|█▉        | 237/1261 [00:43<02:18,  7.38it/s][A
     19%|█▉        | 238/1261 [00:43<02:26,  6.98it/s][A
     19%|█▉        | 239/1261 [00:43<02:41,  6.33it/s][A
     19%|█▉        | 240/1261 [00:44<02:38,  6.45it/s][A
     19%|█▉        | 241/1261 [00:44<03:04,  5.54it/s][A
     19%|█▉        | 242/1261 [00:44<02:49,  6.03it/s][A
     19%|█▉        | 243/1261 [00:44<03:53,  4.36it/s][A
     19%|█▉        | 244/1261 [00:44<03:23,  5.01it/s][A
     19%|█▉        | 245/1261 [00:45<03:05,  5.46it/s][A
     20%|█▉        | 246/1261 [00:45<02:50,  5.96it/s][A
     20%|█▉        | 247/1261 [00:45<02:39,  6.37it/s][A
     20%|█▉        | 248/1261 [00:45<02:32,  6.63it/s][A
     20%|█▉        | 249/1261 [00:45<02:47,  6.04it/s][A
     20%|█▉        | 250/1261 [00:45<02:34,  6.53it/s][A
     20%|█▉        | 251/1261 [00:45<02:31,  6.68it/s][A
     20%|█▉        | 252/1261 [00:46<03:24,  4.94it/s][A
     20%|██        | 253/1261 [00:46<03:16,  5.14it/s][A
     20%|██        | 254/1261 [00:46<02:58,  5.65it/s][A
     20%|██        | 255/1261 [00:46<02:41,  6.24it/s][A
     20%|██        | 256/1261 [00:46<02:35,  6.45it/s][A
     20%|██        | 257/1261 [00:46<02:19,  7.22it/s][A
     20%|██        | 258/1261 [00:47<02:26,  6.84it/s][A
     21%|██        | 259/1261 [00:47<03:02,  5.50it/s][A
     21%|██        | 260/1261 [00:47<02:51,  5.83it/s][A
     21%|██        | 261/1261 [00:47<02:44,  6.10it/s][A
     21%|██        | 262/1261 [00:47<02:32,  6.53it/s][A
     21%|██        | 263/1261 [00:47<02:29,  6.67it/s][A
     21%|██        | 264/1261 [00:48<02:43,  6.11it/s][A
     21%|██        | 265/1261 [00:48<02:34,  6.46it/s][A
     21%|██        | 266/1261 [00:48<02:54,  5.72it/s][A
     21%|██        | 267/1261 [00:48<02:39,  6.24it/s][A
     21%|██▏       | 268/1261 [00:48<02:30,  6.59it/s][A
     21%|██▏       | 269/1261 [00:48<02:25,  6.80it/s][A
     21%|██▏       | 270/1261 [00:49<03:05,  5.33it/s][A
     21%|██▏       | 271/1261 [00:49<02:57,  5.59it/s][A
     22%|██▏       | 272/1261 [00:49<03:45,  4.38it/s][A
     22%|██▏       | 273/1261 [00:49<03:46,  4.35it/s][A
     22%|██▏       | 274/1261 [00:50<03:23,  4.84it/s][A
     22%|██▏       | 275/1261 [00:50<02:59,  5.50it/s][A
     22%|██▏       | 276/1261 [00:50<02:47,  5.87it/s][A
     22%|██▏       | 277/1261 [00:50<02:30,  6.53it/s][A
     22%|██▏       | 278/1261 [00:50<02:51,  5.74it/s][A
     22%|██▏       | 279/1261 [00:50<03:17,  4.97it/s][A
     22%|██▏       | 280/1261 [00:51<03:16,  4.98it/s][A
     22%|██▏       | 281/1261 [00:51<02:51,  5.72it/s][A
     22%|██▏       | 282/1261 [00:51<02:35,  6.29it/s][A
     22%|██▏       | 283/1261 [00:51<03:03,  5.32it/s][A
     23%|██▎       | 284/1261 [00:51<03:37,  4.49it/s][A
     23%|██▎       | 285/1261 [00:52<03:44,  4.34it/s][A
     23%|██▎       | 286/1261 [00:52<03:44,  4.35it/s][A
     23%|██▎       | 287/1261 [00:52<04:08,  3.92it/s][A
     23%|██▎       | 288/1261 [00:53<05:08,  3.15it/s][A
     23%|██▎       | 289/1261 [00:53<05:04,  3.20it/s][A
     23%|██▎       | 290/1261 [00:53<04:41,  3.45it/s][A
     23%|██▎       | 291/1261 [00:53<04:29,  3.61it/s][A
     23%|██▎       | 292/1261 [00:54<04:29,  3.59it/s][A
     23%|██▎       | 293/1261 [00:54<04:16,  3.77it/s][A
     23%|██▎       | 294/1261 [00:54<03:45,  4.28it/s][A
     23%|██▎       | 295/1261 [00:54<03:25,  4.71it/s][A
     23%|██▎       | 296/1261 [00:55<03:32,  4.54it/s][A
     24%|██▎       | 297/1261 [00:55<04:07,  3.90it/s][A
     24%|██▎       | 298/1261 [00:55<04:16,  3.76it/s][A
     24%|██▎       | 299/1261 [00:55<04:13,  3.79it/s][A
     24%|██▍       | 300/1261 [00:56<03:59,  4.01it/s][A
     24%|██▍       | 301/1261 [00:56<03:56,  4.06it/s][A
     24%|██▍       | 302/1261 [00:56<03:42,  4.31it/s][A
     24%|██▍       | 303/1261 [00:56<03:24,  4.69it/s][A
     24%|██▍       | 304/1261 [00:56<03:28,  4.59it/s][A
     24%|██▍       | 305/1261 [00:57<03:17,  4.84it/s][A
     24%|██▍       | 306/1261 [00:57<03:10,  5.01it/s][A
     24%|██▍       | 307/1261 [00:57<03:47,  4.19it/s][A
     24%|██▍       | 308/1261 [00:58<05:08,  3.09it/s][A
     25%|██▍       | 309/1261 [00:58<04:35,  3.46it/s][A
     25%|██▍       | 310/1261 [00:58<03:53,  4.08it/s][A
     25%|██▍       | 311/1261 [00:58<03:18,  4.79it/s][A
     25%|██▍       | 312/1261 [00:58<03:06,  5.10it/s][A
     25%|██▍       | 313/1261 [00:58<02:51,  5.52it/s][A
     25%|██▍       | 314/1261 [00:59<02:40,  5.88it/s][A
     25%|██▍       | 315/1261 [00:59<03:22,  4.67it/s][A
     25%|██▌       | 316/1261 [00:59<03:03,  5.16it/s][A
     25%|██▌       | 317/1261 [01:00<04:05,  3.85it/s][A
     25%|██▌       | 318/1261 [01:00<04:36,  3.41it/s][A
     25%|██▌       | 319/1261 [01:00<04:24,  3.56it/s][A
     25%|██▌       | 320/1261 [01:00<03:59,  3.93it/s][A
     25%|██▌       | 321/1261 [01:01<03:56,  3.98it/s][A
     26%|██▌       | 322/1261 [01:01<03:50,  4.08it/s][A
     26%|██▌       | 323/1261 [01:01<03:19,  4.69it/s][A
     26%|██▌       | 324/1261 [01:01<03:13,  4.85it/s][A
     26%|██▌       | 325/1261 [01:01<03:23,  4.60it/s][A
     26%|██▌       | 326/1261 [01:02<03:22,  4.61it/s][A
     26%|██▌       | 327/1261 [01:02<04:18,  3.61it/s][A
     26%|██▌       | 328/1261 [01:02<03:48,  4.09it/s][A
     26%|██▌       | 329/1261 [01:02<03:17,  4.72it/s][A
     26%|██▌       | 330/1261 [01:03<03:20,  4.63it/s][A
     26%|██▌       | 331/1261 [01:03<03:10,  4.87it/s][A
     26%|██▋       | 332/1261 [01:03<03:08,  4.94it/s][A
     26%|██▋       | 333/1261 [01:03<03:45,  4.12it/s][A
     26%|██▋       | 334/1261 [01:03<03:35,  4.30it/s][A
     27%|██▋       | 335/1261 [01:04<03:43,  4.13it/s][A
     27%|██▋       | 336/1261 [01:04<03:55,  3.93it/s][A
     27%|██▋       | 337/1261 [01:04<03:29,  4.40it/s][A
     27%|██▋       | 338/1261 [01:04<03:32,  4.35it/s][A
     27%|██▋       | 339/1261 [01:05<03:06,  4.94it/s][A
     27%|██▋       | 340/1261 [01:05<02:51,  5.39it/s][A
     27%|██▋       | 341/1261 [01:05<03:14,  4.73it/s][A
     27%|██▋       | 342/1261 [01:05<03:46,  4.06it/s][A
     27%|██▋       | 343/1261 [01:06<03:54,  3.91it/s][A
     27%|██▋       | 344/1261 [01:06<04:06,  3.73it/s][A
     27%|██▋       | 345/1261 [01:06<04:58,  3.07it/s][A
     27%|██▋       | 346/1261 [01:06<04:04,  3.74it/s][A
     28%|██▊       | 347/1261 [01:07<03:59,  3.82it/s][A
     28%|██▊       | 348/1261 [01:07<04:42,  3.23it/s][A
     28%|██▊       | 349/1261 [01:07<04:18,  3.53it/s][A
     28%|██▊       | 350/1261 [01:08<03:52,  3.92it/s][A
     28%|██▊       | 351/1261 [01:08<03:30,  4.32it/s][A
     28%|██▊       | 352/1261 [01:08<03:26,  4.39it/s][A
     28%|██▊       | 353/1261 [01:08<03:27,  4.38it/s][A
     28%|██▊       | 354/1261 [01:08<03:49,  3.95it/s][A
     28%|██▊       | 355/1261 [01:09<03:20,  4.52it/s][A
     28%|██▊       | 356/1261 [01:09<03:22,  4.47it/s][A
     28%|██▊       | 357/1261 [01:09<03:17,  4.57it/s][A
     28%|██▊       | 358/1261 [01:09<03:14,  4.65it/s][A
     28%|██▊       | 359/1261 [01:09<03:15,  4.61it/s][A
     29%|██▊       | 360/1261 [01:10<03:19,  4.52it/s][A
     29%|██▊       | 361/1261 [01:10<03:24,  4.41it/s][A
     29%|██▊       | 362/1261 [01:10<03:03,  4.89it/s][A
     29%|██▉       | 363/1261 [01:10<03:12,  4.67it/s][A
     29%|██▉       | 364/1261 [01:11<03:18,  4.51it/s][A
     29%|██▉       | 365/1261 [01:11<03:03,  4.89it/s][A
     29%|██▉       | 366/1261 [01:11<02:55,  5.10it/s][A
     29%|██▉       | 367/1261 [01:11<02:55,  5.10it/s][A
     29%|██▉       | 368/1261 [01:11<03:07,  4.75it/s][A
     29%|██▉       | 369/1261 [01:11<02:45,  5.39it/s][A
     29%|██▉       | 370/1261 [01:12<02:36,  5.68it/s][A
     29%|██▉       | 371/1261 [01:12<02:31,  5.86it/s][A
     30%|██▉       | 372/1261 [01:12<02:26,  6.05it/s][A
     30%|██▉       | 373/1261 [01:12<02:31,  5.85it/s][A
     30%|██▉       | 374/1261 [01:12<03:03,  4.83it/s][A
     30%|██▉       | 375/1261 [01:13<03:47,  3.90it/s][A
     30%|██▉       | 376/1261 [01:13<03:21,  4.39it/s][A
     30%|██▉       | 377/1261 [01:13<03:07,  4.71it/s][A
     30%|██▉       | 378/1261 [01:13<03:15,  4.52it/s][A
     30%|███       | 379/1261 [01:14<02:55,  5.02it/s][A
     30%|███       | 380/1261 [01:14<02:51,  5.13it/s][A
     30%|███       | 381/1261 [01:14<02:56,  4.98it/s][A
     30%|███       | 382/1261 [01:14<03:09,  4.64it/s][A
     30%|███       | 383/1261 [01:14<03:05,  4.72it/s][A
     30%|███       | 384/1261 [01:15<03:29,  4.19it/s][A
     31%|███       | 385/1261 [01:15<04:05,  3.57it/s][A
     31%|███       | 386/1261 [01:15<04:05,  3.56it/s][A
     31%|███       | 387/1261 [01:15<03:30,  4.16it/s][A
     31%|███       | 388/1261 [01:16<03:12,  4.53it/s][A
     31%|███       | 389/1261 [01:16<02:57,  4.91it/s][A
     31%|███       | 390/1261 [01:16<03:20,  4.34it/s][A
     31%|███       | 391/1261 [01:16<03:36,  4.02it/s][A
     31%|███       | 392/1261 [01:17<04:01,  3.60it/s][A
     31%|███       | 393/1261 [01:17<04:34,  3.16it/s][A
     31%|███       | 394/1261 [01:17<04:36,  3.14it/s][A
     31%|███▏      | 395/1261 [01:18<04:01,  3.59it/s][A
     31%|███▏      | 396/1261 [01:18<03:41,  3.91it/s][A
     31%|███▏      | 397/1261 [01:18<03:23,  4.24it/s][A
     32%|███▏      | 398/1261 [01:18<03:08,  4.57it/s][A
     32%|███▏      | 399/1261 [01:18<02:51,  5.04it/s][A
     32%|███▏      | 400/1261 [01:19<03:00,  4.78it/s][A
     32%|███▏      | 401/1261 [01:19<03:48,  3.77it/s][A
     32%|███▏      | 402/1261 [01:19<04:14,  3.37it/s][A
     32%|███▏      | 403/1261 [01:20<03:42,  3.85it/s][A
     32%|███▏      | 404/1261 [01:20<03:17,  4.34it/s][A
     32%|███▏      | 405/1261 [01:20<02:47,  5.10it/s][A
     32%|███▏      | 406/1261 [01:20<02:40,  5.33it/s][A
     32%|███▏      | 407/1261 [01:20<02:36,  5.46it/s][A
     32%|███▏      | 408/1261 [01:21<03:50,  3.70it/s][A
     32%|███▏      | 409/1261 [01:21<03:52,  3.67it/s][A
     33%|███▎      | 410/1261 [01:21<04:10,  3.40it/s][A
     33%|███▎      | 411/1261 [01:22<04:22,  3.23it/s][A
     33%|███▎      | 412/1261 [01:22<04:02,  3.51it/s][A
     33%|███▎      | 413/1261 [01:22<03:43,  3.80it/s][A
     33%|███▎      | 414/1261 [01:22<03:34,  3.94it/s][A
     33%|███▎      | 415/1261 [01:23<03:40,  3.84it/s][A
     33%|███▎      | 416/1261 [01:23<03:43,  3.78it/s][A
     33%|███▎      | 417/1261 [01:23<03:28,  4.05it/s][A
     33%|███▎      | 418/1261 [01:23<03:34,  3.92it/s][A
     33%|███▎      | 419/1261 [01:24<03:39,  3.84it/s][A
     33%|███▎      | 420/1261 [01:24<03:04,  4.56it/s][A
     33%|███▎      | 421/1261 [01:24<02:42,  5.16it/s][A
     33%|███▎      | 422/1261 [01:24<02:20,  5.96it/s][A
     34%|███▎      | 423/1261 [01:24<02:12,  6.31it/s][A
     34%|███▎      | 424/1261 [01:25<03:25,  4.08it/s][A
     34%|███▎      | 425/1261 [01:25<03:33,  3.91it/s][A
     34%|███▍      | 426/1261 [01:25<03:27,  4.02it/s][A
     34%|███▍      | 427/1261 [01:25<03:19,  4.18it/s][A
     34%|███▍      | 428/1261 [01:25<03:06,  4.46it/s][A
     34%|███▍      | 429/1261 [01:26<03:04,  4.51it/s][A
     34%|███▍      | 430/1261 [01:26<03:40,  3.76it/s][A
     34%|███▍      | 431/1261 [01:26<03:16,  4.22it/s][A
     34%|███▍      | 432/1261 [01:27<03:30,  3.93it/s][A
     34%|███▍      | 433/1261 [01:27<03:08,  4.39it/s][A
     34%|███▍      | 434/1261 [01:27<03:19,  4.14it/s][A
     34%|███▍      | 435/1261 [01:27<03:12,  4.29it/s][A
     35%|███▍      | 436/1261 [01:27<03:27,  3.98it/s][A
     35%|███▍      | 437/1261 [01:28<03:12,  4.28it/s][A
     35%|███▍      | 438/1261 [01:28<02:48,  4.90it/s][A
     35%|███▍      | 439/1261 [01:28<02:32,  5.41it/s][A
     35%|███▍      | 440/1261 [01:28<02:20,  5.83it/s][A
     35%|███▍      | 441/1261 [01:28<02:09,  6.31it/s][A
     35%|███▌      | 442/1261 [01:28<02:06,  6.47it/s][A
     35%|███▌      | 443/1261 [01:29<02:06,  6.48it/s][A
     35%|███▌      | 444/1261 [01:29<02:54,  4.68it/s][A
     35%|███▌      | 445/1261 [01:29<02:36,  5.21it/s][A
     35%|███▌      | 446/1261 [01:29<02:27,  5.54it/s][A
     35%|███▌      | 447/1261 [01:29<02:10,  6.23it/s][A
     36%|███▌      | 448/1261 [01:29<01:59,  6.80it/s][A
     36%|███▌      | 449/1261 [01:29<01:49,  7.39it/s][A
     36%|███▌      | 450/1261 [01:30<01:45,  7.67it/s][A
     36%|███▌      | 451/1261 [01:30<01:53,  7.16it/s][A
     36%|███▌      | 452/1261 [01:30<02:33,  5.26it/s][A
     36%|███▌      | 453/1261 [01:30<02:38,  5.11it/s][A
     36%|███▌      | 454/1261 [01:30<02:19,  5.79it/s][A
     36%|███▌      | 455/1261 [01:31<02:02,  6.58it/s][A
     36%|███▌      | 456/1261 [01:31<01:57,  6.83it/s][A
     36%|███▌      | 457/1261 [01:31<01:49,  7.33it/s][A
     36%|███▋      | 458/1261 [01:31<02:01,  6.58it/s][A
     36%|███▋      | 459/1261 [01:31<02:11,  6.10it/s][A
     36%|███▋      | 460/1261 [01:31<02:18,  5.78it/s][A
     37%|███▋      | 461/1261 [01:31<02:15,  5.91it/s][A
     37%|███▋      | 462/1261 [01:32<02:05,  6.35it/s][A
     37%|███▋      | 463/1261 [01:32<01:55,  6.93it/s][A
     37%|███▋      | 464/1261 [01:32<01:48,  7.32it/s][A
     37%|███▋      | 465/1261 [01:32<01:51,  7.17it/s][A
     37%|███▋      | 466/1261 [01:32<01:48,  7.35it/s][A
     37%|███▋      | 467/1261 [01:32<02:11,  6.05it/s][A
     37%|███▋      | 468/1261 [01:33<02:21,  5.59it/s][A
     37%|███▋      | 469/1261 [01:33<02:12,  5.97it/s][A
     37%|███▋      | 470/1261 [01:33<02:31,  5.23it/s][A
     37%|███▋      | 471/1261 [01:33<02:28,  5.33it/s][A
     37%|███▋      | 472/1261 [01:33<02:31,  5.22it/s][A
     38%|███▊      | 473/1261 [01:33<02:18,  5.69it/s][A
     38%|███▊      | 474/1261 [01:34<02:20,  5.59it/s][A
     38%|███▊      | 475/1261 [01:34<02:26,  5.35it/s][A
     38%|███▊      | 476/1261 [01:34<02:18,  5.68it/s][A
     38%|███▊      | 477/1261 [01:34<02:18,  5.66it/s][A
     38%|███▊      | 478/1261 [01:35<03:07,  4.17it/s][A
     38%|███▊      | 479/1261 [01:35<02:59,  4.36it/s][A
     38%|███▊      | 480/1261 [01:35<02:39,  4.91it/s][A
     38%|███▊      | 481/1261 [01:35<02:36,  5.00it/s][A
     38%|███▊      | 482/1261 [01:35<02:26,  5.30it/s][A
     38%|███▊      | 483/1261 [01:36<02:40,  4.86it/s][A
     38%|███▊      | 484/1261 [01:36<02:57,  4.39it/s][A
     38%|███▊      | 485/1261 [01:36<03:19,  3.89it/s][A
     39%|███▊      | 486/1261 [01:36<03:30,  3.68it/s][A
     39%|███▊      | 487/1261 [01:37<04:12,  3.06it/s][A
     39%|███▊      | 488/1261 [01:37<03:28,  3.71it/s][A
     39%|███▉      | 489/1261 [01:37<02:54,  4.41it/s][A
     39%|███▉      | 490/1261 [01:37<02:35,  4.95it/s][A
     39%|███▉      | 491/1261 [01:37<02:23,  5.37it/s][A
     39%|███▉      | 492/1261 [01:38<03:03,  4.18it/s][A
     39%|███▉      | 493/1261 [01:38<02:49,  4.54it/s][A
     39%|███▉      | 494/1261 [01:38<03:25,  3.74it/s][A
     39%|███▉      | 495/1261 [01:39<03:24,  3.74it/s][A
     39%|███▉      | 496/1261 [01:39<03:49,  3.34it/s][A
     39%|███▉      | 497/1261 [01:39<03:32,  3.60it/s][A
     39%|███▉      | 498/1261 [01:40<03:32,  3.59it/s][A
     40%|███▉      | 499/1261 [01:40<03:08,  4.03it/s][A
     40%|███▉      | 500/1261 [01:40<03:03,  4.15it/s][A
     40%|███▉      | 501/1261 [01:40<02:55,  4.33it/s][A
     40%|███▉      | 502/1261 [01:40<02:57,  4.27it/s][A
     40%|███▉      | 503/1261 [01:41<02:39,  4.74it/s][A
     40%|███▉      | 504/1261 [01:41<02:43,  4.64it/s][A
     40%|████      | 505/1261 [01:41<02:31,  4.99it/s][A
     40%|████      | 506/1261 [01:41<02:45,  4.57it/s][A
     40%|████      | 507/1261 [01:41<02:25,  5.19it/s][A
     40%|████      | 508/1261 [01:42<03:00,  4.16it/s][A
     40%|████      | 509/1261 [01:42<02:54,  4.31it/s][A
     40%|████      | 510/1261 [01:42<02:35,  4.83it/s][A
     41%|████      | 511/1261 [01:42<02:25,  5.16it/s][A
     41%|████      | 512/1261 [01:42<02:21,  5.30it/s][A
     41%|████      | 513/1261 [01:43<02:13,  5.59it/s][A
     41%|████      | 514/1261 [01:43<02:25,  5.13it/s][A
     41%|████      | 515/1261 [01:43<02:20,  5.32it/s][A
     41%|████      | 516/1261 [01:43<02:10,  5.71it/s][A
     41%|████      | 517/1261 [01:43<01:55,  6.43it/s][A
     41%|████      | 518/1261 [01:43<01:55,  6.43it/s][A
     41%|████      | 519/1261 [01:44<02:02,  6.04it/s][A
     41%|████      | 520/1261 [01:44<01:56,  6.34it/s][A
     41%|████▏     | 521/1261 [01:44<01:46,  6.97it/s][A
     41%|████▏     | 522/1261 [01:44<01:40,  7.38it/s][A
     41%|████▏     | 523/1261 [01:44<02:03,  5.99it/s][A
     42%|████▏     | 524/1261 [01:44<01:58,  6.23it/s][A
     42%|████▏     | 525/1261 [01:44<01:48,  6.76it/s][A
     42%|████▏     | 526/1261 [01:45<01:41,  7.27it/s][A
     42%|████▏     | 527/1261 [01:45<01:36,  7.61it/s][A
     42%|████▏     | 528/1261 [01:45<01:33,  7.83it/s][A
     42%|████▏     | 529/1261 [01:45<01:40,  7.27it/s][A
     42%|████▏     | 530/1261 [01:45<01:56,  6.25it/s][A
     42%|████▏     | 531/1261 [01:45<01:53,  6.45it/s][A
     42%|████▏     | 532/1261 [01:45<01:52,  6.48it/s][A
     42%|████▏     | 533/1261 [01:46<01:47,  6.74it/s][A
     42%|████▏     | 534/1261 [01:46<01:46,  6.85it/s][A
     42%|████▏     | 535/1261 [01:46<01:48,  6.69it/s][A
     43%|████▎     | 536/1261 [01:46<01:41,  7.17it/s][A
     43%|████▎     | 537/1261 [01:46<01:41,  7.10it/s][A
     43%|████▎     | 538/1261 [01:46<02:18,  5.22it/s][A
     43%|████▎     | 539/1261 [01:47<02:15,  5.31it/s][A
     43%|████▎     | 540/1261 [01:47<02:02,  5.86it/s][A
     43%|████▎     | 541/1261 [01:47<01:48,  6.64it/s][A
     43%|████▎     | 542/1261 [01:47<02:03,  5.81it/s][A
     43%|████▎     | 543/1261 [01:47<02:53,  4.15it/s][A
     43%|████▎     | 544/1261 [01:48<03:03,  3.90it/s][A
     43%|████▎     | 545/1261 [01:48<02:53,  4.13it/s][A
     43%|████▎     | 546/1261 [01:48<02:36,  4.56it/s][A
     43%|████▎     | 547/1261 [01:48<02:37,  4.55it/s][A
     43%|████▎     | 548/1261 [01:49<02:46,  4.29it/s][A
     44%|████▎     | 549/1261 [01:49<02:19,  5.11it/s][A
     44%|████▎     | 550/1261 [01:49<02:05,  5.67it/s][A
     44%|████▎     | 551/1261 [01:49<01:57,  6.06it/s][A
     44%|████▍     | 552/1261 [01:49<01:46,  6.65it/s][A
     44%|████▍     | 553/1261 [01:49<01:48,  6.50it/s][A
     44%|████▍     | 554/1261 [01:49<01:54,  6.18it/s][A
     44%|████▍     | 555/1261 [01:50<02:13,  5.28it/s][A
     44%|████▍     | 556/1261 [01:50<02:21,  4.99it/s][A
     44%|████▍     | 557/1261 [01:50<02:51,  4.10it/s][A
     44%|████▍     | 558/1261 [01:50<02:45,  4.25it/s][A
     44%|████▍     | 559/1261 [01:51<02:47,  4.18it/s][A
     44%|████▍     | 560/1261 [01:51<02:32,  4.61it/s][A
     44%|████▍     | 561/1261 [01:51<03:03,  3.82it/s][A
     45%|████▍     | 562/1261 [01:52<03:00,  3.88it/s][A
     45%|████▍     | 563/1261 [01:52<02:28,  4.70it/s][A
     45%|████▍     | 565/1261 [01:52<02:04,  5.58it/s][A
     45%|████▍     | 566/1261 [01:52<01:59,  5.82it/s][A
     45%|████▍     | 567/1261 [01:52<02:01,  5.73it/s][A
     45%|████▌     | 568/1261 [01:53<02:41,  4.30it/s][A
     45%|████▌     | 569/1261 [01:53<02:39,  4.35it/s][A
     45%|████▌     | 570/1261 [01:53<02:32,  4.54it/s][A
     45%|████▌     | 571/1261 [01:53<02:11,  5.24it/s][A
     45%|████▌     | 572/1261 [01:53<02:20,  4.92it/s][A
     45%|████▌     | 573/1261 [01:53<02:05,  5.50it/s][A
     46%|████▌     | 574/1261 [01:54<02:00,  5.70it/s][A
     46%|████▌     | 575/1261 [01:54<02:16,  5.04it/s][A
     46%|████▌     | 576/1261 [01:54<02:09,  5.29it/s][A
     46%|████▌     | 577/1261 [01:54<02:17,  4.96it/s][A
     46%|████▌     | 578/1261 [01:55<03:03,  3.71it/s][A
     46%|████▌     | 579/1261 [01:55<03:01,  3.75it/s][A
     46%|████▌     | 580/1261 [01:55<02:30,  4.52it/s][A
     46%|████▌     | 581/1261 [01:55<02:11,  5.18it/s][A
     46%|████▌     | 582/1261 [01:55<02:09,  5.26it/s][A
     46%|████▌     | 583/1261 [01:56<02:18,  4.88it/s][A
     46%|████▋     | 584/1261 [01:56<02:25,  4.64it/s][A
     46%|████▋     | 585/1261 [01:56<02:07,  5.31it/s][A
     46%|████▋     | 586/1261 [01:56<02:12,  5.09it/s][A
     47%|████▋     | 587/1261 [01:56<02:23,  4.68it/s][A
     47%|████▋     | 588/1261 [01:57<02:33,  4.39it/s][A
     47%|████▋     | 589/1261 [01:57<02:47,  4.01it/s][A
     47%|████▋     | 590/1261 [01:57<02:39,  4.20it/s][A
     47%|████▋     | 591/1261 [01:57<02:24,  4.64it/s][A
     47%|████▋     | 592/1261 [01:58<02:12,  5.06it/s][A
     47%|████▋     | 593/1261 [01:58<01:52,  5.94it/s][A
     47%|████▋     | 594/1261 [01:58<01:44,  6.37it/s][A
     47%|████▋     | 595/1261 [01:58<01:51,  5.99it/s][A
     47%|████▋     | 596/1261 [01:58<01:41,  6.55it/s][A
     47%|████▋     | 597/1261 [01:58<01:37,  6.81it/s][A
     47%|████▋     | 598/1261 [01:58<01:34,  7.01it/s][A
     48%|████▊     | 599/1261 [01:59<02:15,  4.88it/s][A
     48%|████▊     | 600/1261 [01:59<01:57,  5.61it/s][A
     48%|████▊     | 601/1261 [01:59<01:54,  5.75it/s][A
     48%|████▊     | 602/1261 [01:59<01:46,  6.21it/s][A
     48%|████▊     | 603/1261 [01:59<01:45,  6.22it/s][A
     48%|████▊     | 604/1261 [01:59<01:42,  6.39it/s][A
     48%|████▊     | 605/1261 [02:00<01:36,  6.77it/s][A
     48%|████▊     | 606/1261 [02:00<02:12,  4.96it/s][A
     48%|████▊     | 607/1261 [02:00<02:08,  5.09it/s][A
     48%|████▊     | 608/1261 [02:00<01:50,  5.93it/s][A
     48%|████▊     | 610/1261 [02:00<01:37,  6.71it/s][A
     48%|████▊     | 611/1261 [02:00<01:27,  7.39it/s][A
     49%|████▊     | 613/1261 [02:01<01:42,  6.32it/s][A
     49%|████▊     | 614/1261 [02:01<01:53,  5.68it/s][A
     49%|████▉     | 616/1261 [02:01<01:39,  6.47it/s][A
     49%|████▉     | 617/1261 [02:01<01:30,  7.14it/s][A
     49%|████▉     | 619/1261 [02:02<01:21,  7.86it/s][A
     49%|████▉     | 620/1261 [02:02<01:27,  7.33it/s][A
     49%|████▉     | 621/1261 [02:02<01:27,  7.31it/s][A
     49%|████▉     | 622/1261 [02:02<02:10,  4.91it/s][A
     49%|████▉     | 623/1261 [02:02<02:12,  4.82it/s][A
     49%|████▉     | 624/1261 [02:03<01:58,  5.38it/s][A
     50%|████▉     | 625/1261 [02:03<01:46,  5.97it/s][A
     50%|████▉     | 626/1261 [02:03<01:38,  6.48it/s][A
     50%|████▉     | 627/1261 [02:03<01:29,  7.10it/s][A
     50%|████▉     | 628/1261 [02:03<01:21,  7.73it/s][A
     50%|████▉     | 629/1261 [02:03<01:43,  6.09it/s][A
     50%|████▉     | 630/1261 [02:04<01:50,  5.72it/s][A
     50%|█████     | 631/1261 [02:04<01:51,  5.64it/s][A
     50%|█████     | 632/1261 [02:04<01:50,  5.69it/s][A
     50%|█████     | 633/1261 [02:04<01:48,  5.76it/s][A
     50%|█████     | 634/1261 [02:04<01:44,  6.00it/s][A
     50%|█████     | 635/1261 [02:04<01:44,  5.98it/s][A
     50%|█████     | 636/1261 [02:04<01:38,  6.32it/s][A
     51%|█████     | 637/1261 [02:05<01:53,  5.52it/s][A
     51%|█████     | 638/1261 [02:05<01:48,  5.72it/s][A
     51%|█████     | 639/1261 [02:05<01:40,  6.19it/s][A
     51%|█████     | 640/1261 [02:05<01:29,  6.97it/s][A
     51%|█████     | 641/1261 [02:05<01:22,  7.48it/s][A
     51%|█████     | 642/1261 [02:05<01:18,  7.91it/s][A
     51%|█████     | 643/1261 [02:06<01:28,  6.99it/s][A
     51%|█████     | 644/1261 [02:06<01:31,  6.72it/s][A
     51%|█████     | 645/1261 [02:06<01:27,  7.03it/s][A
     51%|█████     | 646/1261 [02:06<01:54,  5.38it/s][A
     51%|█████▏    | 647/1261 [02:06<01:50,  5.56it/s][A
     51%|█████▏    | 649/1261 [02:06<01:35,  6.42it/s][A
     52%|█████▏    | 650/1261 [02:07<01:34,  6.46it/s][A
     52%|█████▏    | 651/1261 [02:07<01:27,  6.94it/s][A
     52%|█████▏    | 652/1261 [02:07<01:30,  6.70it/s][A
     52%|█████▏    | 653/1261 [02:07<01:30,  6.74it/s][A
     52%|█████▏    | 654/1261 [02:07<02:06,  4.80it/s][A
     52%|█████▏    | 655/1261 [02:08<02:05,  4.83it/s][A
     52%|█████▏    | 656/1261 [02:08<01:47,  5.63it/s][A
     52%|█████▏    | 657/1261 [02:08<01:35,  6.32it/s][A
     52%|█████▏    | 659/1261 [02:08<01:23,  7.18it/s][A
     52%|█████▏    | 660/1261 [02:08<01:17,  7.76it/s][A
     52%|█████▏    | 661/1261 [02:08<01:16,  7.81it/s][A
     52%|█████▏    | 662/1261 [02:09<01:54,  5.25it/s][A
     53%|█████▎    | 663/1261 [02:09<01:51,  5.36it/s][A
     53%|█████▎    | 664/1261 [02:09<01:39,  6.00it/s][A
     53%|█████▎    | 665/1261 [02:09<01:28,  6.75it/s][A
     53%|█████▎    | 667/1261 [02:09<01:18,  7.59it/s][A
     53%|█████▎    | 668/1261 [02:09<01:16,  7.72it/s][A
     53%|█████▎    | 669/1261 [02:09<01:18,  7.56it/s][A
     53%|█████▎    | 670/1261 [02:10<01:54,  5.15it/s][A
     53%|█████▎    | 671/1261 [02:10<01:47,  5.49it/s][A
     53%|█████▎    | 672/1261 [02:10<01:36,  6.09it/s][A
     53%|█████▎    | 674/1261 [02:10<01:24,  6.92it/s][A
     54%|█████▎    | 676/1261 [02:11<01:34,  6.21it/s][A
     54%|█████▎    | 677/1261 [02:11<01:33,  6.25it/s][A
     54%|█████▍    | 678/1261 [02:11<01:36,  6.03it/s][A
     54%|█████▍    | 679/1261 [02:11<01:40,  5.79it/s][A
     54%|█████▍    | 680/1261 [02:11<01:39,  5.83it/s][A
     54%|█████▍    | 681/1261 [02:11<01:33,  6.19it/s][A
     54%|█████▍    | 682/1261 [02:12<01:52,  5.15it/s][A
     54%|█████▍    | 684/1261 [02:12<01:36,  5.99it/s][A
     54%|█████▍    | 685/1261 [02:12<01:35,  6.06it/s][A
     54%|█████▍    | 686/1261 [02:12<01:28,  6.47it/s][A
     54%|█████▍    | 687/1261 [02:13<01:51,  5.13it/s][A
     55%|█████▍    | 688/1261 [02:13<01:51,  5.15it/s][A
     55%|█████▍    | 689/1261 [02:13<01:34,  6.02it/s][A
     55%|█████▍    | 691/1261 [02:13<01:22,  6.90it/s][A
     55%|█████▍    | 692/1261 [02:13<01:15,  7.55it/s][A
     55%|█████▍    | 693/1261 [02:13<01:28,  6.45it/s][A
     55%|█████▌    | 694/1261 [02:14<01:59,  4.74it/s][A
     55%|█████▌    | 695/1261 [02:14<02:05,  4.51it/s][A
     55%|█████▌    | 697/1261 [02:14<01:43,  5.45it/s][A
     55%|█████▌    | 698/1261 [02:14<01:30,  6.22it/s][A
     55%|█████▌    | 699/1261 [02:14<01:21,  6.90it/s][A
     56%|█████▌    | 700/1261 [02:14<01:15,  7.48it/s][A
     56%|█████▌    | 701/1261 [02:15<01:45,  5.32it/s][A
     56%|█████▌    | 702/1261 [02:15<01:32,  6.05it/s][A
     56%|█████▌    | 703/1261 [02:15<01:23,  6.72it/s][A
     56%|█████▌    | 704/1261 [02:15<01:15,  7.40it/s][A
     56%|█████▌    | 705/1261 [02:15<01:09,  7.96it/s][A
     56%|█████▌    | 706/1261 [02:15<01:08,  8.14it/s][A
     56%|█████▌    | 707/1261 [02:15<01:09,  7.99it/s][A
     56%|█████▌    | 708/1261 [02:16<01:26,  6.42it/s][A
     56%|█████▌    | 709/1261 [02:16<01:22,  6.67it/s][A
     56%|█████▋    | 710/1261 [02:16<01:35,  5.74it/s][A
     56%|█████▋    | 711/1261 [02:16<01:37,  5.65it/s][A
     56%|█████▋    | 712/1261 [02:16<01:27,  6.31it/s][A
     57%|█████▋    | 713/1261 [02:16<01:19,  6.88it/s][A
     57%|█████▋    | 714/1261 [02:17<01:25,  6.40it/s][A
     57%|█████▋    | 715/1261 [02:17<01:25,  6.41it/s][A
     57%|█████▋    | 716/1261 [02:17<01:25,  6.35it/s][A
     57%|█████▋    | 717/1261 [02:17<01:16,  7.08it/s][A
     57%|█████▋    | 718/1261 [02:17<01:21,  6.68it/s][A
     57%|█████▋    | 719/1261 [02:17<01:18,  6.90it/s][A
     57%|█████▋    | 720/1261 [02:17<01:19,  6.83it/s][A
     57%|█████▋    | 721/1261 [02:18<01:16,  7.04it/s][A
     57%|█████▋    | 722/1261 [02:18<01:33,  5.79it/s][A
     57%|█████▋    | 723/1261 [02:18<01:25,  6.27it/s][A
     57%|█████▋    | 724/1261 [02:18<01:20,  6.69it/s][A
     57%|█████▋    | 725/1261 [02:18<01:14,  7.24it/s][A
     58%|█████▊    | 726/1261 [02:18<01:22,  6.51it/s][A
     58%|█████▊    | 727/1261 [02:19<01:19,  6.72it/s][A
     58%|█████▊    | 728/1261 [02:19<01:18,  6.81it/s][A
     58%|█████▊    | 729/1261 [02:19<01:16,  6.92it/s][A
     58%|█████▊    | 730/1261 [02:19<01:29,  5.93it/s][A
     58%|█████▊    | 731/1261 [02:19<01:27,  6.05it/s][A
     58%|█████▊    | 732/1261 [02:19<01:19,  6.68it/s][A
     58%|█████▊    | 733/1261 [02:19<01:20,  6.59it/s][A
     58%|█████▊    | 734/1261 [02:20<01:13,  7.15it/s][A
     58%|█████▊    | 735/1261 [02:20<01:12,  7.22it/s][A
     58%|█████▊    | 736/1261 [02:20<01:10,  7.43it/s][A
     58%|█████▊    | 737/1261 [02:20<01:07,  7.74it/s][A
     59%|█████▊    | 738/1261 [02:20<01:26,  6.08it/s][A
     59%|█████▊    | 739/1261 [02:20<01:40,  5.21it/s][A
     59%|█████▊    | 740/1261 [02:21<01:27,  5.99it/s][A
     59%|█████▉    | 741/1261 [02:21<01:25,  6.05it/s][A
     59%|█████▉    | 742/1261 [02:21<01:20,  6.42it/s][A
     59%|█████▉    | 743/1261 [02:21<01:17,  6.73it/s][A
     59%|█████▉    | 744/1261 [02:21<01:16,  6.76it/s][A
     59%|█████▉    | 745/1261 [02:21<01:18,  6.58it/s][A
     59%|█████▉    | 746/1261 [02:21<01:15,  6.82it/s][A
     59%|█████▉    | 747/1261 [02:22<01:10,  7.33it/s][A
     59%|█████▉    | 748/1261 [02:22<01:07,  7.59it/s][A
     59%|█████▉    | 749/1261 [02:22<01:06,  7.73it/s][A
     59%|█████▉    | 750/1261 [02:22<01:04,  7.89it/s][A
     60%|█████▉    | 751/1261 [02:22<01:06,  7.63it/s][A
     60%|█████▉    | 752/1261 [02:22<01:30,  5.65it/s][A
     60%|█████▉    | 753/1261 [02:23<01:40,  5.04it/s][A
     60%|█████▉    | 754/1261 [02:23<01:26,  5.89it/s][A
     60%|█████▉    | 755/1261 [02:23<01:16,  6.61it/s][A
     60%|█████▉    | 756/1261 [02:23<01:09,  7.31it/s][A
     60%|██████    | 757/1261 [02:23<01:04,  7.78it/s][A
     60%|██████    | 758/1261 [02:23<01:12,  6.97it/s][A
     60%|██████    | 759/1261 [02:23<01:12,  6.96it/s][A
     60%|██████    | 760/1261 [02:24<01:36,  5.20it/s][A
     60%|██████    | 761/1261 [02:24<01:29,  5.61it/s][A
     61%|██████    | 763/1261 [02:24<01:18,  6.35it/s][A
     61%|██████    | 764/1261 [02:24<01:11,  6.99it/s][A
     61%|██████    | 765/1261 [02:24<01:05,  7.57it/s][A
     61%|██████    | 766/1261 [02:24<01:17,  6.36it/s][A
     61%|██████    | 767/1261 [02:25<01:14,  6.62it/s][A
     61%|██████    | 768/1261 [02:25<01:08,  7.19it/s][A
     61%|██████    | 769/1261 [02:25<01:05,  7.48it/s][A
     61%|██████    | 770/1261 [02:25<01:18,  6.22it/s][A
     61%|██████    | 771/1261 [02:25<01:15,  6.47it/s][A
     61%|██████    | 772/1261 [02:25<01:31,  5.34it/s][A
     61%|██████▏   | 773/1261 [02:26<01:23,  5.85it/s][A
     61%|██████▏   | 774/1261 [02:26<01:27,  5.54it/s][A
     61%|██████▏   | 775/1261 [02:26<01:17,  6.29it/s][A
     62%|██████▏   | 776/1261 [02:26<01:17,  6.29it/s][A
     62%|██████▏   | 777/1261 [02:26<01:11,  6.75it/s][A
     62%|██████▏   | 778/1261 [02:26<01:17,  6.23it/s][A
     62%|██████▏   | 779/1261 [02:26<01:12,  6.68it/s][A
     62%|██████▏   | 780/1261 [02:27<01:17,  6.21it/s][A
     62%|██████▏   | 781/1261 [02:27<01:43,  4.64it/s][A
     62%|██████▏   | 782/1261 [02:27<01:27,  5.45it/s][A
     62%|██████▏   | 784/1261 [02:27<01:15,  6.31it/s][A
     62%|██████▏   | 785/1261 [02:27<01:16,  6.20it/s][A
     62%|██████▏   | 786/1261 [02:28<01:11,  6.61it/s][A
     62%|██████▏   | 787/1261 [02:28<01:09,  6.82it/s][A
     62%|██████▏   | 788/1261 [02:28<01:04,  7.29it/s][A
     63%|██████▎   | 789/1261 [02:28<01:00,  7.81it/s][A
     63%|██████▎   | 790/1261 [02:28<00:57,  8.23it/s][A
     63%|██████▎   | 791/1261 [02:28<00:55,  8.51it/s][A
     63%|██████▎   | 792/1261 [02:28<00:55,  8.46it/s][A
     63%|██████▎   | 793/1261 [02:28<00:56,  8.32it/s][A
     63%|██████▎   | 794/1261 [02:29<01:15,  6.22it/s][A
     63%|██████▎   | 795/1261 [02:29<01:09,  6.69it/s][A
     63%|██████▎   | 796/1261 [02:29<01:23,  5.56it/s][A
     63%|██████▎   | 797/1261 [02:29<01:14,  6.26it/s][A
     63%|██████▎   | 798/1261 [02:29<01:13,  6.29it/s][A
     63%|██████▎   | 799/1261 [02:29<01:13,  6.29it/s][A
     63%|██████▎   | 800/1261 [02:30<01:13,  6.26it/s][A
     64%|██████▎   | 801/1261 [02:30<01:06,  6.90it/s][A
     64%|██████▎   | 802/1261 [02:30<01:04,  7.09it/s][A
     64%|██████▎   | 803/1261 [02:30<01:02,  7.34it/s][A
     64%|██████▍   | 804/1261 [02:30<01:04,  7.09it/s][A
     64%|██████▍   | 805/1261 [02:30<01:15,  6.06it/s][A
     64%|██████▍   | 806/1261 [02:31<01:18,  5.79it/s][A
     64%|██████▍   | 807/1261 [02:31<01:11,  6.32it/s][A
     64%|██████▍   | 808/1261 [02:31<01:05,  6.87it/s][A
     64%|██████▍   | 809/1261 [02:31<01:06,  6.76it/s][A
     64%|██████▍   | 810/1261 [02:31<01:10,  6.39it/s][A
     64%|██████▍   | 811/1261 [02:31<01:04,  7.02it/s][A
     64%|██████▍   | 812/1261 [02:31<00:59,  7.61it/s][A
     64%|██████▍   | 813/1261 [02:31<00:56,  7.88it/s][A
     65%|██████▍   | 814/1261 [02:32<00:54,  8.19it/s][A
     65%|██████▍   | 815/1261 [02:32<00:53,  8.29it/s][A
     65%|██████▍   | 816/1261 [02:32<00:53,  8.36it/s][A
     65%|██████▍   | 817/1261 [02:32<01:20,  5.55it/s][A
     65%|██████▍   | 818/1261 [02:32<01:18,  5.64it/s][A
     65%|██████▍   | 819/1261 [02:32<01:10,  6.29it/s][A
     65%|██████▌   | 820/1261 [02:33<01:03,  6.90it/s][A
     65%|██████▌   | 821/1261 [02:33<00:59,  7.34it/s][A
     65%|██████▌   | 822/1261 [02:33<00:57,  7.67it/s][A
     65%|██████▌   | 823/1261 [02:33<00:54,  7.99it/s][A
     65%|██████▌   | 824/1261 [02:33<00:53,  8.13it/s][A
     65%|██████▌   | 825/1261 [02:33<00:59,  7.35it/s][A
     66%|██████▌   | 826/1261 [02:33<00:56,  7.72it/s][A
     66%|██████▌   | 827/1261 [02:33<01:02,  6.90it/s][A
     66%|██████▌   | 828/1261 [02:34<01:11,  6.06it/s][A
     66%|██████▌   | 829/1261 [02:34<01:06,  6.46it/s][A
     66%|██████▌   | 830/1261 [02:34<01:12,  5.98it/s][A
     66%|██████▌   | 831/1261 [02:34<01:07,  6.39it/s][A
     66%|██████▌   | 832/1261 [02:34<01:02,  6.89it/s][A
     66%|██████▌   | 833/1261 [02:34<01:09,  6.16it/s][A
     66%|██████▌   | 834/1261 [02:35<01:05,  6.50it/s][A
     66%|██████▌   | 835/1261 [02:35<01:03,  6.74it/s][A
     66%|██████▋   | 836/1261 [02:35<01:04,  6.64it/s][A
     66%|██████▋   | 837/1261 [02:35<01:08,  6.21it/s][A
     66%|██████▋   | 838/1261 [02:35<01:01,  6.86it/s][A
     67%|██████▋   | 839/1261 [02:35<01:06,  6.37it/s][A
     67%|██████▋   | 840/1261 [02:36<01:04,  6.56it/s][A
     67%|██████▋   | 841/1261 [02:36<01:00,  6.89it/s][A
     67%|██████▋   | 842/1261 [02:36<00:59,  7.01it/s][A
     67%|██████▋   | 843/1261 [02:36<01:17,  5.40it/s][A
     67%|██████▋   | 844/1261 [02:36<01:15,  5.49it/s][A
     67%|██████▋   | 845/1261 [02:36<01:07,  6.15it/s][A
     67%|██████▋   | 846/1261 [02:36<01:00,  6.82it/s][A
     67%|██████▋   | 848/1261 [02:37<00:54,  7.59it/s][A
     67%|██████▋   | 849/1261 [02:37<00:50,  8.13it/s][A
     67%|██████▋   | 850/1261 [02:37<00:48,  8.48it/s][A
     67%|██████▋   | 851/1261 [02:37<01:12,  5.65it/s][A
     68%|██████▊   | 852/1261 [02:37<01:06,  6.17it/s][A
     68%|██████▊   | 853/1261 [02:37<00:59,  6.89it/s][A
     68%|██████▊   | 854/1261 [02:38<00:55,  7.38it/s][A
     68%|██████▊   | 855/1261 [02:38<00:51,  7.91it/s][A
     68%|██████▊   | 856/1261 [02:38<00:49,  8.20it/s][A
     68%|██████▊   | 857/1261 [02:38<01:12,  5.55it/s][A
     68%|██████▊   | 858/1261 [02:38<01:05,  6.12it/s][A
     68%|██████▊   | 859/1261 [02:38<00:59,  6.73it/s][A
     68%|██████▊   | 860/1261 [02:38<00:56,  7.08it/s][A
     68%|██████▊   | 861/1261 [02:39<00:54,  7.31it/s][A
     68%|██████▊   | 862/1261 [02:39<00:51,  7.72it/s][A
     68%|██████▊   | 863/1261 [02:39<01:12,  5.50it/s][A
     69%|██████▊   | 864/1261 [02:39<01:06,  5.94it/s][A
     69%|██████▊   | 865/1261 [02:39<01:02,  6.31it/s][A
     69%|██████▊   | 866/1261 [02:39<00:59,  6.64it/s][A
     69%|██████▉   | 867/1261 [02:39<00:55,  7.04it/s][A
     69%|██████▉   | 868/1261 [02:40<00:51,  7.69it/s][A
     69%|██████▉   | 870/1261 [02:40<00:49,  7.87it/s][A
     69%|██████▉   | 871/1261 [02:40<00:47,  8.24it/s][A
     69%|██████▉   | 872/1261 [02:40<00:46,  8.35it/s][A
     69%|██████▉   | 873/1261 [02:40<00:47,  8.25it/s][A
     69%|██████▉   | 874/1261 [02:40<00:47,  8.11it/s][A
     69%|██████▉   | 875/1261 [02:40<00:47,  8.10it/s][A
     69%|██████▉   | 876/1261 [02:41<00:47,  8.13it/s][A
     70%|██████▉   | 877/1261 [02:41<01:08,  5.59it/s][A
     70%|██████▉   | 878/1261 [02:41<01:06,  5.78it/s][A
     70%|██████▉   | 879/1261 [02:41<00:59,  6.44it/s][A
     70%|██████▉   | 880/1261 [02:41<00:53,  7.12it/s][A
     70%|██████▉   | 881/1261 [02:41<00:51,  7.37it/s][A
     70%|██████▉   | 882/1261 [02:41<00:48,  7.81it/s][A
     70%|███████   | 883/1261 [02:42<01:07,  5.62it/s][A
     70%|███████   | 884/1261 [02:42<01:02,  6.02it/s][A
     70%|███████   | 885/1261 [02:42<00:56,  6.65it/s][A
     70%|███████   | 886/1261 [02:42<00:51,  7.28it/s][A
     70%|███████   | 887/1261 [02:42<00:49,  7.56it/s][A
     70%|███████   | 888/1261 [02:42<00:47,  7.79it/s][A
     70%|███████   | 889/1261 [02:43<01:07,  5.49it/s][A
     71%|███████   | 890/1261 [02:43<01:02,  5.97it/s][A
     71%|███████   | 891/1261 [02:43<00:58,  6.38it/s][A
     71%|███████   | 892/1261 [02:43<00:52,  7.06it/s][A
     71%|███████   | 893/1261 [02:43<00:50,  7.29it/s][A
     71%|███████   | 894/1261 [02:43<00:50,  7.27it/s][A
     71%|███████   | 895/1261 [02:44<01:07,  5.39it/s][A
     71%|███████   | 896/1261 [02:44<01:12,  5.02it/s][A
     71%|███████   | 897/1261 [02:44<01:16,  4.78it/s][A
     71%|███████   | 898/1261 [02:44<01:09,  5.24it/s][A
     71%|███████▏  | 899/1261 [02:45<01:18,  4.63it/s][A
     71%|███████▏  | 900/1261 [02:45<01:07,  5.34it/s][A
     71%|███████▏  | 901/1261 [02:45<01:34,  3.81it/s][A
     72%|███████▏  | 902/1261 [02:45<01:30,  3.97it/s][A
     72%|███████▏  | 903/1261 [02:45<01:22,  4.34it/s][A
     72%|███████▏  | 904/1261 [02:46<01:13,  4.84it/s][A
     72%|███████▏  | 905/1261 [02:46<01:06,  5.39it/s][A
     72%|███████▏  | 906/1261 [02:46<01:00,  5.88it/s][A
     72%|███████▏  | 907/1261 [02:46<00:56,  6.23it/s][A
     72%|███████▏  | 908/1261 [02:46<00:59,  5.91it/s][A
     72%|███████▏  | 909/1261 [02:47<01:26,  4.07it/s][A
     72%|███████▏  | 910/1261 [02:47<01:17,  4.54it/s][A
     72%|███████▏  | 911/1261 [02:47<01:05,  5.38it/s][A
     72%|███████▏  | 912/1261 [02:47<00:57,  6.12it/s][A
     72%|███████▏  | 913/1261 [02:47<00:51,  6.74it/s][A
     73%|███████▎  | 915/1261 [02:48<00:57,  6.04it/s][A
     73%|███████▎  | 916/1261 [02:48<01:00,  5.68it/s][A
     73%|███████▎  | 918/1261 [02:48<00:55,  6.16it/s][A
     73%|███████▎  | 919/1261 [02:48<00:51,  6.70it/s][A
     73%|███████▎  | 920/1261 [02:48<00:46,  7.29it/s][A
     73%|███████▎  | 921/1261 [02:48<00:44,  7.59it/s][A
     73%|███████▎  | 922/1261 [02:49<00:49,  6.88it/s][A
     73%|███████▎  | 923/1261 [02:49<00:57,  5.84it/s][A
     73%|███████▎  | 924/1261 [02:49<00:50,  6.62it/s][A
     73%|███████▎  | 926/1261 [02:49<00:44,  7.45it/s][A
     74%|███████▎  | 927/1261 [02:49<00:50,  6.57it/s][A
     74%|███████▎  | 928/1261 [02:49<00:51,  6.47it/s][A
     74%|███████▍  | 930/1261 [02:50<00:48,  6.82it/s][A
     74%|███████▍  | 931/1261 [02:50<00:44,  7.38it/s][A
     74%|███████▍  | 932/1261 [02:50<00:44,  7.44it/s][A
     74%|███████▍  | 933/1261 [02:50<00:51,  6.35it/s][A
     74%|███████▍  | 934/1261 [02:50<00:49,  6.56it/s][A
     74%|███████▍  | 935/1261 [02:50<00:51,  6.34it/s][A
     74%|███████▍  | 936/1261 [02:51<00:50,  6.46it/s][A
     74%|███████▍  | 937/1261 [02:51<00:47,  6.76it/s][A
     74%|███████▍  | 938/1261 [02:51<00:54,  5.96it/s][A
     74%|███████▍  | 939/1261 [02:51<00:50,  6.38it/s][A
     75%|███████▍  | 940/1261 [02:51<00:47,  6.81it/s][A
     75%|███████▍  | 941/1261 [02:51<00:45,  6.97it/s][A
     75%|███████▍  | 942/1261 [02:51<00:41,  7.65it/s][A
     75%|███████▍  | 943/1261 [02:52<00:40,  7.86it/s][A
     75%|███████▍  | 944/1261 [02:52<00:41,  7.65it/s][A
     75%|███████▍  | 945/1261 [02:52<00:46,  6.74it/s][A
     75%|███████▌  | 946/1261 [02:52<00:49,  6.40it/s][A
     75%|███████▌  | 947/1261 [02:52<00:45,  6.93it/s][A
     75%|███████▌  | 948/1261 [02:52<00:41,  7.46it/s][A
     75%|███████▌  | 949/1261 [02:52<00:38,  8.04it/s][A
     75%|███████▌  | 950/1261 [02:52<00:37,  8.36it/s][A
     75%|███████▌  | 951/1261 [02:53<00:54,  5.69it/s][A
     75%|███████▌  | 952/1261 [02:53<00:54,  5.66it/s][A
     76%|███████▌  | 953/1261 [02:53<00:47,  6.48it/s][A
     76%|███████▌  | 954/1261 [02:53<00:43,  7.10it/s][A
     76%|███████▌  | 955/1261 [02:53<00:39,  7.69it/s][A
     76%|███████▌  | 956/1261 [02:53<00:38,  7.93it/s][A
     76%|███████▌  | 957/1261 [02:53<00:36,  8.44it/s][A
     76%|███████▌  | 958/1261 [02:54<00:44,  6.84it/s][A
     76%|███████▌  | 959/1261 [02:54<00:41,  7.30it/s][A
     76%|███████▌  | 960/1261 [02:54<00:47,  6.34it/s][A
     76%|███████▌  | 961/1261 [02:54<00:44,  6.71it/s][A
     76%|███████▋  | 962/1261 [02:54<00:41,  7.23it/s][A
     76%|███████▋  | 963/1261 [02:55<00:55,  5.36it/s][A
     76%|███████▋  | 964/1261 [02:55<00:54,  5.47it/s][A
     77%|███████▋  | 965/1261 [02:55<00:48,  6.14it/s][A
     77%|███████▋  | 966/1261 [02:55<00:44,  6.63it/s][A
     77%|███████▋  | 967/1261 [02:55<00:40,  7.19it/s][A
     77%|███████▋  | 968/1261 [02:55<00:37,  7.79it/s][A
     77%|███████▋  | 969/1261 [02:55<00:37,  7.69it/s][A
     77%|███████▋  | 970/1261 [02:56<00:42,  6.78it/s][A
     77%|███████▋  | 971/1261 [02:56<00:48,  5.97it/s][A
     77%|███████▋  | 972/1261 [02:56<00:46,  6.17it/s][A
     77%|███████▋  | 973/1261 [02:56<00:42,  6.85it/s][A
     77%|███████▋  | 974/1261 [02:56<00:38,  7.37it/s][A
     77%|███████▋  | 975/1261 [02:56<00:48,  5.91it/s][A
     77%|███████▋  | 976/1261 [02:57<00:47,  6.05it/s][A
     77%|███████▋  | 977/1261 [02:57<00:41,  6.81it/s][A
     78%|███████▊  | 978/1261 [02:57<00:39,  7.14it/s][A
     78%|███████▊  | 979/1261 [02:57<00:54,  5.17it/s][A
     78%|███████▊  | 980/1261 [02:57<00:53,  5.23it/s][A
     78%|███████▊  | 981/1261 [02:57<00:46,  6.09it/s][A
     78%|███████▊  | 982/1261 [02:57<00:41,  6.66it/s][A
     78%|███████▊  | 983/1261 [02:58<00:39,  7.01it/s][A
     78%|███████▊  | 984/1261 [02:58<00:37,  7.41it/s][A
     78%|███████▊  | 985/1261 [02:58<00:35,  7.73it/s][A
     78%|███████▊  | 986/1261 [02:58<00:44,  6.17it/s][A
     78%|███████▊  | 987/1261 [02:58<00:46,  5.91it/s][A
     78%|███████▊  | 988/1261 [02:58<00:42,  6.44it/s][A
     78%|███████▊  | 989/1261 [02:58<00:39,  6.96it/s][A
     79%|███████▊  | 990/1261 [02:59<00:39,  6.88it/s][A
     79%|███████▊  | 991/1261 [02:59<00:49,  5.51it/s][A
     79%|███████▊  | 992/1261 [02:59<00:55,  4.87it/s][A
     79%|███████▊  | 993/1261 [02:59<00:51,  5.18it/s][A
     79%|███████▉  | 994/1261 [02:59<00:50,  5.33it/s][A
     79%|███████▉  | 995/1261 [03:00<00:45,  5.89it/s][A
     79%|███████▉  | 996/1261 [03:00<00:39,  6.67it/s][A
     79%|███████▉  | 997/1261 [03:00<00:37,  6.97it/s][A
     79%|███████▉  | 998/1261 [03:00<00:50,  5.25it/s][A
     79%|███████▉  | 999/1261 [03:00<00:47,  5.48it/s][A
     79%|███████▉  | 1000/1261 [03:00<00:45,  5.69it/s][A
     79%|███████▉  | 1001/1261 [03:01<00:41,  6.29it/s][A
     79%|███████▉  | 1002/1261 [03:01<00:39,  6.58it/s][A
     80%|███████▉  | 1003/1261 [03:01<00:37,  6.96it/s][A
     80%|███████▉  | 1004/1261 [03:01<00:34,  7.48it/s][A
     80%|███████▉  | 1005/1261 [03:01<00:32,  7.76it/s][A
     80%|███████▉  | 1006/1261 [03:01<00:38,  6.66it/s][A
     80%|███████▉  | 1007/1261 [03:01<00:40,  6.25it/s][A
     80%|███████▉  | 1008/1261 [03:02<00:41,  6.11it/s][A
     80%|████████  | 1009/1261 [03:02<00:38,  6.61it/s][A
     80%|████████  | 1010/1261 [03:02<00:37,  6.69it/s][A
     80%|████████  | 1011/1261 [03:02<00:35,  7.11it/s][A
     80%|████████  | 1012/1261 [03:02<00:45,  5.53it/s][A
     80%|████████  | 1013/1261 [03:02<00:44,  5.58it/s][A
     80%|████████  | 1014/1261 [03:03<00:41,  5.99it/s][A
     80%|████████  | 1015/1261 [03:03<00:50,  4.90it/s][A
     81%|████████  | 1016/1261 [03:03<00:49,  4.97it/s][A
     81%|████████  | 1017/1261 [03:03<00:44,  5.48it/s][A
     81%|████████  | 1018/1261 [03:03<00:39,  6.16it/s][A
     81%|████████  | 1019/1261 [03:03<00:36,  6.69it/s][A
     81%|████████  | 1020/1261 [03:04<00:33,  7.10it/s][A
     81%|████████  | 1021/1261 [03:04<00:31,  7.59it/s][A
     81%|████████  | 1022/1261 [03:04<00:30,  7.79it/s][A
     81%|████████  | 1023/1261 [03:04<00:46,  5.07it/s][A
     81%|████████  | 1024/1261 [03:04<00:46,  5.07it/s][A
     81%|████████▏ | 1025/1261 [03:04<00:40,  5.79it/s][A
     81%|████████▏ | 1026/1261 [03:05<00:36,  6.41it/s][A
     81%|████████▏ | 1027/1261 [03:05<00:33,  6.90it/s][A
     82%|████████▏ | 1028/1261 [03:05<00:31,  7.32it/s][A
     82%|████████▏ | 1029/1261 [03:05<00:41,  5.65it/s][A
     82%|████████▏ | 1030/1261 [03:05<00:37,  6.13it/s][A
     82%|████████▏ | 1031/1261 [03:06<00:48,  4.70it/s][A
     82%|████████▏ | 1032/1261 [03:06<00:44,  5.16it/s][A
     82%|████████▏ | 1033/1261 [03:06<00:39,  5.80it/s][A
     82%|████████▏ | 1034/1261 [03:06<00:34,  6.61it/s][A
     82%|████████▏ | 1035/1261 [03:06<00:31,  7.16it/s][A
     82%|████████▏ | 1036/1261 [03:06<00:30,  7.35it/s][A
     82%|████████▏ | 1037/1261 [03:07<00:43,  5.12it/s][A
     82%|████████▏ | 1038/1261 [03:07<00:41,  5.41it/s][A
     82%|████████▏ | 1039/1261 [03:07<00:35,  6.18it/s][A
     82%|████████▏ | 1040/1261 [03:07<00:33,  6.67it/s][A
     83%|████████▎ | 1041/1261 [03:07<00:31,  6.95it/s][A
     83%|████████▎ | 1042/1261 [03:07<00:30,  7.14it/s][A
     83%|████████▎ | 1043/1261 [03:07<00:29,  7.29it/s][A
     83%|████████▎ | 1044/1261 [03:08<00:41,  5.19it/s][A
     83%|████████▎ | 1045/1261 [03:08<00:38,  5.61it/s][A
     83%|████████▎ | 1046/1261 [03:08<00:34,  6.21it/s][A
     83%|████████▎ | 1047/1261 [03:08<00:36,  5.91it/s][A
     83%|████████▎ | 1048/1261 [03:08<00:35,  5.95it/s][A
     83%|████████▎ | 1049/1261 [03:08<00:33,  6.30it/s][A
     83%|████████▎ | 1050/1261 [03:09<00:41,  5.05it/s][A
     83%|████████▎ | 1051/1261 [03:09<00:37,  5.62it/s][A
     83%|████████▎ | 1052/1261 [03:09<00:33,  6.32it/s][A
     84%|████████▎ | 1053/1261 [03:09<00:30,  6.84it/s][A
     84%|████████▎ | 1054/1261 [03:09<00:28,  7.37it/s][A
     84%|████████▎ | 1055/1261 [03:09<00:33,  6.09it/s][A
     84%|████████▎ | 1056/1261 [03:10<00:35,  5.84it/s][A
     84%|████████▍ | 1057/1261 [03:10<00:34,  5.94it/s][A
     84%|████████▍ | 1058/1261 [03:10<00:41,  4.90it/s][A
     84%|████████▍ | 1059/1261 [03:10<00:36,  5.49it/s][A
     84%|████████▍ | 1060/1261 [03:10<00:32,  6.23it/s][A
     84%|████████▍ | 1061/1261 [03:10<00:31,  6.45it/s][A
     84%|████████▍ | 1062/1261 [03:11<00:30,  6.58it/s][A
     84%|████████▍ | 1063/1261 [03:11<00:32,  6.16it/s][A
     84%|████████▍ | 1064/1261 [03:11<00:43,  4.52it/s][A
     84%|████████▍ | 1065/1261 [03:11<00:38,  5.04it/s][A
     85%|████████▍ | 1066/1261 [03:11<00:34,  5.72it/s][A
     85%|████████▍ | 1067/1261 [03:11<00:30,  6.45it/s][A
     85%|████████▍ | 1068/1261 [03:12<00:27,  7.15it/s][A
     85%|████████▍ | 1069/1261 [03:12<00:25,  7.64it/s][A
     85%|████████▍ | 1070/1261 [03:12<00:27,  6.92it/s][A
     85%|████████▍ | 1071/1261 [03:12<00:38,  4.90it/s][A
     85%|████████▌ | 1072/1261 [03:12<00:36,  5.21it/s][A
     85%|████████▌ | 1073/1261 [03:13<00:33,  5.57it/s][A
     85%|████████▌ | 1074/1261 [03:13<00:31,  5.93it/s][A
     85%|████████▌ | 1075/1261 [03:13<00:28,  6.48it/s][A
     85%|████████▌ | 1076/1261 [03:13<00:27,  6.73it/s][A
     85%|████████▌ | 1077/1261 [03:13<00:24,  7.40it/s][A
     85%|████████▌ | 1078/1261 [03:13<00:33,  5.50it/s][A
     86%|████████▌ | 1079/1261 [03:14<00:35,  5.16it/s][A
     86%|████████▌ | 1080/1261 [03:14<00:32,  5.59it/s][A
     86%|████████▌ | 1081/1261 [03:14<00:33,  5.36it/s][A
     86%|████████▌ | 1082/1261 [03:14<00:32,  5.56it/s][A
     86%|████████▌ | 1083/1261 [03:14<00:29,  6.06it/s][A
     86%|████████▌ | 1084/1261 [03:14<00:37,  4.70it/s][A
     86%|████████▌ | 1085/1261 [03:15<00:38,  4.55it/s][A
     86%|████████▌ | 1086/1261 [03:15<00:32,  5.40it/s][A
     86%|████████▌ | 1087/1261 [03:15<00:27,  6.23it/s][A
     86%|████████▋ | 1088/1261 [03:15<00:25,  6.89it/s][A
     86%|████████▋ | 1089/1261 [03:15<00:22,  7.54it/s][A
     87%|████████▋ | 1091/1261 [03:16<00:26,  6.32it/s][A
     87%|████████▋ | 1092/1261 [03:16<00:29,  5.68it/s][A
     87%|████████▋ | 1093/1261 [03:16<00:25,  6.49it/s][A
     87%|████████▋ | 1094/1261 [03:16<00:23,  7.19it/s][A
     87%|████████▋ | 1096/1261 [03:16<00:20,  7.96it/s][A
     87%|████████▋ | 1097/1261 [03:17<00:31,  5.28it/s][A
     87%|████████▋ | 1098/1261 [03:17<00:28,  5.64it/s][A
     87%|████████▋ | 1099/1261 [03:17<00:26,  6.17it/s][A
     87%|████████▋ | 1100/1261 [03:17<00:24,  6.56it/s][A
     87%|████████▋ | 1101/1261 [03:17<00:30,  5.28it/s][A
     87%|████████▋ | 1102/1261 [03:17<00:30,  5.26it/s][A
     88%|████████▊ | 1104/1261 [03:18<00:25,  6.14it/s][A
     88%|████████▊ | 1106/1261 [03:18<00:22,  6.86it/s][A
     88%|████████▊ | 1108/1261 [03:18<00:25,  6.11it/s][A
     88%|████████▊ | 1109/1261 [03:18<00:25,  5.94it/s][A
     88%|████████▊ | 1110/1261 [03:19<00:22,  6.73it/s][A
     88%|████████▊ | 1112/1261 [03:19<00:19,  7.61it/s][A
     88%|████████▊ | 1113/1261 [03:19<00:20,  7.29it/s][A
     88%|████████▊ | 1114/1261 [03:19<00:19,  7.62it/s][A
     88%|████████▊ | 1115/1261 [03:19<00:18,  7.85it/s][A
     89%|████████▊ | 1116/1261 [03:19<00:24,  5.91it/s][A
     89%|████████▊ | 1117/1261 [03:20<00:24,  5.83it/s][A
     89%|████████▊ | 1119/1261 [03:20<00:21,  6.56it/s][A
     89%|████████▉ | 1120/1261 [03:20<00:19,  7.15it/s][A
     89%|████████▉ | 1121/1261 [03:20<00:18,  7.41it/s][A
     89%|████████▉ | 1122/1261 [03:20<00:17,  7.90it/s][A
     89%|████████▉ | 1123/1261 [03:20<00:17,  7.86it/s][A
     89%|████████▉ | 1124/1261 [03:21<00:26,  5.16it/s][A
     89%|████████▉ | 1125/1261 [03:21<00:25,  5.42it/s][A
     89%|████████▉ | 1126/1261 [03:21<00:22,  6.06it/s][A
     89%|████████▉ | 1127/1261 [03:21<00:19,  6.70it/s][A
     90%|████████▉ | 1129/1261 [03:21<00:17,  7.47it/s][A
     90%|████████▉ | 1130/1261 [03:21<00:16,  8.08it/s][A
     90%|████████▉ | 1131/1261 [03:21<00:15,  8.27it/s][A
     90%|████████▉ | 1132/1261 [03:22<00:23,  5.41it/s][A
     90%|████████▉ | 1133/1261 [03:22<00:22,  5.65it/s][A
     90%|████████▉ | 1134/1261 [03:22<00:19,  6.49it/s][A
     90%|█████████ | 1135/1261 [03:22<00:17,  7.25it/s][A
     90%|█████████ | 1136/1261 [03:22<00:16,  7.63it/s][A
     90%|█████████ | 1137/1261 [03:22<00:15,  8.07it/s][A
     90%|█████████ | 1138/1261 [03:22<00:14,  8.40it/s][A
     90%|█████████ | 1139/1261 [03:23<00:21,  5.63it/s][A
     90%|█████████ | 1140/1261 [03:23<00:21,  5.75it/s][A
     90%|█████████ | 1141/1261 [03:23<00:19,  6.31it/s][A
     91%|█████████ | 1143/1261 [03:23<00:16,  7.07it/s][A
     91%|█████████ | 1144/1261 [03:23<00:15,  7.72it/s][A
     91%|█████████ | 1145/1261 [03:23<00:14,  7.88it/s][A
     91%|█████████ | 1146/1261 [03:24<00:17,  6.39it/s][A
     91%|█████████ | 1147/1261 [03:24<00:19,  5.98it/s][A
     91%|█████████ | 1148/1261 [03:24<00:17,  6.57it/s][A
     91%|█████████ | 1149/1261 [03:24<00:15,  7.10it/s][A
     91%|█████████ | 1150/1261 [03:24<00:14,  7.62it/s][A
     91%|█████████▏| 1151/1261 [03:24<00:13,  8.07it/s][A
     91%|█████████▏| 1152/1261 [03:24<00:12,  8.49it/s][A
     91%|█████████▏| 1153/1261 [03:25<00:20,  5.24it/s][A
     92%|█████████▏| 1154/1261 [03:25<00:18,  5.86it/s][A
     92%|█████████▏| 1155/1261 [03:25<00:16,  6.48it/s][A
     92%|█████████▏| 1156/1261 [03:25<00:14,  7.04it/s][A
     92%|█████████▏| 1157/1261 [03:25<00:13,  7.69it/s][A
     92%|█████████▏| 1158/1261 [03:25<00:12,  7.99it/s][A
     92%|█████████▏| 1159/1261 [03:25<00:12,  7.97it/s][A
     92%|█████████▏| 1160/1261 [03:26<00:17,  5.69it/s][A
     92%|█████████▏| 1161/1261 [03:26<00:17,  5.58it/s][A
     92%|█████████▏| 1162/1261 [03:26<00:15,  6.30it/s][A
     92%|█████████▏| 1163/1261 [03:26<00:14,  6.89it/s][A
     92%|█████████▏| 1164/1261 [03:26<00:12,  7.46it/s][A
     92%|█████████▏| 1165/1261 [03:27<00:16,  5.69it/s][A
     92%|█████████▏| 1166/1261 [03:27<00:17,  5.58it/s][A
     93%|█████████▎| 1168/1261 [03:27<00:14,  6.41it/s][A
     93%|█████████▎| 1169/1261 [03:27<00:13,  7.08it/s][A
     93%|█████████▎| 1170/1261 [03:27<00:12,  7.54it/s][A
     93%|█████████▎| 1171/1261 [03:27<00:11,  8.00it/s][A
     93%|█████████▎| 1172/1261 [03:28<00:16,  5.52it/s][A
     93%|█████████▎| 1173/1261 [03:28<00:14,  6.02it/s][A
     93%|█████████▎| 1175/1261 [03:28<00:12,  6.88it/s][A
     93%|█████████▎| 1176/1261 [03:28<00:15,  5.45it/s][A
     93%|█████████▎| 1177/1261 [03:28<00:13,  6.08it/s][A
     93%|█████████▎| 1178/1261 [03:28<00:12,  6.73it/s][A
     93%|█████████▎| 1179/1261 [03:28<00:11,  7.34it/s][A
     94%|█████████▎| 1180/1261 [03:29<00:10,  7.73it/s][A
     94%|█████████▎| 1181/1261 [03:29<00:10,  7.94it/s][A
     94%|█████████▎| 1182/1261 [03:29<00:14,  5.63it/s][A
     94%|█████████▍| 1183/1261 [03:29<00:13,  5.75it/s][A
     94%|█████████▍| 1184/1261 [03:29<00:11,  6.56it/s][A
     94%|█████████▍| 1185/1261 [03:29<00:10,  7.00it/s][A
     94%|█████████▍| 1186/1261 [03:30<00:10,  7.30it/s][A
     94%|█████████▍| 1187/1261 [03:30<00:11,  6.41it/s][A
     94%|█████████▍| 1188/1261 [03:30<00:10,  6.72it/s][A
     94%|█████████▍| 1189/1261 [03:30<00:09,  7.28it/s][A
     94%|█████████▍| 1190/1261 [03:30<00:09,  7.72it/s][A
     94%|█████████▍| 1191/1261 [03:30<00:09,  7.62it/s][A
     95%|█████████▍| 1192/1261 [03:30<00:09,  7.11it/s][A
     95%|█████████▍| 1193/1261 [03:31<00:09,  7.31it/s][A
     95%|█████████▍| 1194/1261 [03:31<00:09,  6.86it/s][A
     95%|█████████▍| 1195/1261 [03:31<00:09,  7.15it/s][A
     95%|█████████▍| 1196/1261 [03:31<00:09,  6.54it/s][A
     95%|█████████▍| 1197/1261 [03:31<00:09,  6.64it/s][A
     95%|█████████▌| 1198/1261 [03:31<00:10,  6.20it/s][A
     95%|█████████▌| 1199/1261 [03:31<00:08,  6.90it/s][A
     95%|█████████▌| 1200/1261 [03:32<00:08,  6.91it/s][A
     95%|█████████▌| 1201/1261 [03:32<00:08,  7.06it/s][A
     95%|█████████▌| 1202/1261 [03:32<00:08,  6.97it/s][A
     95%|█████████▌| 1203/1261 [03:32<00:08,  7.18it/s][A
     95%|█████████▌| 1204/1261 [03:32<00:10,  5.49it/s][A
     96%|█████████▌| 1205/1261 [03:32<00:10,  5.41it/s][A
     96%|█████████▌| 1206/1261 [03:33<00:09,  6.06it/s][A
     96%|█████████▌| 1207/1261 [03:33<00:08,  6.73it/s][A
     96%|█████████▌| 1208/1261 [03:33<00:07,  7.12it/s][A
     96%|█████████▌| 1209/1261 [03:33<00:07,  6.81it/s][A
     96%|█████████▌| 1210/1261 [03:33<00:06,  7.33it/s][A
     96%|█████████▌| 1211/1261 [03:33<00:06,  7.93it/s][A
     96%|█████████▌| 1212/1261 [03:33<00:06,  7.27it/s][A
     96%|█████████▌| 1213/1261 [03:33<00:06,  7.70it/s][A
     96%|█████████▋| 1214/1261 [03:34<00:07,  6.39it/s][A
     96%|█████████▋| 1215/1261 [03:34<00:07,  6.13it/s][A
     96%|█████████▋| 1216/1261 [03:34<00:08,  5.47it/s][A
     97%|█████████▋| 1217/1261 [03:34<00:07,  5.87it/s][A
     97%|█████████▋| 1218/1261 [03:34<00:06,  6.62it/s][A
     97%|█████████▋| 1219/1261 [03:34<00:05,  7.19it/s][A
     97%|█████████▋| 1220/1261 [03:35<00:05,  7.27it/s][A
     97%|█████████▋| 1221/1261 [03:35<00:05,  7.52it/s][A
     97%|█████████▋| 1222/1261 [03:35<00:07,  5.23it/s][A
     97%|█████████▋| 1223/1261 [03:35<00:06,  5.75it/s][A
     97%|█████████▋| 1224/1261 [03:35<00:06,  6.15it/s][A
     97%|█████████▋| 1225/1261 [03:35<00:05,  6.87it/s][A
     97%|█████████▋| 1226/1261 [03:36<00:04,  7.35it/s][A
     97%|█████████▋| 1227/1261 [03:36<00:04,  7.87it/s][A
     97%|█████████▋| 1228/1261 [03:36<00:04,  7.81it/s][A
     97%|█████████▋| 1229/1261 [03:36<00:05,  6.01it/s][A
     98%|█████████▊| 1230/1261 [03:36<00:04,  6.36it/s][A
     98%|█████████▊| 1231/1261 [03:36<00:04,  7.00it/s][A
     98%|█████████▊| 1232/1261 [03:37<00:05,  5.76it/s][A
     98%|█████████▊| 1233/1261 [03:37<00:04,  6.03it/s][A
     98%|█████████▊| 1234/1261 [03:37<00:04,  6.59it/s][A
     98%|█████████▊| 1235/1261 [03:37<00:03,  6.71it/s][A
     98%|█████████▊| 1236/1261 [03:37<00:04,  5.78it/s][A
     98%|█████████▊| 1237/1261 [03:37<00:04,  5.56it/s][A
     98%|█████████▊| 1239/1261 [03:38<00:03,  6.37it/s][A
     98%|█████████▊| 1241/1261 [03:38<00:02,  7.11it/s][A
     98%|█████████▊| 1242/1261 [03:38<00:03,  5.89it/s][A
     99%|█████████▊| 1243/1261 [03:38<00:02,  6.07it/s][A
     99%|█████████▊| 1244/1261 [03:38<00:02,  6.51it/s][A
     99%|█████████▊| 1245/1261 [03:38<00:02,  6.86it/s][A
     99%|█████████▉| 1246/1261 [03:39<00:02,  5.13it/s][A
     99%|█████████▉| 1247/1261 [03:39<00:02,  5.25it/s][A
     99%|█████████▉| 1248/1261 [03:39<00:02,  6.06it/s][A
     99%|█████████▉| 1249/1261 [03:39<00:01,  6.68it/s][A
     99%|█████████▉| 1250/1261 [03:39<00:01,  7.25it/s][A
     99%|█████████▉| 1251/1261 [03:39<00:01,  7.73it/s][A
     99%|█████████▉| 1252/1261 [03:39<00:01,  8.07it/s][A
     99%|█████████▉| 1253/1261 [03:40<00:01,  5.51it/s][A
     99%|█████████▉| 1254/1261 [03:40<00:01,  5.37it/s][A
    100%|█████████▉| 1255/1261 [03:40<00:00,  6.15it/s][A
    100%|█████████▉| 1257/1261 [03:40<00:00,  6.98it/s][A
    100%|█████████▉| 1259/1261 [03:40<00:00,  7.79it/s][A
    100%|█████████▉| 1260/1261 [03:41<00:00,  7.98it/s][A
    [A

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test_videos_output/project_video.mp4 
    
    CPU times: user 9min 55s, sys: 13.5 s, total: 10min 9s
    Wall time: 3min 42s


Essentially all lane lines of the video *project_video.mp4* are detected correctly by the algorithm. The algorithm is able to handle variations in brightness as well as shadows on the road. In particular there is no catastrophic failure of the pipline which whould cause the car to leave its current lane and potentially crash. 

The resulting video was uploaded on youtube (see link below). The region of the image identified by the algorithm as actual road (bordered by the detected lane lines) is highlighted in green. 


```python
%time process_video('challenge_video.mp4', 'challenge_video.mp4') 
```

    [MoviePy] >>>> Building video test_videos_output/challenge_video.mp4
    [MoviePy] Writing video test_videos_output/challenge_video.mp4


    
      0%|          | 0/485 [00:00<?, ?it/s][A
      0%|          | 2/485 [00:00<01:49,  4.41it/s][A
      1%|          | 3/485 [00:00<01:33,  5.16it/s][A
      1%|          | 5/485 [00:00<01:17,  6.17it/s][A
      1%|▏         | 7/485 [00:00<01:05,  7.26it/s][A
      2%|▏         | 9/485 [00:01<00:57,  8.32it/s][A
      2%|▏         | 11/485 [00:01<00:51,  9.24it/s][A
      3%|▎         | 13/485 [00:01<00:52,  9.02it/s][A
      3%|▎         | 15/485 [00:01<00:47,  9.81it/s][A
      4%|▎         | 17/485 [00:01<00:45, 10.33it/s][A
      4%|▍         | 19/485 [00:01<00:43, 10.82it/s][A
      4%|▍         | 21/485 [00:02<00:41, 11.22it/s][A
      5%|▍         | 23/485 [00:02<00:40, 11.55it/s][A
      5%|▌         | 25/485 [00:02<00:39, 11.76it/s][A
      6%|▌         | 27/485 [00:02<00:39, 11.73it/s][A
      6%|▌         | 29/485 [00:02<00:38, 11.88it/s][A
      6%|▋         | 31/485 [00:02<00:37, 12.08it/s][A
      7%|▋         | 33/485 [00:03<00:36, 12.31it/s][A
      7%|▋         | 35/485 [00:03<00:36, 12.48it/s][A
      8%|▊         | 37/485 [00:03<00:35, 12.59it/s][A
      8%|▊         | 39/485 [00:03<00:35, 12.59it/s][A
      8%|▊         | 41/485 [00:03<00:35, 12.64it/s][A
      9%|▉         | 43/485 [00:03<00:39, 11.27it/s][A
      9%|▉         | 45/485 [00:04<00:48,  9.02it/s][A
      9%|▉         | 46/485 [00:04<00:47,  9.22it/s][A
     10%|▉         | 48/485 [00:04<00:48,  9.09it/s][A
     10%|█         | 49/485 [00:04<00:47,  9.13it/s][A
     10%|█         | 50/485 [00:04<00:47,  9.23it/s][A
     11%|█         | 52/485 [00:04<00:44,  9.65it/s][A
     11%|█         | 53/485 [00:05<00:50,  8.53it/s][A
     11%|█         | 54/485 [00:05<01:01,  7.04it/s][A
     11%|█▏        | 55/485 [00:05<00:58,  7.32it/s][A
     12%|█▏        | 56/485 [00:05<00:55,  7.67it/s][A
     12%|█▏        | 58/485 [00:05<00:51,  8.28it/s][A
     12%|█▏        | 59/485 [00:05<00:54,  7.76it/s][A
     12%|█▏        | 60/485 [00:06<00:58,  7.25it/s][A
     13%|█▎        | 61/485 [00:06<00:55,  7.67it/s][A
     13%|█▎        | 62/485 [00:06<00:57,  7.38it/s][A
     13%|█▎        | 63/485 [00:06<00:53,  7.93it/s][A
     13%|█▎        | 64/485 [00:06<00:53,  7.88it/s][A
     13%|█▎        | 65/485 [00:06<00:59,  7.09it/s][A
     14%|█▍        | 67/485 [00:06<00:52,  7.96it/s][A
     14%|█▍        | 68/485 [00:07<00:50,  8.25it/s][A
     14%|█▍        | 70/485 [00:07<00:47,  8.80it/s][A
     15%|█▍        | 71/485 [00:07<01:09,  5.95it/s][A
     15%|█▍        | 72/485 [00:07<01:12,  5.70it/s][A
     15%|█▌        | 74/485 [00:07<01:02,  6.57it/s][A
     16%|█▌        | 76/485 [00:08<00:55,  7.39it/s][A
     16%|█▌        | 77/485 [00:08<00:52,  7.72it/s][A
     16%|█▌        | 78/485 [00:08<01:10,  5.75it/s][A
     16%|█▋        | 79/485 [00:08<01:05,  6.22it/s][A
     16%|█▋        | 80/485 [00:08<00:59,  6.80it/s][A
     17%|█▋        | 81/485 [00:08<00:55,  7.32it/s][A
     17%|█▋        | 83/485 [00:09<00:49,  8.18it/s][A
     17%|█▋        | 84/485 [00:09<00:58,  6.81it/s][A
     18%|█▊        | 85/485 [00:09<00:53,  7.48it/s][A
     18%|█▊        | 86/485 [00:09<00:50,  7.85it/s][A
     18%|█▊        | 87/485 [00:09<01:08,  5.77it/s][A
     18%|█▊        | 88/485 [00:09<01:01,  6.43it/s][A
     18%|█▊        | 89/485 [00:10<00:59,  6.60it/s][A
     19%|█▊        | 90/485 [00:10<01:19,  4.95it/s][A
     19%|█▉        | 91/485 [00:10<01:14,  5.26it/s][A
     19%|█▉        | 92/485 [00:10<01:06,  5.90it/s][A
     19%|█▉        | 93/485 [00:10<01:02,  6.32it/s][A
     20%|█▉        | 95/485 [00:10<00:53,  7.31it/s][A
     20%|█▉        | 96/485 [00:11<01:03,  6.09it/s][A
     20%|██        | 97/485 [00:11<01:02,  6.16it/s][A
     20%|██        | 98/485 [00:11<01:08,  5.62it/s][A
     20%|██        | 99/485 [00:11<01:00,  6.35it/s][A
     21%|██        | 100/485 [00:11<00:56,  6.87it/s][A
     21%|██        | 101/485 [00:11<00:52,  7.28it/s][A
     21%|██        | 102/485 [00:11<00:49,  7.67it/s][A
     21%|██        | 103/485 [00:12<00:47,  8.06it/s][A
     21%|██▏       | 104/485 [00:12<00:54,  6.93it/s][A
     22%|██▏       | 105/485 [00:12<01:01,  6.19it/s][A
     22%|██▏       | 106/485 [00:12<00:56,  6.69it/s][A
     22%|██▏       | 107/485 [00:12<00:55,  6.79it/s][A
     22%|██▏       | 108/485 [00:12<00:51,  7.37it/s][A
     22%|██▏       | 109/485 [00:13<00:56,  6.66it/s][A
     23%|██▎       | 110/485 [00:13<00:59,  6.34it/s][A
     23%|██▎       | 112/485 [00:13<00:55,  6.71it/s][A
     23%|██▎       | 113/485 [00:13<00:55,  6.76it/s][A
     24%|██▎       | 114/485 [00:13<00:55,  6.69it/s][A
     24%|██▎       | 115/485 [00:13<00:52,  7.05it/s][A
     24%|██▍       | 116/485 [00:14<01:02,  5.94it/s][A
     24%|██▍       | 117/485 [00:14<00:58,  6.33it/s][A
     24%|██▍       | 118/485 [00:14<00:52,  7.00it/s][A
     25%|██▍       | 119/485 [00:14<00:51,  7.16it/s][A
     25%|██▍       | 120/485 [00:14<00:48,  7.53it/s][A
     25%|██▌       | 122/485 [00:14<00:47,  7.68it/s][A
     25%|██▌       | 123/485 [00:14<00:44,  8.17it/s][A
     26%|██▌       | 124/485 [00:15<01:03,  5.71it/s][A
     26%|██▌       | 125/485 [00:15<00:56,  6.32it/s][A
     26%|██▌       | 126/485 [00:15<00:51,  6.96it/s][A
     26%|██▋       | 128/485 [00:15<00:46,  7.74it/s][A
     27%|██▋       | 129/485 [00:15<00:44,  7.96it/s][A
     27%|██▋       | 130/485 [00:16<01:01,  5.80it/s][A
     27%|██▋       | 131/485 [00:16<00:55,  6.37it/s][A
     27%|██▋       | 133/485 [00:16<00:47,  7.35it/s][A
     28%|██▊       | 135/485 [00:16<00:43,  8.12it/s][A
     28%|██▊       | 136/485 [00:16<00:55,  6.25it/s][A
     28%|██▊       | 137/485 [00:16<00:55,  6.31it/s][A
     28%|██▊       | 138/485 [00:17<00:54,  6.40it/s][A
     29%|██▊       | 139/485 [00:17<00:50,  6.88it/s][A
     29%|██▉       | 140/485 [00:17<00:46,  7.37it/s][A
     29%|██▉       | 142/485 [00:17<00:44,  7.76it/s][A
     29%|██▉       | 143/485 [00:17<01:00,  5.68it/s][A
     30%|██▉       | 144/485 [00:18<01:04,  5.27it/s][A
     30%|██▉       | 145/485 [00:18<00:57,  5.96it/s][A
     30%|███       | 146/485 [00:18<00:51,  6.59it/s][A
     30%|███       | 147/485 [00:18<00:47,  7.10it/s][A
     31%|███       | 148/485 [00:18<00:46,  7.18it/s][A
     31%|███       | 149/485 [00:18<00:44,  7.59it/s][A
     31%|███       | 150/485 [00:18<01:02,  5.37it/s][A
     31%|███       | 151/485 [00:19<00:55,  5.97it/s][A
     31%|███▏      | 152/485 [00:19<00:51,  6.46it/s][A
     32%|███▏      | 153/485 [00:19<00:47,  6.92it/s][A
     32%|███▏      | 154/485 [00:19<00:49,  6.72it/s][A
     32%|███▏      | 155/485 [00:19<00:46,  7.12it/s][A
     32%|███▏      | 156/485 [00:19<01:01,  5.34it/s][A
     32%|███▏      | 157/485 [00:20<01:04,  5.10it/s][A
     33%|███▎      | 158/485 [00:20<00:54,  5.98it/s][A
     33%|███▎      | 159/485 [00:20<00:55,  5.91it/s][A
     33%|███▎      | 160/485 [00:20<00:48,  6.70it/s][A
     33%|███▎      | 161/485 [00:20<00:45,  7.17it/s][A
     33%|███▎      | 162/485 [00:20<00:42,  7.60it/s][A
     34%|███▍      | 164/485 [00:20<00:40,  7.86it/s][A
     34%|███▍      | 165/485 [00:21<00:51,  6.27it/s][A
     34%|███▍      | 166/485 [00:21<00:49,  6.38it/s][A
     34%|███▍      | 167/485 [00:21<00:45,  6.92it/s][A
     35%|███▍      | 168/485 [00:21<00:48,  6.55it/s][A
     35%|███▍      | 169/485 [00:21<00:50,  6.21it/s][A
     35%|███▌      | 170/485 [00:21<00:47,  6.65it/s][A
     35%|███▌      | 171/485 [00:22<00:46,  6.69it/s][A
     35%|███▌      | 172/485 [00:22<00:49,  6.27it/s][A
     36%|███▌      | 173/485 [00:22<00:45,  6.80it/s][A
     36%|███▌      | 174/485 [00:22<00:42,  7.23it/s][A
     36%|███▌      | 175/485 [00:22<00:41,  7.51it/s][A
     36%|███▋      | 176/485 [00:22<00:49,  6.23it/s][A
     36%|███▋      | 177/485 [00:23<00:52,  5.82it/s][A
     37%|███▋      | 178/485 [00:23<00:53,  5.77it/s][A
     37%|███▋      | 179/485 [00:23<00:55,  5.56it/s][A
     37%|███▋      | 180/485 [00:23<00:49,  6.11it/s][A
     37%|███▋      | 181/485 [00:23<00:45,  6.65it/s][A
     38%|███▊      | 182/485 [00:23<00:47,  6.35it/s][A
     38%|███▊      | 183/485 [00:24<00:47,  6.31it/s][A
     38%|███▊      | 184/485 [00:24<00:50,  5.96it/s][A
     38%|███▊      | 185/485 [00:24<00:48,  6.21it/s][A
     38%|███▊      | 186/485 [00:24<00:50,  5.88it/s][A
     39%|███▊      | 187/485 [00:24<00:49,  6.04it/s][A
     39%|███▉      | 188/485 [00:24<00:51,  5.76it/s][A
     39%|███▉      | 189/485 [00:25<00:46,  6.32it/s][A
     39%|███▉      | 190/485 [00:25<00:50,  5.85it/s][A
     39%|███▉      | 191/485 [00:25<00:50,  5.86it/s][A
     40%|███▉      | 192/485 [00:25<00:45,  6.43it/s][A
     40%|████      | 194/485 [00:25<00:40,  7.17it/s][A
     40%|████      | 195/485 [00:25<00:41,  7.02it/s][A
     40%|████      | 196/485 [00:26<00:48,  5.94it/s][A
     41%|████      | 197/485 [00:26<00:45,  6.30it/s][A
     41%|████      | 198/485 [00:26<00:41,  6.96it/s][A
     41%|████      | 199/485 [00:26<00:46,  6.14it/s][A
     41%|████      | 200/485 [00:26<00:43,  6.48it/s][A
     41%|████▏     | 201/485 [00:26<00:39,  7.17it/s][A
     42%|████▏     | 202/485 [00:27<00:52,  5.38it/s][A
     42%|████▏     | 203/485 [00:27<00:51,  5.45it/s][A
     42%|████▏     | 205/485 [00:27<00:43,  6.39it/s][A
     42%|████▏     | 206/485 [00:27<00:39,  7.13it/s][A
     43%|████▎     | 207/485 [00:27<00:36,  7.57it/s][A
     43%|████▎     | 208/485 [00:27<00:34,  8.02it/s][A
     43%|████▎     | 209/485 [00:28<00:45,  6.04it/s][A
     43%|████▎     | 210/485 [00:28<00:50,  5.41it/s][A
     44%|████▎     | 211/485 [00:28<00:49,  5.53it/s][A
     44%|████▎     | 212/485 [00:28<00:45,  5.94it/s][A
     44%|████▍     | 213/485 [00:28<00:46,  5.89it/s][A
     44%|████▍     | 214/485 [00:28<00:47,  5.67it/s][A
     44%|████▍     | 215/485 [00:29<00:48,  5.57it/s][A
     45%|████▍     | 216/485 [00:29<00:44,  6.09it/s][A
     45%|████▍     | 217/485 [00:29<00:41,  6.44it/s][A
     45%|████▍     | 218/485 [00:29<00:37,  7.12it/s][A
     45%|████▌     | 219/485 [00:29<00:34,  7.71it/s][A
     46%|████▌     | 221/485 [00:29<00:32,  8.22it/s][A
     46%|████▌     | 222/485 [00:30<00:44,  5.93it/s][A
     46%|████▌     | 223/485 [00:30<00:41,  6.30it/s][A
     46%|████▌     | 224/485 [00:30<00:40,  6.38it/s][A
     46%|████▋     | 225/485 [00:30<00:41,  6.25it/s][A
     47%|████▋     | 226/485 [00:30<00:41,  6.18it/s][A
     47%|████▋     | 227/485 [00:30<00:42,  6.09it/s][A
     47%|████▋     | 228/485 [00:31<00:41,  6.18it/s][A
     47%|████▋     | 229/485 [00:31<00:36,  6.95it/s][A
     48%|████▊     | 231/485 [00:31<00:33,  7.57it/s][A
     48%|████▊     | 232/485 [00:31<00:30,  8.16it/s][A
     48%|████▊     | 233/485 [00:31<00:34,  7.35it/s][A
     48%|████▊     | 234/485 [00:31<00:33,  7.47it/s][A
     48%|████▊     | 235/485 [00:31<00:32,  7.64it/s][A
     49%|████▊     | 236/485 [00:32<00:35,  6.92it/s][A
     49%|████▉     | 237/485 [00:32<00:38,  6.46it/s][A
     49%|████▉     | 238/485 [00:32<00:34,  7.16it/s][A
     49%|████▉     | 239/485 [00:32<00:32,  7.61it/s][A
     49%|████▉     | 240/485 [00:32<00:30,  8.10it/s][A
     50%|████▉     | 242/485 [00:32<00:35,  6.77it/s][A
     50%|█████     | 243/485 [00:33<00:36,  6.63it/s][A
     51%|█████     | 245/485 [00:33<00:32,  7.50it/s][A
     51%|█████     | 247/485 [00:33<00:29,  8.11it/s][A
     51%|█████▏    | 249/485 [00:33<00:33,  6.96it/s][A
     52%|█████▏    | 250/485 [00:34<00:34,  6.76it/s][A
     52%|█████▏    | 251/485 [00:34<00:32,  7.30it/s][A
     52%|█████▏    | 253/485 [00:34<00:30,  7.69it/s][A
     52%|█████▏    | 254/485 [00:34<00:29,  7.96it/s][A
     53%|█████▎    | 255/485 [00:34<00:29,  7.90it/s][A
     53%|█████▎    | 256/485 [00:34<00:27,  8.35it/s][A
     53%|█████▎    | 257/485 [00:35<00:39,  5.73it/s][A
     53%|█████▎    | 258/485 [00:35<00:39,  5.78it/s][A
     54%|█████▎    | 260/485 [00:35<00:33,  6.70it/s][A
     54%|█████▍    | 261/485 [00:35<00:32,  6.93it/s][A
     54%|█████▍    | 263/485 [00:35<00:29,  7.65it/s][A
     54%|█████▍    | 264/485 [00:36<00:40,  5.47it/s][A
     55%|█████▍    | 265/485 [00:36<00:39,  5.52it/s][A
     55%|█████▌    | 267/485 [00:36<00:34,  6.40it/s][A
     55%|█████▌    | 269/485 [00:36<00:29,  7.24it/s][A
     56%|█████▌    | 270/485 [00:36<00:27,  7.75it/s][A
     56%|█████▌    | 271/485 [00:36<00:35,  6.01it/s][A
     56%|█████▌    | 272/485 [00:37<00:32,  6.53it/s][A
     56%|█████▋    | 274/485 [00:37<00:29,  7.19it/s][A
     57%|█████▋    | 275/485 [00:37<00:28,  7.26it/s][A
     57%|█████▋    | 276/485 [00:37<00:29,  7.00it/s][A
     57%|█████▋    | 277/485 [00:37<00:27,  7.65it/s][A
     57%|█████▋    | 278/485 [00:37<00:26,  7.77it/s][A
     58%|█████▊    | 279/485 [00:38<00:34,  5.90it/s][A
     58%|█████▊    | 281/485 [00:38<00:31,  6.44it/s][A
     58%|█████▊    | 282/485 [00:38<00:34,  5.97it/s][A
     58%|█████▊    | 283/485 [00:38<00:32,  6.27it/s][A
     59%|█████▊    | 284/485 [00:38<00:29,  6.81it/s][A
     59%|█████▉    | 285/485 [00:38<00:28,  7.13it/s][A
     59%|█████▉    | 287/485 [00:39<00:29,  6.82it/s][A
     59%|█████▉    | 288/485 [00:39<00:28,  6.82it/s][A
     60%|█████▉    | 289/485 [00:39<00:27,  7.03it/s][A
     60%|█████▉    | 290/485 [00:39<00:26,  7.36it/s][A
     60%|██████    | 291/485 [00:39<00:29,  6.52it/s][A
     60%|██████    | 292/485 [00:39<00:26,  7.16it/s][A
     60%|██████    | 293/485 [00:40<00:26,  7.34it/s][A
     61%|██████    | 294/485 [00:40<00:25,  7.46it/s][A
     61%|██████    | 295/485 [00:40<00:25,  7.50it/s][A
     61%|██████    | 296/485 [00:40<00:28,  6.58it/s][A
     61%|██████    | 297/485 [00:40<00:27,  6.74it/s][A
     62%|██████▏   | 299/485 [00:40<00:24,  7.58it/s][A
     62%|██████▏   | 300/485 [00:40<00:23,  7.93it/s][A
     62%|██████▏   | 301/485 [00:41<00:22,  8.32it/s][A
     62%|██████▏   | 302/485 [00:41<00:21,  8.46it/s][A
     62%|██████▏   | 303/485 [00:41<00:27,  6.58it/s][A
     63%|██████▎   | 304/485 [00:41<00:26,  6.83it/s][A
     63%|██████▎   | 305/485 [00:41<00:24,  7.48it/s][A
     63%|██████▎   | 306/485 [00:41<00:25,  7.05it/s][A
     63%|██████▎   | 307/485 [00:41<00:24,  7.38it/s][A
     64%|██████▎   | 308/485 [00:42<00:22,  7.77it/s][A
     64%|██████▎   | 309/485 [00:42<00:28,  6.19it/s][A
     64%|██████▍   | 310/485 [00:42<00:27,  6.46it/s][A
     64%|██████▍   | 311/485 [00:42<00:25,  6.74it/s][A
     64%|██████▍   | 312/485 [00:42<00:28,  6.15it/s][A
     65%|██████▍   | 313/485 [00:42<00:27,  6.33it/s][A
     65%|██████▍   | 315/485 [00:43<00:25,  6.72it/s][A
     65%|██████▌   | 316/485 [00:43<00:25,  6.74it/s][A
     65%|██████▌   | 317/485 [00:43<00:24,  6.97it/s][A
     66%|██████▌   | 318/485 [00:43<00:23,  7.17it/s][A
     66%|██████▌   | 320/485 [00:43<00:21,  7.51it/s][A
     66%|██████▌   | 321/485 [00:43<00:21,  7.67it/s][A
     66%|██████▋   | 322/485 [00:44<00:29,  5.60it/s][A
     67%|██████▋   | 323/485 [00:44<00:26,  6.11it/s][A
     67%|██████▋   | 324/485 [00:44<00:24,  6.49it/s][A
     67%|██████▋   | 326/485 [00:44<00:21,  7.29it/s][A
     68%|██████▊   | 328/485 [00:44<00:21,  7.39it/s][A
     68%|██████▊   | 329/485 [00:45<00:19,  7.86it/s][A
     68%|██████▊   | 330/485 [00:45<00:21,  7.17it/s][A
     68%|██████▊   | 331/485 [00:45<00:27,  5.59it/s][A
     68%|██████▊   | 332/485 [00:45<00:25,  6.08it/s][A
     69%|██████▊   | 333/485 [00:45<00:25,  6.01it/s][A
     69%|██████▉   | 334/485 [00:45<00:25,  5.85it/s][A
     69%|██████▉   | 336/485 [00:46<00:23,  6.38it/s][A
     69%|██████▉   | 337/485 [00:46<00:20,  7.10it/s][A
     70%|██████▉   | 338/485 [00:46<00:20,  7.14it/s][A
     70%|██████▉   | 339/485 [00:46<00:20,  7.11it/s][A
     70%|███████   | 340/485 [00:46<00:21,  6.61it/s][A
     71%|███████   | 342/485 [00:46<00:19,  7.50it/s][A
     71%|███████   | 343/485 [00:47<00:17,  7.95it/s][A
     71%|███████   | 344/485 [00:47<00:16,  8.45it/s][A
     71%|███████   | 345/485 [00:47<00:16,  8.66it/s][A
     71%|███████▏  | 346/485 [00:47<00:17,  7.91it/s][A
     72%|███████▏  | 347/485 [00:47<00:17,  7.79it/s][A
     72%|███████▏  | 348/485 [00:47<00:17,  7.81it/s][A
     72%|███████▏  | 349/485 [00:47<00:20,  6.66it/s][A
     72%|███████▏  | 350/485 [00:48<00:20,  6.43it/s][A
     72%|███████▏  | 351/485 [00:48<00:20,  6.40it/s][A
     73%|███████▎  | 352/485 [00:48<00:19,  6.91it/s][A
     73%|███████▎  | 353/485 [00:48<00:17,  7.44it/s][A
     73%|███████▎  | 355/485 [00:48<00:15,  8.26it/s][A
     73%|███████▎  | 356/485 [00:48<00:15,  8.52it/s][A
     74%|███████▎  | 357/485 [00:48<00:21,  5.88it/s][A
     74%|███████▍  | 358/485 [00:49<00:22,  5.76it/s][A
     74%|███████▍  | 359/485 [00:49<00:20,  6.30it/s][A
     74%|███████▍  | 361/485 [00:49<00:17,  7.24it/s][A
     75%|███████▍  | 363/485 [00:49<00:15,  8.08it/s][A
     75%|███████▌  | 364/485 [00:49<00:14,  8.42it/s][A
     75%|███████▌  | 365/485 [00:50<00:19,  6.06it/s][A
     75%|███████▌  | 366/485 [00:50<00:19,  5.99it/s][A
     76%|███████▌  | 367/485 [00:50<00:17,  6.70it/s][A
     76%|███████▌  | 368/485 [00:50<00:17,  6.60it/s][A
     76%|███████▌  | 369/485 [00:50<00:16,  7.25it/s][A
     76%|███████▋  | 371/485 [00:50<00:15,  7.47it/s][A
     77%|███████▋  | 372/485 [00:50<00:14,  8.01it/s][A
     77%|███████▋  | 373/485 [00:51<00:14,  7.88it/s][A
     77%|███████▋  | 374/485 [00:51<00:13,  8.25it/s][A
     77%|███████▋  | 375/485 [00:51<00:13,  8.31it/s][A
     78%|███████▊  | 376/485 [00:51<00:12,  8.72it/s][A
     78%|███████▊  | 377/485 [00:51<00:14,  7.55it/s][A
     78%|███████▊  | 378/485 [00:51<00:13,  8.04it/s][A
     78%|███████▊  | 379/485 [00:51<00:18,  5.59it/s][A
     78%|███████▊  | 380/485 [00:52<00:18,  5.78it/s][A
     79%|███████▉  | 382/485 [00:52<00:15,  6.59it/s][A
     79%|███████▉  | 384/485 [00:52<00:13,  7.63it/s][A
     79%|███████▉  | 385/485 [00:52<00:12,  7.98it/s][A
     80%|███████▉  | 386/485 [00:52<00:16,  6.04it/s][A
     80%|███████▉  | 387/485 [00:52<00:14,  6.75it/s][A
     80%|████████  | 388/485 [00:53<00:15,  6.20it/s][A
     80%|████████  | 390/485 [00:53<00:13,  7.14it/s][A
     81%|████████  | 391/485 [00:53<00:16,  5.54it/s][A
     81%|████████  | 392/485 [00:53<00:15,  5.82it/s][A
     81%|████████  | 393/485 [00:53<00:14,  6.45it/s][A
     81%|████████  | 394/485 [00:54<00:13,  6.92it/s][A
     82%|████████▏ | 396/485 [00:54<00:11,  7.62it/s][A
     82%|████████▏ | 397/485 [00:54<00:13,  6.38it/s][A
     82%|████████▏ | 398/485 [00:54<00:12,  7.00it/s][A
     82%|████████▏ | 399/485 [00:54<00:11,  7.45it/s][A
     82%|████████▏ | 400/485 [00:54<00:11,  7.44it/s][A
     83%|████████▎ | 401/485 [00:54<00:11,  7.16it/s][A
     83%|████████▎ | 403/485 [00:55<00:12,  6.31it/s][A
     83%|████████▎ | 404/485 [00:55<00:12,  6.49it/s][A
     84%|████████▎ | 406/485 [00:55<00:10,  7.26it/s][A
     84%|████████▍ | 408/485 [00:55<00:09,  7.98it/s][A
     84%|████████▍ | 409/485 [00:56<00:13,  5.65it/s][A
     85%|████████▍ | 410/485 [00:56<00:13,  5.76it/s][A
     85%|████████▍ | 412/485 [00:56<00:10,  6.71it/s][A
     85%|████████▌ | 413/485 [00:56<00:09,  7.43it/s][A
     85%|████████▌ | 414/485 [00:56<00:08,  8.00it/s][A
     86%|████████▌ | 416/485 [00:57<00:09,  6.93it/s][A
     86%|████████▌ | 417/485 [00:57<00:10,  6.68it/s][A
     86%|████████▌ | 418/485 [00:57<00:09,  7.41it/s][A
     87%|████████▋ | 420/485 [00:57<00:07,  8.20it/s][A
     87%|████████▋ | 422/485 [00:57<00:07,  8.72it/s][A
     87%|████████▋ | 423/485 [00:57<00:09,  6.89it/s][A
     87%|████████▋ | 424/485 [00:58<00:09,  6.34it/s][A
     88%|████████▊ | 425/485 [00:58<00:09,  6.63it/s][A
     88%|████████▊ | 426/485 [00:58<00:08,  7.37it/s][A
     88%|████████▊ | 428/485 [00:58<00:06,  8.19it/s][A
     88%|████████▊ | 429/485 [00:58<00:06,  8.55it/s][A
     89%|████████▉ | 431/485 [00:59<00:07,  7.13it/s][A
     89%|████████▉ | 432/485 [00:59<00:07,  6.74it/s][A
     89%|████████▉ | 434/485 [00:59<00:06,  7.57it/s][A
     90%|████████▉ | 436/485 [00:59<00:05,  8.30it/s][A
     90%|█████████ | 437/485 [00:59<00:06,  6.94it/s][A
     90%|█████████ | 438/485 [01:00<00:09,  5.18it/s][A
     91%|█████████ | 439/485 [01:00<00:07,  6.00it/s][A
     91%|█████████ | 440/485 [01:00<00:06,  6.70it/s][A
     91%|█████████ | 441/485 [01:00<00:06,  7.29it/s][A
     91%|█████████ | 442/485 [01:00<00:05,  7.86it/s][A
     91%|█████████▏| 443/485 [01:00<00:05,  8.33it/s][A
     92%|█████████▏| 444/485 [01:00<00:04,  8.31it/s][A
     92%|█████████▏| 445/485 [01:01<00:06,  5.72it/s][A
     92%|█████████▏| 446/485 [01:01<00:06,  5.80it/s][A
     92%|█████████▏| 448/485 [01:01<00:05,  6.79it/s][A
     93%|█████████▎| 449/485 [01:01<00:04,  7.22it/s][A
     93%|█████████▎| 451/485 [01:01<00:04,  8.05it/s][A
     93%|█████████▎| 452/485 [01:02<00:05,  5.60it/s][A
     93%|█████████▎| 453/485 [01:02<00:05,  5.69it/s][A
     94%|█████████▎| 454/485 [01:02<00:05,  6.11it/s][A
     94%|█████████▍| 455/485 [01:02<00:04,  6.75it/s][A
     94%|█████████▍| 457/485 [01:02<00:04,  6.81it/s][A
     94%|█████████▍| 458/485 [01:02<00:04,  6.13it/s][A
     95%|█████████▍| 460/485 [01:03<00:03,  6.95it/s][A
     95%|█████████▌| 461/485 [01:03<00:04,  4.98it/s][A
     95%|█████████▌| 462/485 [01:03<00:04,  5.07it/s][A
     95%|█████████▌| 463/485 [01:03<00:03,  5.69it/s][A
     96%|█████████▌| 464/485 [01:03<00:03,  6.46it/s][A
     96%|█████████▌| 465/485 [01:03<00:02,  6.96it/s][A
     96%|█████████▌| 466/485 [01:04<00:02,  7.55it/s][A
     96%|█████████▋| 467/485 [01:04<00:02,  7.66it/s][A
     96%|█████████▋| 468/485 [01:04<00:03,  5.20it/s][A
     97%|█████████▋| 469/485 [01:04<00:02,  5.61it/s][A
     97%|█████████▋| 470/485 [01:04<00:02,  6.04it/s][A
     97%|█████████▋| 471/485 [01:04<00:02,  6.79it/s][A
     97%|█████████▋| 472/485 [01:05<00:02,  5.29it/s][A
     98%|█████████▊| 473/485 [01:05<00:02,  5.39it/s][A
     98%|█████████▊| 475/485 [01:05<00:01,  6.40it/s][A
     98%|█████████▊| 476/485 [01:05<00:01,  7.03it/s][A
     99%|█████████▊| 478/485 [01:05<00:00,  7.93it/s][A
     99%|█████████▉| 479/485 [01:06<00:00,  7.64it/s][A
     99%|█████████▉| 480/485 [01:06<00:00,  5.52it/s][A
     99%|█████████▉| 481/485 [01:06<00:00,  5.60it/s][A
     99%|█████████▉| 482/485 [01:06<00:00,  6.44it/s][A
    100%|█████████▉| 484/485 [01:06<00:00,  7.21it/s][A
    100%|██████████| 485/485 [01:06<00:00,  7.25it/s][A

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: test_videos_output/challenge_video.mp4 
    
    CPU times: user 3min 25s, sys: 4.04 s, total: 3min 29s
    Wall time: 1min 8s


In contrast to video *project_video.mp4* the algorithm has problems to detect lane lines in video *challenge_video.mp4* reliably (but still there wouldn't be any catastrophic failure). In particular the left lane line - which is close to a shadow - is sometimes described by a polynom being curved in the direction of the shadow. Clearly, additional suppression of background pixels is needed here. 

* [project video](https://youtu.be/psQJ1KA-wtk)
* [challenge video](https://www.youtube.com/watch?v=zMv0oHytInE)

## 9. Conclusions

The objective of the  project was met. We developed an algorithm which is able to detect lane lines in video streams using methods like camera calibration, color thresholding, region selection, perspective transformation and outlier detection. The algorithm worked satisfactory for the video *project_video.mp4* but has some problems with video *challenge_video.mp4* due to shadows on the left-hand side of the road. Possible improvements are:
* better suppression of background pixels
* use more robust fit method (least-square fit is sensitive to outliers)
