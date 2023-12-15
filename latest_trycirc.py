###############################################
# USAGE: press any key to continue
###############################################

import cv2
import numpy as np


def display_img(img, windowname="image"):
    cv2.imshow(windowname, img)
    # cv2.waitKey(10000)
    cv2.waitKey()


# img = cv2.imread('./outlet.jpg')
# img = cv2.imread('./with_shadow.jpg')
# img = cv2.imread('./fullimg_outlet.JPG')  # 30
# img = cv2.imread('./fullimg_withwater.JPG')  # 30
# img = cv2.imread('./fullimg_with_shadow.jpg')  # 50
# img = cv2.imread('./fullimg_vague.jpg')  # 30
img = cv2.imread('./DSC0056 copy.JPG')  # 30

# make image smaller
# img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
img = cv2.resize(img, (1024, 720))
# grab the height of the image
img_height = img.shape[0]
print("image height: {}".format(img_height))
# print img.shape
print("image shape: {}".format(img.shape))

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
# # calculate the standard deviation of the histogram
# stddev = np.std(img)
# print("stddev: {}".format(stddev))
# high_contrast = np.any(histogram[:10] > 0) and np.any(histogram[245:] > 0)
# print("high contrast: {}".format(high_contrast))

threshold_value = 40
histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
cdf = histogram.cumsum()
cdf_normalized = cdf * float(histogram.max()) / cdf.max()
low_contrast_threshold = np.argmax(cdf_normalized > 0.20 * cdf_normalized[-1])
high_contrast_threshold = np.argmax(cdf_normalized > 0.80 * cdf_normalized[-1])
print("low contrast threshold: {}".format(low_contrast_threshold))
print("high contrast threshold: {}".format(high_contrast_threshold))
use_equalization = high_contrast_threshold - \
    low_contrast_threshold < threshold_value


# apply equalizeHist
if use_equalization:
    gray = cv2.equalizeHist(gray)
    print("using equalization")

#gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 101, 0)
#gray = cv2.addWeighted(gray, 1.2, 255-gray, -1, 0)

# Apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (13, 13), 0)


# Apply median blur to reduce noise
blur = cv2.medianBlur(blur, 17)

# Apply Hough Transform to detect circles
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1,
                           max(blur.shape), param1=50, param2=30, minRadius=0, maxRadius=img_height//4)

# Draw the detected circles on a black background
output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(output, (x, y), r, (255, 180, 180), -1)

# Display the input and output images
display_img(img, "input")
display_img(blur, "blur")
display_img(output, "output")


# Save the output image
# cv2.imwrite('water_outlet_processed.jpg', output)
