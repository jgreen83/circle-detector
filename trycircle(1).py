###############################################
# USAGE: press any key to continue
###############################################

import cv2
import numpy as np


def display_img(img, windowname="image"):
    cv2.imshow(windowname, img)
    cv2.waitKey(10000)
    # cv2.waitKey()


# img = cv2.imread('./outlet.jpg')
# img = cv2.imread('./fullimg_nowater.JPG')
img = cv2.imread('./00d62881-9.jpg')

# make image smaller
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

equalized = cv2.equalizeHist(gray)

# Apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(equalized, (13, 13), 0)

# Apply median blur to reduce noise
blur = cv2.medianBlur(blur, 17)

# Apply Hough Transform to detect circles
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1,
                           max(blur.shape), param1=50, param2=50, minRadius=0, maxRadius=0)

# Draw the detected circles on a black background
output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(output, (x, y), r, (255, 255, 255), -1)

# Display the input and output images
display_img(img, "input")
display_img(blur, "blur")
display_img(output, "output")


# Save the output image
# cv2.imwrite('water_outlet_processed.jpg', output)
