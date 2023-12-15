###############################################
# USAGE: press any key to continue
###############################################

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


def display_img(img, windowname="image"):
    cv2.imshow(windowname, img)
    cv2.waitKey(10000)
    # cv2.waitKey()
imgfile = 'outlet(1).jpg'

# img = cv2.imread('./outlet.jpg')
# img = cv2.imread('./fullimg_nowater.JPG')
img = cv2.imread(imgfile)

# make image smaller
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


equalized = cv2.equalizeHist(255 - gray)
contrast1 = gray.std()
contrastEq = equalized.std()
print(contrast1)
print(contrastEq)


negative_eq = 255 - equalized
combination = cv2.addWeighted(equalized, -1, negative_eq, 1.2, 0)
#combination = cv2.GaussianBlur(combination, (7, 7), 0)
#equalized = cv2.GaussianBlur(equalized, (7, 7), 0)
Canny = cv2.Canny(combination, 170, 250)
contours, _ = cv2.findContours(Canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

min_contour_length = 300
long_contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) > min_contour_length]
longest_contour = max(contours, key=lambda contour: cv2.arcLength(contour, True))

contour_img = np.zeros_like(img)
contour_longest = np.zeros_like(img)
cv2.drawContours(contour_img, long_contours, -1, (255, 255, 255), thickness=cv2.FILLED)
cv2.drawContours(contour_longest, longest_contour, -1, (255, 255, 255), thickness=cv2.FILLED)

plt.figure(3, figsize=(10,10))
plt.subplot(121)
plt.imshow(Canny, cmap='gray')
plt.axis('off')
plt.title("Canny Image")
plt.subplot(122)
plt.imshow(contour_img)
plt.title("contours")
plt.axis('off')

if len(contour_img.shape) == 3:
    gray_img = cv2.cvtColor(contour_img, cv2.COLOR_BGR2GRAY)
else:
    gray_img = contour_img

# Extract points where the pixel value is 255 (white)
y, x = np.where(gray_img == 255)
points = np.column_stack((x, y))
"""
# Cluster the points using DBSCAN
dbscan = DBSCAN(eps=5, min_samples=1)
clusters = dbscan.fit_predict(points)

# Initialize an empty image to draw ellipses
ellipse_image = np.zeros_like(gray_img)

# Define acceptance ratio for filtering ellipses
acceptance_ratio = 1.2

# Process each cluster
for cluster in set(clusters):
    if cluster == -1:
        # Skip noise points
        continue

    # Extract points belonging to the current cluster
    cluster_points = points[clusters == cluster]

    if len(cluster_points) >= 5:
        # Fit an ellipse to the cluster
        ellipse = cv2.fitEllipse(np.array(cluster_points))

        # Extract the lengths of the major and minor axes
        major_axis_length = max(ellipse[1])
        minor_axis_length = min(ellipse[1])

        # Filter ellipses based on axis lengths
        if major_axis_length / (minor_axis_length+0.0000000000001) <= acceptance_ratio:
            # Draw the ellipse
            cv2.ellipse(img, ellipse, (0, 0, 255), 2)

plt.figure(4, figsize=(10,10))
plt.subplot(121)
plt.imshow(contour_longest, cmap='gray')
plt.axis('off')
plt.title("Longest Contour")
plt.subplot(122)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title("Fitted Ellipse")




plt.figure(1, figsize=(10,10))
plt.subplot(121)
plt.imshow(img)
plt.axis('off')
plt.title("Original Image")
plt.subplot(122)
plt.imshow(equalized, cmap="gray")
plt.title("equalized grayscale image")
plt.axis('off')

plt.figure(2, figsize=(10,10))
plt.subplot(121)
plt.imshow(negative_eq, cmap='gray')
plt.axis('off')
plt.title("Negative Image")
plt.subplot(122)
plt.imshow(combination, cmap="gray")
plt.title("Weighted Negative + Original Image")
plt.axis('off')
#plt.show()
"""

#thresh = cv2.cvtColor(equalized, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(negative_eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 101, 0)
display_img(thresh,"thresh")
display_img(255-thresh,"neg thresh")
neg_thresh = 255-thresh

combinationC = cv2.addWeighted(thresh, 1.2, neg_thresh, -1, 0)

display_img(combinationC,"comb")

#thresh canne thing
Canny = cv2.Canny(thresh, 170, 250)
contours, _ = cv2.findContours(Canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

min_contour_length = 500
long_contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) > min_contour_length]
longest_contour = max(contours, key=lambda contour: cv2.arcLength(contour, True))

contour_img = np.zeros_like(img)
contour_longest = np.zeros_like(img)
cv2.drawContours(contour_img, long_contours, -1, (255, 255, 255), thickness=cv2.FILLED)
cv2.drawContours(contour_longest, longest_contour, -1, (255, 255, 255), thickness=cv2.FILLED)

plt.clf()
plt.figure(3, figsize=(10,10))
plt.subplot(131)
plt.imshow(img)
plt.axis('off')
plt.title("Original Image")
plt.subplot(132)
plt.imshow(Canny, cmap='gray')
plt.axis('off')
plt.title("Canny Image")
plt.subplot(133)
plt.imshow(contour_img)
plt.title("contours")
plt.axis('off')
plt.show()

if len(contour_img.shape) == 3:
    gray_img = cv2.cvtColor(contour_img, cv2.COLOR_BGR2GRAY)
else:
    gray_img = contour_img

# Extract points where the pixel value is 255 (white)
y, x = np.where(gray_img == 255)
points = np.column_stack((x, y))

# Cluster the points using DBSCAN
dbscan = DBSCAN(eps=5, min_samples=1)
clusters = dbscan.fit_predict(points)


# Initialize an empty image to draw ellipses
ellipse_image = np.zeros_like(gray_img)

# Define acceptance ratio for filtering ellipses
acceptance_ratio = 1.2

# Process each cluster
for cluster in set(clusters):
    if cluster == -1:
        # Skip noise points
        continue

    # Extract points belonging to the current cluster
    cluster_points = points[clusters == cluster]

    if len(cluster_points) >= 5:
        # Fit an ellipse to the cluster
        ellipse = cv2.fitEllipse(np.array(cluster_points))


        # Extract the lengths of the major and minor axes
        major_axis_length = max(ellipse[1])
        minor_axis_length = min(ellipse[1])

        # Filter ellipses based on axis lengths
        if major_axis_length / (minor_axis_length+0.0000000000001) <= acceptance_ratio:
            # Draw the ellipse
            cv2.ellipse(img, ellipse, (0, 0, 255), 2)
            print(ellipse[0][0])
            print(ellipse[0][1])
        #cv2.ellipse(img, ellipse, (0, 0, 255), 2)

plt.figure(4, figsize=(10,10))
plt.subplot(121)
plt.imshow(contour_longest, cmap='gray')
plt.axis('off')
plt.title("Longest Contour")
#plt.subplot(121)
#plt.imshow(cv2.imread(imgfile))
#plt.axis('off')
#plt.title("Original Image")
plt.subplot(122)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title("Fitted Ellipse")




plt.figure(1, figsize=(10,10))
plt.subplot(121)
plt.imshow(img)
plt.axis('off')
plt.title("Original Image")
plt.subplot(122)
plt.imshow(equalized, cmap="gray")
plt.title("equalized grayscale image")
plt.axis('off')











print(type(equalized))

# Apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(equalized, (13, 13), 0)

# Apply median blur to reduce noise
blur = cv2.medianBlur(blur, 17)
"""
#--- First obtain the threshold using the greyscale image ---
ret,th = cv2.threshold(gray,127,255, 0)

#--- Find all the contours in the binary image ---
contours,hierarchy = cv2.findContours(thresh,2,1)
cnt = contours
big_contour = []
max = 0
for i in cnt:
   area = cv2.contourArea(i) #--- find the contour having biggest area ---#
   if(area > max):
       max = area
       big_contour = i

final = cv2.drawContours(img, big_contour, -1, (0,255,0), 3)
display_img(final,"final")
"""

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
display_img(equalized, "equalized image")

display_img(blur, "blur")
display_img(output, "output")


# Save the output image
# cv2.imwrite('water_outlet_processed.jpg', output)
