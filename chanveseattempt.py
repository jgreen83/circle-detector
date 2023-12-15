import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.segmentation import chan_vese
import cv2
import numpy as np
from sklearn.cluster import DBSCAN


def display_img(img, windowname="image"):
    cv2.imshow(windowname, img)
    cv2.waitKey(10000)

imgfile = 'outlet(1).JPG'
image = cv2.imread(imgfile)
image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Feel free to play around with the parameters to see how they impact the result
cv = chan_vese(gray, mu=0.25, lambda1=1, lambda2=1, tol=1e-3,
               max_num_iter=200, dt=0.5, init_level_set="checkerboard",
               extended_output=False)
cv_im = np.uint8(cv)

fig, axes = plt.subplots(1, 2, figsize=(8, 8))
ax = axes.flatten()

ax[0].imshow(gray, cmap="gray")
ax[0].set_axis_off()
ax[0].set_title("Original Image", fontsize=12)

ax[1].imshow(cv_im, cmap="gray")
ax[1].set_axis_off()
title = f'Chan-Vese segmentation'
ax[1].set_title(title, fontsize=12)
"""
ax[2].imshow(cv[1], cmap="gray")
ax[2].set_axis_off()
ax[2].set_title("Final Level Set", fontsize=12)

ax[3].plot(cv[2])
ax[3].set_title("Evolution of energy over iterations", fontsize=12)
"""

fig.tight_layout()
plt.show()

#cv_im = np.matrix(cv)
#print(cv.shape)
#thresh canne thing
Canny = cv2.Canny(255*cv_im, 170, 250)
contours, _ = cv2.findContours(Canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contour_img = np.zeros_like(image)

plt.figure(3, figsize=(10,10))
plt.subplot(121)
plt.imshow(Canny, cmap='gray')
plt.axis('off')
plt.title("Canny Image")
plt.subplot(122)
plt.imshow(contour_img)
plt.title("contours")
plt.axis('off')
plt.show()

min_contour_length = 500
long_contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) > min_contour_length]
longest_contour = max(contours, key=lambda contour: cv2.arcLength(contour, True))

contour_img = np.zeros_like(image)
contour_longest = np.zeros_like(image)
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
#print(len(np.where(gray_img >= 130)[0]))
points = np.column_stack((x, y))

# Cluster the points using DBSCAN
dbscan = DBSCAN(eps=5, min_samples=5)
clusters = dbscan.fit_predict(points)


# Initialize an empty image to draw ellipses
ellipse_image = np.zeros_like(gray_img)

# Define acceptance ratio for filtering ellipses
acceptance_ratio = 1.35

cluster_lens = []
for cluster in set(clusters):
    cluster_lens.append(len(points[clusters==cluster]))



# Process each cluster
for cluster in set(clusters):
    if cluster == -1:
        # Skip noise points
        continue

    # Extract points belonging to the current cluster
    cluster_points = points[clusters == cluster]

    if len(cluster_points) >= np.percentile(cluster_lens, 80):
        # Fit an ellipse to the cluster
        ellipse = cv2.fitEllipse(np.array(cluster_points))

        # Extract the lengths of the major and minor axes
        major_axis_length = max(ellipse[1])
        minor_axis_length = min(ellipse[1])

        # Filter ellipses based on axis lengths
        if major_axis_length / (minor_axis_length+0.0000000000001) <= acceptance_ratio:
            # Draw the ellipse
            cv2.ellipse(image, ellipse, (0, 0, 255), 2)
        #cv2.ellipse(image, ellipse, (0, 0, 255), 2)

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
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.title("Fitted Ellipse")
plt.show()


"""

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
plt.show()
"""