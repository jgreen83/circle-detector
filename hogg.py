import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt

# Load image and compute HOG
image = cv2.imread('./DSC0056.JPG')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
fd, hog_image = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# Define sliding window parameters
window_size = (128, 128)
step_size = 64
threshold_value = 0.06

# Initialize a mask
mask = np.zeros_like(gray)

# Slide window
for y in range(0, hog_image_rescaled.shape[0] - window_size[1], step_size):
    for x in range(0, hog_image_rescaled.shape[1] - window_size[0], step_size):
        window = hog_image_rescaled[y:y + window_size[1], x:x + window_size[0]]

        # Check if the window meets the criteria
        if np.mean(window) < threshold_value:
            mask[y:y + window_size[1], x:x + window_size[0]] = 255

# Apply the mask to the original image
masked_image = cv2.bitwise_and(image, image, mask=mask)

# Convert the masked image to grayscale
masked_gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

# Turn the masked image into a binary image
_, binary_image = cv2.threshold(masked_gray, 30, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding box for each contour
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the masked image with the bounding box
masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(15,15))
plt.subplot(131)
plt.imshow(masked_image)
plt.axis('off')
plt.subplot(132)
plt.imshow(mask, cmap='gray')
plt.axis('off')
plt.subplot(133)
plt.imshow(image)
plt.axis('off')
plt.show()