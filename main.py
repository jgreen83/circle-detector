###############################################
# USAGE: press any key to continue
###############################################

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, Point
import shapely
from matplotlib.patches import Ellipse
import os


def parse_xml_for_polygon(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    polygons = []
    for object in root.findall('object'):
        polygon = object.find('polygon')
        points = []
        i = 1
        while True:
            x_tag = f'x{i}'
            y_tag = f'y{i}'
            x = polygon.find(x_tag)
            y = polygon.find(y_tag)
            if x is not None and y is not None:
                x_coord = int(float(x.text))
                y_coord = int(float(y.text))
                points.append((x_coord, y_coord))
                i += 1
            else:
                break
        polygons.append(points)
    return polygons


def calculate_iou_circle_polygon(circle, polygon_points):
    circle_geom = Point(circle[0], circle[1]).buffer(circle[2])
    polygon_geom = Polygon(polygon_points)
    intersection = circle_geom.intersection(polygon_geom).area
    union = circle_geom.union(polygon_geom).area
    return intersection / union if union != 0 else 0


def calculate_iou_ellipse_polygon(ellipse, polygon_points):
    ellipse = Ellipse((ellipse[0][0],ellipse[0][1]), width=ellipse[1], height=ellipse[2], angle=ellipse[3])
    vertices = ellipse.get_verts()
    ellipse_geom = Polygon(vertices)
    polygon_geom = Polygon(polygon_points)
    intersection = ellipse_geom.intersection(polygon_geom).area
    union = ellipse_geom.union(polygon_geom).area
    return intersection / union if union != 0 else 0


def display_img(img, windowname="image"):
    cv2.imshow(windowname, img)
    # cv2.waitKey(10000)
    cv2.waitKey()


def do_pred_fin(img_path):
    img = cv2.imread(img_path)

    # align the image shape 1024x720 as close as possible
    factor = 1024.0 / img.shape[1]
    print(img.shape)
    img = cv2.resize(img, (0, 0), fx=factor, fy=factor)
    print(img.shape)

    # img = cv2.resize(img, (1024, 720))
    # grab the height of the image
    img_height = img.shape[0]
    # print img.shape
    print("image shape: {}".format(img.shape))

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

    # use_equalization = True
    # apply equalizeHist
    if use_equalization:
        gray = cv2.equalizeHist(gray)
        print("using equalization")


    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (13, 13), 0)

    # Apply median blur to reduce noise
    blur = cv2.medianBlur(blur, 17)

    # Apply Hough Transform to detect circles
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1,
                               max(blur.shape), param1=50, param2=30, minRadius=0, maxRadius=img_height // 4)

    # Draw the detected circles on a black background
    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    output = cv2.resize(output, (0, 0), fx=1 / factor, fy=1 / factor)

    circlesFin = []

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(output, (int(x // factor), int(y // factor)), int(r // factor), (255, 180, 180), -1)
            estCent = [int(x // factor), int(y // factor)]
            estR = int(r // factor)
            circlesFin.append([estCent[0],estCent[1],estR])

    # Display the input and output images
    #display_img(img, "input")
    #display_img(blur, "blur")
    #display_img(output, "output")

    # Save the output image
    # cv2.imwrite('water_outlet_processed.jpg', output)

    return circlesFin, output


def adap_thresh(img_path):
    img = cv2.imread(img_path)

    # make image smaller
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_height = img.shape[0]
    # print img.shape
    print("image shape: {}".format(img.shape))

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

    # use_equalization = True
    # apply equalizeHist
    if use_equalization:
        gray = cv2.equalizeHist(gray)
        print("using equalization")

    equalized = gray
    negative_eq = 255 - equalized

    # adaptive thresholding
    thresh = cv2.adaptiveThreshold(negative_eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 101, 0)
    #display_img(thresh, "thresh")
    #display_img(255 - thresh, "neg thresh")
    neg_thresh = 255 - thresh
    combinationC = cv2.addWeighted(thresh, 1.2, neg_thresh, -1, 0)
    #display_img(combinationC,"comb")

    Canny = cv2.Canny(thresh, 170, 250)
    contours, _ = cv2.findContours(Canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_length = 350
    long_contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) > min_contour_length]
    longest_contour = max(contours, key=lambda contour: cv2.arcLength(contour, True))

    contour_img = np.zeros_like(img)
    contour_longest = np.zeros_like(img)
    cv2.drawContours(contour_img, long_contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    cv2.drawContours(contour_longest, longest_contour, -1, (255, 255, 255), thickness=cv2.FILLED)

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
    finalElls = []

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
            if major_axis_length / (minor_axis_length + 0.0000000000001) <= acceptance_ratio:
                # Draw the ellipse
                cv2.ellipse(img, ellipse, (0, 0, 255), 2)
                #display_img(img, "ellipse!")
                # change sizing factor if necessary
                estCen = [ellipse[0][0] * 2, ellipse[0][1] * 2]
                estHor = (ellipse[1][0] / 2) * 2
                estVert = (ellipse[1][1] / 2) * 2
                degs = ellipse[2]
                finalElls.append([estCen, estHor, estVert, degs])
            #cv2.ellipse(img, ellipse, (0, 0, 255), 2)
    #display_img(img,"ellipseee")
    """
    plt.figure(1, figsize=(10, 10))
    plt.subplot(121)
    plt.imshow(Canny, cmap='gray')
    plt.axis('off')
    plt.title("canny")
    plt.subplot(122)
    plt.imshow(gray_img, cmap="gray")
    plt.title("contours gray")
    plt.axis('off')
    plt.show()
    """
    return finalElls, img


#elll,outt = adap_thresh('./DSC0056.JPG')


# image_folder = '/content/drive/MyDrive/IOU Test/Images'
image_folder = '/Users/Greencat/Desktop/Desktop - MacBook Pro (3)/official stuff/school but college/ece 588/Images'
# xml_folder = '/content/drive/MyDrive/IOU Test/Labels'
xml_folder = '/Users/Greencat/Desktop/Desktop - MacBook Pro (3)/official stuff/school but college/ece 588/Temp'
# output_folder = '/content/drive/MyDrive/IOU Test/Output'
output_folder = '/Users/Greencat/Desktop/Desktop - MacBook Pro (3)/official stuff/school but college/ece 588/Output'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

iou_values_circ = []
iou_values_ell = []

for image_name in os.listdir(image_folder):
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    base_name = os.path.splitext(image_name)[0]
    img_path = os.path.join(image_folder, image_name)
    xml_path = os.path.join(xml_folder, base_name + '.xml')
    output_path = os.path.join(output_folder, 'output_' + image_name)

    img = cv2.imread(img_path)
    detected_circles, circOut = do_pred_fin(img_path)
    detected_ellipses, ellOut = adap_thresh(img_path)
    ground_truth_polygons = parse_xml_for_polygon(xml_path)
    if ground_truth_polygons is -1:
      continue

    for polygon in ground_truth_polygons:
        polygon_np = np.array([polygon], np.int32)
        cv2.polylines(img, [polygon_np], True, (0, 0, 255), 8)  # Red color for polygons

    if detected_circles is not None:
        for (x, y, r) in detected_circles:
            cv2.circle(img, (x, y), r, (255, 0, 0), 8)  # Blue color for circles
            for polygon in ground_truth_polygons:
                iou = calculate_iou_circle_polygon((x, y, r), polygon)
                iou_values_circ.append(iou)

    if detected_ellipses is not None:
        for ell in detected_ellipses:
            cv2.ellipse(img, center=(int(ell[0][0]),int(ell[0][1])), axes=(int(ell[1]),int(ell[2])), angle=ell[3],startAngle=0,endAngle=360,color=(0, 255, 0),thickness=8)  # green color for ellipses
            for polygon in ground_truth_polygons:
                iou = calculate_iou_ellipse_polygon(ell, polygon)
                iou_values_ell.append(iou)

    cv2.imwrite(output_path, img)

print(np.mean(iou_values_circ))
print(np.mean(iou_values_ell))

per8C = 0
per8E = 0
for i in range(len(iou_values_circ)):
    if(iou_values_circ[i] >= 0.8):
        per8C += 1
for i in range(len(iou_values_ell)):
    if(iou_values_ell[i] >= 0.8):
        per8E += 1

per8C = per8C/len(iou_values_circ)
per8E = per8E/len(iou_values_ell)

print(per8C)
print(per8E)

plt.figure(1, figsize=(20,10))
plt.subplot(121)
plt.hist(iou_values_circ, bins=10, range=(0,1))
plt.title("Histogram of IoU Values - Hough Circles")
plt.xlabel("IoU")
plt.ylabel("Frequency")
plt.subplot(122)
plt.hist(iou_values_ell, bins=10, range=(0,1))
plt.title("Histogram of IoU Values - Canny")
plt.xlabel("IoU")
plt.ylabel("Frequency")
plt.savefig("iou_vals.png")

# Plotting the histogram
plt.hist(iou_values_circ, bins=10, range=(0,1))
plt.title("Histogram of IoU Values - Hough Circles")
plt.xlabel("IoU")
plt.ylabel("Frequency")
plt.show()
plt.hist(iou_values_ell, bins=10, range=(0,1))
plt.title("Histogram of IoU Values - Canny")
plt.xlabel("IoU")
plt.ylabel("Frequency")
plt.savefig("canny_iou.png")
