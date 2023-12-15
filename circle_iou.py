import math
import numpy as np


def circle_area(radius):
    return np.pi * (radius ** 2)


def distance_between_centers(c1, c2):
    return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def circle_intersection_area(c1, c2):
    d = distance_between_centers(c1, c2)

    # If one circle is contained within the other
    if d <= abs(c1[2] - c2[2]):
        return np.pi * min(c1[2], c2[2]) ** 2

    # If no intersection
    if d >= c1[2] + c2[2] or d == 0:
        return 0

    # Using the circle intersection formula
    r1, r2 = c1[2], c2[2]
    part1 = r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2 * d * r1))
    part2 = r2**2 * np.arccos((d**2 + r2**2 - r1**2) / (2 * d * r2))
    part3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2)
                          * (d - r1 + r2) * (d + r1 + r2))

    return part1 + part2 - part3


def iou_circles(c_groundTruth, c_generated):
    area_intersect = circle_intersection_area(c_groundTruth, c_generated)
    area_union = circle_area(
        c_groundTruth[2]) + circle_area(c_generated[2]) - area_intersect
    return area_intersect / area_union


# Example circles
c_groundTruth = [0, 0, 5]  # x, y, r
c_generated = [3, 3, 4]    # x, y, r

iou = iou_circles(c_groundTruth, c_generated)
print(iou)
