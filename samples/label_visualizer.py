from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import re
from shapely.geometry import Polygon, Point


def extract_coordinates(polygon_element):
    # This helper function extracts (x,y) coordinates from the polygon element
    coords = []
    for coord in polygon_element:
        if re.match(r'x\d+', coord.tag):
            x = float(coord.text)
        if re.match(r'y\d+', coord.tag):
            y = float(coord.text)
            coords.append((x, y))
    return coords


# Load the image
image_path = './Images/D6.jpg'
image = Image.open(image_path)
circle_generated = [615, 254, 72]

# Parse the XML content
xml_content = './Labels/D6.xml'
# xml_tree = ET.ElementTree(ET.fromstring(xml_content))
xml_tree = ET.parse(xml_content)

# Initialize a drawing context
draw = ImageDraw.Draw(image)

# Draw polygons based on the XML annotation
for object_tag in xml_tree.getroot().iter('object'):
    polygon_tag = object_tag.find('polygon')
    if polygon_tag is not None:
        coordinates = extract_coordinates(polygon_tag)
        if coordinates:
            draw.polygon(coordinates, outline='red')
        # create a polygon
        polygon = Polygon(coordinates)
        polygon_area = polygon.area
        print("polygon area: {}".format(polygon_area))
        # create the circle
        circle = Point(circle_generated[0], circle_generated[1]).buffer(
            circle_generated[2])
        # calculate the Intersectoin over Union (IoU)
        intersection = polygon.intersection(circle).area
        union = polygon.union(circle).area
        iou = intersection/union if union else 0
        print("IoU: {}".format(iou))

# Save or display the annotated image
# annotated_image_path = 'annotated.jpg'
# image.save(annotated_image_path)
image.show()
