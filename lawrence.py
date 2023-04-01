import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2

def load_data(image_folder, annotation_folder):
    filenames, images, bounding_boxes = [], [], []

    for filename in os.listdir(annotation_folder):
        if filename.endswith(".xml"):
            xml_path = os.path.join(annotation_folder, filename)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # check if the name tag is 5, if yes, then skip this file
            name_tag = root.find('object/name')
            if name_tag is not None and name_tag.text == '5':
                continue

            img_path = os.path.join(image_folder, root.find('filename').text)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = cv2.resize(img, (224, 224))

            boxes = []
            for obj in root.findall('object'):
                bbox = [float(obj.find('bndbox/xmin').text), float(obj.find('bndbox/ymin').text),
                        float(obj.find('bndbox/xmax').text), float(obj.find('bndbox/ymax').text)]
                boxes.append(bbox)

            filenames.append(filename)
            images.append(img)
            bounding_boxes.append(boxes)

    return filenames, np.array(images), bounding_boxes

image_folder = r'images'
annotation_folder = r'labels'
filenames, images, bounding_boxes = load_data(image_folder, annotation_folder)



def display_image_with_box(image, bbox):
    image_with_box = image.copy()
    for i in bbox:
        cv2.rectangle(image_with_box, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (0, 255, 0), thickness=2)
    cv2.imshow("Image", image_with_box)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Display first image with its bounding box
display_image_with_box(images[3602], bounding_boxes[3602])

for i, j in enumerate(filenames):
    if "000000000984" in j:
        display_image_with_box(images[i], bounding_boxes[i])

