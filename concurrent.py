import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from multiprocessing import Pool

def load_image(path): 
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_annotation(path):
    if filename.endswith(".xml"):
            xml_path = os.path.join(annotation_folder, filename)
            tree = ET.parse(xml_path)
            root = tree.getroot()

def load_single_file(args):
    image_folder, annotation_folder, filename = args
    # Load the image and bounding box data for the given filename
    # You need to replace the following lines with your custom loading function
    image = load_image(os.path.join(image_folder, filename))
    bounding_box = load_annotation(os.path.join(annotation_folder, filename))
    return filename, image, bounding_box

def load_data(image_folder, annotation_folder):
    filenames = os.listdir(image_folder)

    with Pool() as pool:
        args = [(image_folder, annotation_folder, filename) for filename in filenames]
        results = pool.map(load_single_file, args)

    filenames, images, bounding_boxes = zip(*results)
    
    return list(filenames), list(images), list(bounding_boxes)

image_folder = r'images'
annotation_folder = r'labels'
filenames, images, bounding_boxes = load_data(image_folder, annotation_folder)

