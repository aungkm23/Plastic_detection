from PIL import Image
import glob
import cv2
import xml.etree.ElementTree as ET

def DoG(img):
    # fn = raw_input("Enter image file name and path: ") #cv2.imshow here
    # fn_no_ext = fn.split('.')[0]
    # outputFile = fn_no_ext + 'DoG.jpg'
    # read the input file


    # run a 5x5 gaussian blur then a 3x3 gaussian blr
    blur5 = cv2.GaussianBlur(img, (5, 5), 0)
    blur3 = cv2.GaussianBlur(img, (3, 3), 0)

    # write the results of the previous step to new files
    # cv2.imshow('3x3', blur3)
    # cv2.imshow('5x5', blur5)

    DoGim = blur5 - blur3
    # cv2.imshow("outputFile", DoGim)
    # cv2.waitKey(0)
    return DoGim


# img_path = 'Image/00000850/20151101_072507.jpg'
image_list = []
imgdict = {}
resized_images = []

for filename in glob.glob('images/*.tif'):
    # print(filename)
    # img = Image.open(filename)
    img = cv2.imread(filename)
    # print(img.shape)
    # img = cv2.resize(img, (400,400))
    image_list.append(img)
    name = filename.split('/')[1]
    imgdict[name] = img

labels = {}
for filename in glob.glob('labels/*.xml'):
    tree = ET.parse(filename)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        bbox = [float(obj.find('bndbox/xmin').text), float(obj.find('bndbox/ymin').text),
                float(obj.find('bndbox/xmax').text), float(obj.find('bndbox/ymax').text)]
        boxes.append(bbox)
    name = filename.split('/')[1]
    labels[name] = boxes[0]

print(labels["000000033107.xml"])

# for image in image_list:
#     image = DoG(image)
#     resized_images.append(image)


def display_image_with_box(image, bbox):
    image_with_box = image.copy()
    # for i in bbox:
    #     cv2.rectangle(image_with_box, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (0, 255, 0), thickness=2)
    cv2.rectangle(image_with_box, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), thickness=2)
    cv2.imshow("Image", image_with_box)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


display_image_with_box(imgdict["000000000987.tif"], labels["000000000987.xml"])

def apply_blurring(): 
    
    return null


# for (i, new) in enumerate(resized_images):
#     cv2.imwrite( 'DogImage/test2/'+ str(i+77)+'.jpg', new)
#     # new.save('{}{}{}'.format('DogImage/test1/pic', i+1, '.jpg'))
# cv2.imshow('000000000435.tif dog', DoG(mydict["000000000435.tif"]))
# cv2.imshow('000000000435.tif', mydict["000000000435.tif"])
# # cv2.imshow('test', image_list[300])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print(len(image_list))
# print(len(resized_images))
for i in range(len(image_list)):
    result = cv2.resize(image_list[i], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
print(len(result))