import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.preprocessing import LabelBinarizer

class_names = ['people', 'body', 'head']

data_img = "/Users/vladimir/PycharmProjects/SiriusPeopleDetection/People detection/train_image/"
data_xml = "/Users/vladimir/PycharmProjects/SiriusPeopleDetection/People detection/train_xml/"

x1, y1 = [], []

for img_file in os.listdir(data_img):
    if img_file == ".DS_Store":
        continue
    img_path = os.path.join(data_img, img_file)
    img = cv2.imread(img_path)
    x1.append(img)

    xml_file = img_file.replace(".jpg", ".xml")
    xml_path = os.path.join(data_xml, xml_file)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    arr = []

    for obj in root.findall('object'):
        name = obj.find('name').text
        if name in class_names:
            xmin = int(obj.find('bndbox').find('xmin').text)
            xmax = int(obj.find('bndbox').find('xmax').text)
            ymin = int(obj.find('bndbox').find('ymin').text)
            ymax = int(obj.find('bndbox').find('ymax').text)

            arr.append([xmin, ymin, xmax, ymax])

    y1.append(arr)

y = y1
i = 0
while i < len(y):
    if len(y[i]) != 3:
        x1.pop(i)
        y.pop(i)
    else:
        i += 1

label = LabelBinarizer()
x_tensor, y_tensor = np.asarray(x1).astype(np.float_), np.asarray(y).astype(np.float_)
print(y_tensor[0])
class_names = label.fit_transform(class_names)
print(type(class_names))

# for i in range(617, len(y_train)):
#     for boxes in y_train[i]:
#         cv2.rectangle(x_train[i], (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (0, 255, 0), 2)
#         cv2.putText(x_train[i], boxes[4], (int(boxes[0]), int(boxes[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#     cv2.imshow('Image', x_train[i])
#     cv2.waitKey(0)
#
# cv2.destroyAllWindows()