import os
import cv2
import xml.etree.ElementTree as ET

from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer
import numpy as np

class_names = ['people', 'head', 'leg', 'detect', 'undetected']

data_img = "/Users/vladimir/PycharmProjects/SiriusPeopleDetection/database_xml/train_img/"
data_xml = "/Users/vladimir/PycharmProjects/SiriusPeopleDetection/database_xml/train_xml/"

x1, y1, y_pol = [], [], []

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
    arr_pol_train = []

    for obj in root.findall('object'):
        arr_pol = []
        name = obj.find('name').text
        if name in class_names:
            xmin = float(obj.find('bndbox').find('xmin').text)
            xmax = float(obj.find('bndbox').find('xmax').text)
            ymin = float(obj.find('bndbox').find('ymin').text)
            ymax = float(obj.find('bndbox').find('ymax').text)

            if obj.find('polygon') is not None:
                for polygon in obj.find('polygon'):
                    arr_pol.append(float(polygon.text))

            if name in ['detect', 'undetected']:
                arr_pol_train.append(arr_pol)
            else:
                arr.append([xmin, ymin, xmax, ymax])
    y_pol.append(arr_pol_train)
    y1.append(arr)

y_final = []
for i in range(len(y1)):
    for pol in y1[i]:
        points = np.array(pol).astype(np.int_).reshape(1, -1, 2)
        mask = cv2.polylines(x1[i], [points], True, (0, 255, 0), 2)
        y_final.append(mask)

print(len(y_final))

x, y = np.asarray(x1).astype(np.float_), np.asarray(y_final).astype(np.float_)

print(x.shape, y.shape)
