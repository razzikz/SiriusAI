import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer
from PIL import Image

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
    count = 0
    mask = []
    for pol in y1[i]:
        if count == 0:
            points = np.array(pol).astype(np.int_).reshape(1, -1, 2)
            mask = cv2.rectangle(x1[i], (points[0][0][0], points[0][0][1]), (points[0][1][0], points[0][1][1]), (0, 255, 0), 2)
            count += 1
    y_final.append(mask)


x, y = np.asarray(x1).astype(np.float_), np.asarray(y_final).astype(np.float_)


for row in range(1, len(y_final)):
    for col in range(1, len(y_final[row])):
        cv2.imshow("image", np.array(y_final[row][col]))
        cv2.waitKey(0)

cv2.destroyAllWindows()

if __name__ == "__main__":
    print("LoadData.py is runned!")
