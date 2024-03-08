import tensorflow as tf
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("CrackDetector.keras")

allimages = []
nfiles = r"C:\Users\afnan\PycharmProjects\pythonProject\Crack\Negative\*"
pfiles = r"C:\Users\afnan\PycharmProjects\pythonProject\Crack\Positive\*"

for images in glob.glob(nfiles):
    img = cv2.imread(images)
    allimages.append(img)
for images in glob.glob(pfiles):
    img = cv2.imread(images)
    allimages.append(img)

allimages = np.array(allimages)

img = allimages[-20:-1]

print(model.predict(img))