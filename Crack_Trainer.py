import cv2
import tensorflow as tf
import glob
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D
from keras.models  import Sequential
import numpy as np
from sklearn.model_selection import train_test_split

def ClassificationModel():
    model = Sequential()
    model.add(Conv2D(64,kernel_size=(3,3),strides=(1,1),activation='relu',input_shape=(227,227,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer='adam')
    return model

allimages = []
train = []
nfiles = r"C:\Users\afnan\PycharmProjects\pythonProject\Crack\Negative\*"
pfiles = r"C:\Users\afnan\PycharmProjects\pythonProject\Crack\Positive\*"

for images in glob.glob(nfiles):
    img = cv2.imread(images)
    allimages.append(img)
    train.append(0)
for images in glob.glob(pfiles):
    img = cv2.imread(images)
    allimages.append(img)
    train.append(1)

allimages = np.array(allimages)
train = np.array(train)

X_train, X_test, y_train, y_test = train_test_split(allimages, train, test_size=0.33, random_state=42)

model = ClassificationModel()
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=5)



model.save("CrackDetector.keras")
