from AI import *


import pydicom
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from skimage import morphology
import glob
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance, PILLOW_VERSION
import pydicom.uid
from scipy import ndimage
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
from fpdf import FPDF
import datetime
from time import gmtime, strftime
import shutil

#test Show
img = cv2.imread("./drive/MyDrive/tests/logo.jpg")
show(img)

# load test samples

file_path="./drive/My Drive/tests/FILE0.dcm"
medical_image = pydicom.read_file(file_path)
image = medical_image.pixel_array
print("main image")
show(image)

# test fuction transform_to_hu
hu_image = transform_to_hu(medical_image,image)
print("hu_image")
show(hu_image)

# test fuction window_image
liver_image = window_image(hu_image, 40, 100)
print("window_image")
show(liver_image)

# test fuction remove_noise
im11=remove_noise(file_path)
print("remove noise")

# output of test
print("output of preprocessing for input")
show(im11)

# test for function loadFiles
loadFiles("./drive/MyDrive/PatientData/*/*","./drive/MyDrive/sample_png1")

# test for creat_model
create_model()

# test for my_predict
res1=my_predict('./drive/MyDrive/sample_png1/')
print(res1)

# test for report_gen
report_gen(res1)

# Test for Main Function
main_func("./drive/MyDrive/PatientData")