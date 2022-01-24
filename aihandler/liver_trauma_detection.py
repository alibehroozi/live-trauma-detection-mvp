
"""UseTrainedModel (2).ipynb

Original file is located at
    https://colab.research.google.com/drive/1TiTCJp_Ono1unIR5qT5HUUTmLx4bb2_g
"""

import pydicom
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from skimage import morphology
import glob
# import pandas as pd
from PIL import Image
import pydicom.uid
from scipy import ndimage
# import seaborn as sns
# import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
# from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from fpdf import FPDF
import datetime
from time import gmtime, strftime
import shutil
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def show(im):
    return
    # plt.figure()
    # plt.imshow(im, cmap="gray")


def transform_to_hu(medical_image, image):
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    hu_image = image * slope + intercept

    return hu_image


def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max

    return window_image


def remove_noise(file_path, display=True):
    medical_image = pydicom.read_file(file_path)
    image = medical_image.pixel_array

    hu_image = transform_to_hu(medical_image, image)
    brain_image = window_image(hu_image, 40, 80)

    segmentation = morphology.dilation(brain_image, np.ones((5, 5)))
    labels, label_nb = ndimage.label(segmentation)

    label_count = np.bincount(labels.ravel().astype(np.int))

    label_count[0] = 0

    mask = labels == label_count.argmax()

    mask = morphology.dilation(mask, np.ones((5, 5)))
    mask = ndimage.morphology.binary_fill_holes(mask)
    mask = morphology.dilation(mask, np.ones((3, 3)))

    masked_image = mask * brain_image

    if display:
        plt.figure(figsize=(15, 2.5))
        plt.subplot(141)
        plt.imshow(brain_image, cmap="gray")
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(142)
        plt.imshow(mask, cmap="gray")
        plt.title('Mask')
        plt.axis('off')

        plt.subplot(143)
        plt.imshow(masked_image, cmap="gray")
        plt.title('Final Image')
        plt.axis('off')

    return masked_image


def loadFiles(sourcePath, destPath):
    all_files_level1 = glob.glob(sourcePath + "/*.dcm")
    all_files_level2 = glob.glob(sourcePath + "/*/*.dcm")
    i = 0
    all_files = all_files_level1+all_files_level2

    for file in all_files:
        file_path = file
        medical_image = pydicom.read_file(file)
        image = medical_image.pixel_array

        hu_image = transform_to_hu(medical_image, image)
        liver_image = window_image(hu_image, 40, 100)
        show(image)
        show(liver_image)
        im11 = remove_noise(file_path)
        final_image = np.uint8(im11)
        final_image = Image.fromarray(final_image)
        final_image.save(destPath+f"/{i}.png")
        i += 1
    dcm = pydicom.dcmread(all_files[0])
    patientID = (dcm.PatientID)
    return patientID


def create_model():
    model2 = Sequential()
    model2.add(Conv2D(32, 3, padding="same",
               activation="relu", input_shape=(224, 224, 3)))
    model2.add(MaxPool2D())

    model2.add(Conv2D(32, 3, padding="same", activation="relu"))
    model2.add(MaxPool2D())

    model2.add(Conv2D(64, 3, padding="same", activation="relu"))
    model2.add(MaxPool2D())
    model2.add(Dropout(0.4))

    model2.add(Flatten())
    model2.add(Dense(128, activation="relu"))
    model2.add(Dense(2, activation="softmax"))

    model2.summary()
    opt = Adam(lr=0.000001)
    model2.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True), metrics=['accuracy'])

    return model2


def my_predict(pathOfTestFile):
    model2 = create_model()
    checkpoint_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "models/cp.ckpt"))
    model2.load_weights(checkpoint_path)
    img_size = 224
    data = []
    path = pathOfTestFile
    for img in os.listdir(path):
        try:
            # convert BGR to RGB format
            img_arr = cv2.imread(os.path.join(path, img))[..., ::-1]
            # Reshaping images to preferred size
            resized_arr = cv2.resize(img_arr, (img_size, img_size))
            data.append(resized_arr)

        except Exception as e:
            print(e)
            print(img)
            # np.array(x_val) / 255
    val1 = np.array(data)/255
    return model2.predict(val1)


def report_gen(result):
    for x in result:
        if(x[0] > x[1]):
            return ("Algorithm Diagnosis: Teruma ("+str(x[0]*100)+"%)")
        else:
            return ("Algorithm Diagnosis: Normal ("+str(x[1]*100)+"%)")


def PDF_report(patientID, st1, img_path, dest_path, reportPath):
    pdf = FPDF()

    # Add a page
    pdf.add_page()

    # set style and size of font
    # that you want in the pdf
    pdf.set_font("Arial", size=15)
    pdf.image(img_path, w=58)
    # add another cell
    pdf.cell(200, 10, txt="REPORT.",
             ln=2, align='L')
    # create a cell
    pdf.cell(200, 10, txt=st1,
             ln=3, align='L')

    pdf.cell(200, 10, txt=" ",
             ln=4, align='L')
    pdf.cell(200, 10, txt=" ",
             ln=5, align='L')
    pdf.cell(200, 10, txt=" ",
             ln=5, align='L')
    pdf.cell(200, 10, txt=" ",
             ln=6, align='L')
    pdf.cell(200, 10, txt=" ",
             ln=7, align='L')

    datetime.datetime.now()

    pdf.cell(200, 10, txt=strftime("%Y-%m-%d %H:%M:%S", gmtime()),
             ln=10, align='R')

    pdf.output(dest_path+"/REPORT_"+patientID+".pdf")
    shutil.make_archive(reportPath+"/report", 'zip', dest_path)


def make_dirs(path1):
    pdfResultPath = Path(path1) / "result_pdf"
    dicomResultPath = Path(path1) / "result_dicom"
    pngResultPath = Path(path1) / "result_png"
    reportZipPath = Path(path1) / "report_final"
    if pdfResultPath.exists() and pdfResultPath.is_dir():
        shutil.rmtree(pdfResultPath)
    os.mkdir(pdfResultPath)
    if dicomResultPath.exists() and dicomResultPath.is_dir():
        shutil.rmtree(dicomResultPath)
    os.mkdir(dicomResultPath)
    if reportZipPath.exists() and reportZipPath.is_dir():
        shutil.rmtree(reportZipPath)
    os.mkdir(reportZipPath)
    if pngResultPath.exists() and pngResultPath.is_dir():
        shutil.rmtree(pngResultPath)
    os.mkdir(pngResultPath)

    return [pdfResultPath, dicomResultPath, pngResultPath, reportZipPath]


def unzip_files(sourceZipDir, dicomSourcesPath):
    shutil.unpack_archive(sourceZipDir, dicomSourcesPath)


def ai_main_func(workingDir, dicomsZipFile):
    pdfResultPath, dicomResultPath, pngResultPath, reportZipPath = make_dirs(
        workingDir)
    unzip_files(dicomsZipFile, str(dicomResultPath))
    patientID = loadFiles(str(dicomResultPath), str(pngResultPath))
    str1 = report_gen(my_predict(str(pngResultPath)))
    PDF_report(patientID, str1, os.path.join(
        str(pngResultPath), "0.png"), str(pdfResultPath), str(reportZipPath))

# read workingDir/result_pdf/report.zip
