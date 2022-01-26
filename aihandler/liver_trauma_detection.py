
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
    list1=[]
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
        list1.append(im11)
    dcm = pydicom.dcmread(all_files[0])
    patientID = (dcm.PatientID)
    return [patientID,list1]


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
            return ("Algorithm Diagnosis: Trauma ("+str(x[0]*100)+"%)")
        else:
            return ("Algorithm Diagnosis: Normal ("+str(x[1]*100)+"%)")


def PDF_report(patientID, st1, img_path, dest_path, reportPath,img111):

    import matplotlib.pyplot as plt
    from skimage import data, filters

    fig, ax = plt.subplots(nrows=1, ncols=5)
    fig.set_size_inches(18.5, 10.5)

    image = img111
    edges = filters.sobel(image)

    low = 0.1
    high = 0.35

    lowt = (edges > low).astype(int)
    hight = (edges > high).astype(int)
    hyst = filters.apply_hysteresis_threshold(edges, low, high)

    # ax[0].imshow(liver_image, cmap='gray')
    # ax[0].set_title('Original image')

    # ax[ 1].imshow(edges, cmap='magma')
    # ax[1].set_title('Sobel edges')

    # ax[2].imshow(lowt, cmap='magma')
    # ax[2].set_title('Low threshold')

    # ax[3].imshow(hight + hyst, cmap='magma')
    # ax[3].set_title('Hysteresis threshold')

    # for a in ax.ravel():
    #     a.axis('off')

    from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
    from skimage import filters

    @adapt_rgb(each_channel)
    def sobel_each(image):
        return filters.sobel(image)

    @adapt_rgb(hsv_value)
    def sobel_hsv(image):
        return filters.sobel(image)

    from skimage import data
    from skimage.exposure import rescale_intensity
    import matplotlib.pyplot as plt

    # image = data.astronaut()

    # fig, (ax_each, ax_hsv) = plt.subplots(ncols=2, figsize=(14, 7))

    # # We use 1 - sobel_each(image) but this won't work if image is not normalized
    # ax_each.imshow(rescale_intensity(1 - sobel_each(image)))
    # ax_each.set_xticks([]), ax_each.set_yticks([])
    # ax_each.set_title("Sobel filter computed\n on individual RGB channels")

    # # We use 1 - sobel_hsv(image) but this won't work if image is not normalized
    # ax_hsv.imshow(rescale_intensity(1 - sobel_hsv(image)))
    # ax_hsv.set_xticks([]), ax_hsv.set_yticks([])
    # ax_hsv.set_title("Sobel filter computed\n on (V)alue converted image (HSV)")

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    from skimage import data
    from skimage.filters import threshold_multiotsu

    # Setting the font size for all plots.
    matplotlib.rcParams['font.size'] = 9

    # The input image.
    # image = data.camera()

    # Applying multi-Otsu threshold for the default value, generating
    # three classes.
    thresholds = threshold_multiotsu(image)

    # Using the threshold values, we generate the three regions.
    regions = np.digitize(image, bins=thresholds)

    # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))

    # Plotting the original image.
    # ax[0].imshow(image, cmap='gray')
    # ax[0].set_title('Original')
    # ax[0].axis('off')

    # # Plotting the histogram and the two thresholds obtained from
    # # multi-Otsu.
    # ax[1].hist(image.ravel(), bins=255)
    # ax[1].set_title('Histogram')
    # for thresh in thresholds:
    #     ax[1].axvline(thresh, color='r')

    # # Plotting the Multi Otsu result.
    # ax[2].imshow(regions, cmap='jet')
    # ax[2].set_title('Multi-Otsu result')
    # ax[2].axis('off')

    # plt.subplots_adjust()

    # plt.show()

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import numpy as np

    plt.figure()
    ax = plt.gca()
    im = ax.imshow(img111, cmap='gray')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.savefig("foo1.png", bbox_inches='tight')

    plt.figure()
    ax = plt.gca()
    im = ax.imshow(edges, cmap='magma')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.savefig("foo2.png", bbox_inches='tight')

    plt.figure()
    ax = plt.gca()
    im = ax.imshow(rescale_intensity(1 - sobel_hsv(image)))
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.savefig("foo3.png", bbox_inches='tight')

    plt.figure()
    ax = plt.gca()
    im = ax.imshow(regions, cmap='jet')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.savefig("foo4.png", bbox_inches='tight')

    from skimage import data, io, segmentation, color
    from skimage.future import graph
    import numpy as np

    def _weight_mean_color(graph, src, dst, n):
        """Callback to handle merging nodes by recomputing mean color.

        The method expects that the mean color of `dst` is already computed.

        Parameters
        ----------
        graph : RAG
            The graph under consideration.
        src, dst : int
            The vertices in `graph` to be merged.
        n : int
            A neighbor of `src` or `dst` or both.

        Returns
        -------
        data : dict
            A dictionary with the `"weight"` attribute set as the absolute
            difference of the mean color between node `dst` and `n`.
        """

        diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
        diff = np.linalg.norm(diff)
        return {'weight': diff}

    def merge_mean_color(graph, src, dst):

        graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
        graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
        graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                            graph.nodes[dst]['pixel count'])

    img = img111
    labels = segmentation.slic(
        img, compactness=30, n_segments=4000, start_label=1)
    g = graph.rag_mean_color(img, labels)

    labels2 = graph.merge_hierarchical(labels, g, thresh=35, rag_copy=False,
                                        in_place_merge=True,
                                        merge_func=merge_mean_color,
                                        weight_func=_weight_mean_color)

    out = color.label2rgb(labels2, img, kind='avg', bg_label=0)
    out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))

    plt.figure()
    ax = plt.gca()
    im = ax.imshow(out, cmap='gray')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.savefig("foo5.png", bbox_inches='tight')

    # plt.show()
    # Add a page
    pdf.add_page()

    # set style and size of font
    # that you want in the pdf
    pdf.set_font("Arial", size=15)
    # final_image = np.uint8(im111)
    # final_image = Image.fromarray(final_image)
    # final_image.save("./"+f"/{1}.png")

    from PIL import Image

    new_im = Image.new(
        'RGB', (Image.open("./foo1.png").size[0]*5, Image.open("./foo1.png").size[1]))

    x_offset = 0

    new_im.paste(Image.open("./foo1.png"), (x_offset, 0))
    x_offset += Image.open("./foo1.png").size[0]
    new_im.paste(Image.open("./foo5.png"), (x_offset, 0))
    x_offset += Image.open("./foo5.png").size[0]
    new_im.paste(Image.open("./foo2.png"), (x_offset, 0))
    x_offset += Image.open("./foo2.png").size[0]
    new_im.paste(Image.open("./foo3.png"), (x_offset, 0))
    x_offset += Image.open("./foo3.png").size[0]
    new_im.paste(Image.open("./foo4.png"), (x_offset, 0))
    x_offset += Image.open("./foo4.png").size[0]

    new_im.save('foo6.png')
  # add another cell
    pdf.cell(200, 10, txt="REPORT",
           ln=2, align='C')
    pdf.cell(200, 10, txt=" ",
           ln=7, align='L')
  # create a cell
    pdf.cell(200, 10, txt=st1,
           ln=3, align='L')

    pdf.cell(200, 10, txt=" ",
           ln=4, align='L')
    pdf.cell(200, 10, txt=" ",
           ln=5, align='L')
    pdf.cell(200, 10, txt=" ",
           ln=6, align='L')

    pdf.image("./foo6.png", h=35)

    pdf.cell(200, 10, txt=" ",
           ln=5, align='L')

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
    ret_val = loadFiles(str(dicomResultPath), str(pngResultPath))
    patientID=ret_val[0]
    str1 = report_gen(my_predict(str(pngResultPath)))
    PDF_report(patientID, str1, os.path.join(str(pngResultPath), "0.png"), str(pdfResultPath), str(reportZipPath),ret_val[1][0])

# read workingDir/result_pdf/report.zip
