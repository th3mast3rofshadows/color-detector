from matplotlib.figure import Figure
from sklearn.cluster import KMeans
import numpy as np
import cv2
from collections import Counter
import tkinter as tk
from tkinter import filedialog as fd
import pandas as pd
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
from PIL import Image, ImageTk

# Threshold to detect object
thres = 0.55
color_names = pd.read_csv("color_names.csv")


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def closest_color(rgb):
    colors = color_names[['R', 'G', 'B']].to_numpy()
    color = rgb
    distances = np.sqrt(np.sum((colors - color) ** 2, axis=1))
    index_of_smallest = np.where(distances == np.amin(distances))
    df_item = color_names[(color_names['R'] == colors[index_of_smallest][0][0])
                          & (color_names['G'] == colors[index_of_smallest][0][1])
                          & (color_names['B'] == colors[index_of_smallest][0][2])].iloc[0]

    return df_item['color_name']


def quantize_colors(img, n_colors=6):
    arr = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(arr)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    return centers[labels].reshape(img.shape).astype('uint8')


def get_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def plot_colors(img, number_of_colors):
    # img = quantize_colors(img, number_of_colors)

    # First, we resize the image to the size 600 x 400.
    # It is not required to resize it to a smaller size but we do so to lessen
    # the pixels which’ll reduce the time needed to extract the colors from the image.
    # KMeans expects the input to be of two dimensions,
    # so we use Numpy’s reshape function to reshape the image data.
    modified_image = cv2.resize(img, (600, 400), interpolation=cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0] * modified_image.shape[1], 3)

    # KMeans algorithm creates clusters based on the supplied count of clusters.
    # In our case, it will form clusters of colors and these clusters will be our top colors.
    clf = KMeans(n_clusters=number_of_colors)

    # We then fit and predict on the same image to extract the prediction into the variable labels.
    labels = clf.fit_predict(modified_image)

    # We use Counter to get count of all labels.
    counts = Counter(labels)

    # To find the colors, we use clf.cluster_centers_
    center_colors = clf.cluster_centers_

    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    named_colors = [closest_color(ordered_colors[i]) for i in counts.keys()]
    labels = [name + ' (' + hex + ')' for name, hex in zip(named_colors, hex_colors)]

    fig = Figure()
    ax = fig.add_subplot(111)
    ax.pie(counts.values(), labels=labels, colors=hex_colors)

    return fig


def getObjects(img):
    classFile = 'coco.names'

    with open(classFile, 'rt') as f:
        class_names = f.read().rstrip('\n').split('\n')

    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    objects = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            x1, y1, width, height = box
            x2, y2 = x1 + width, y1 + height
            objects.append(img[y1:y2, x1:x2])

    if not objects:
        objects.append(img)

    return objects


def getObjects(file_name):
    thres = 0.45  # Threshold to detect object

    classFile = 'coco.names'

    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(640, 640)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    img = cv2.imread(file_name)
    try:
        classIds, confs, bbox = net.detect(img, confThreshold=thres)
        print(classIds, bbox)
    except Exception as e:
        print(str(e))
        exit()

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    return img


def openImage():
    global plotCanvas, filename

    if plotCanvas:
        plotCanvas.get_tk_widget().destroy()

    filename = fd.askopenfilename()
    image = Image.open(filename)
    photo = ImageTk.PhotoImage(image)
    my_label = tk.Label(image=photo)
    my_label.image = photo
    my_label.grid(column=0, row=0, sticky=tk.W, padx=5, pady=5)


def plotColors():
    global filename
    plot = plot_colors(get_image(filename), 5)

    plotCanvas = FigureCanvasTkAgg(plot, root)
    plotCanvas.get_tk_widget().grid(column=2, row=0, sticky=tk.E, padx=5, pady=5)
    plotCanvas.draw()


def detectObjects():
    global filename
    img = getObjects(filename)
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    photo = ImageTk.PhotoImage(image)
    my_label = tk.Label(image=photo)
    my_label.image = photo
    my_label.grid(column=2, row=2, sticky=tk.W, padx=5, pady=5)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    root = tk.Tk()
    root.title('Image Color Plotting')
    root.resizable(False, False)

    root.columnconfigure(0, weight=4)
    root.columnconfigure(1, weight=1)
    root.columnconfigure(2, weight=4)

    plotCanvas = None
    filename = None

    openImage_btn = tk.Button(root, text='Open Image', command=openImage, fg='white')
    openImage_btn.grid(column=0, row=1, sticky=tk.W, padx=5, pady=5)

    openImage_btn = tk.Button(root, text='Plot Colors', command=plotColors, fg='white')
    openImage_btn.grid(column=1, row=0, sticky=tk.W, padx=5, pady=5)

    openImage_btn = tk.Button(root, text='Identify Objects', command=detectObjects, fg='white')
    openImage_btn.grid(column=1, row=1, sticky=tk.W, padx=5, pady=5)

    root.mainloop()

    # mask = Image.new("L", im2.size, 0)
    # x, y = im2.size
    # draw = ImageDraw.Draw(mask)
    # draw.ellipse((120, 30, 200, 100), fill=255)
    # composite = Image.composite(im1, im2, mask)
    #
    # composite.convert("RGB").save('images/masked_result.jpg', quality=95)
    #
    # composite.show()

    # image = cv2.imread(file.name)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # objects = get_objects(image)
    #
    # for i, obj in enumerate(objects):
    #     plt.imshow(obj)
    #
    # get_colors(objects[0], 5, True)
