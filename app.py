import cv2
import edgeiq

import numpy as np
#from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
#from skimage.color import rgb2lab, deltaE_cie76
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import os


"""
Use object detection to detect objects on a batch of images. The types
of objects detected can be changed by selecting different models.
Different images can be used by updating the files in the *images/*
directory. Note that when developing for a remote device, removing
images in the local *images/* directory won't remove images from the
device. They can be removed using the `aai app shell` command and
deleting them from the *images/* directory on the remote device.

To change the computer vision model, follow this guide:
https://dashboard.alwaysai.co/docs/application_development/changing_the_model.html

To change the engine and accelerator, follow this guide:
https://dashboard.alwaysai.co/docs/application_development/changing_the_engine_and_accelerator.html
"""

flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
len(flags)
flags[40]

##### for pie chart pixel analyzation
# def get_colors(image, num_colors, show_chart):
#     modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
#     modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)

#     clf = KMeans(n_clusters = number_of_colors)
#     labels = clf.fit_predict(modified_image)

#     counts = Counter(labels)

#     center_colors = clf.cluster_centers_
#  ordered colors by iterating through the keys
#     ordered_colors = [center_colors[i] for i in counts.keys()]
#     hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
#     rgb_colors = [ordered_colors[i] for i in counts.keys()]

#     if (show_chart):
#         plt.figure(figsize = (8, 6))
#         plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)

#     return rgb_colors


def rgb2hex(rgb: np.ndarray) -> str:
    # rgb = rgb.reshape(3, ...) //don't need cuz swap already formats as needed
    return "#{:02x}{:02x}{:02x}".format(*rgb)

#bgr to rgb converter
def swap(inlist): 
    # list[pos1], list[pos2] = list[pos2], list[pos1] //only swaps indexes, not the actual object of the list to located objects
    ret = [[hex(int(tup[2])), hex(int(tup[1])), hex(int(tup[0]))] for tup in inlist] #typecasting tuples within a list of lists
    return ret

##### pie chart code
def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def main():
    obj_detect = edgeiq.ObjectDetection(
            "alwaysai/ssd_mobilenet_v1_coco_2018_01_28")
    obj_detect.load(engine=edgeiq.Engine.DNN)

    print("Engine: {}".format(obj_detect.engine))
    print("Accelerator: {}\n".format(obj_detect.accelerator))
    print("Model:\n{}\n".format(obj_detect.model_id))
    print("Labels:\n{}\n".format(obj_detect.labels))

    image_paths = sorted(list(edgeiq.list_images("images/")))
    print("Images:\n{}\n".format(image_paths))

    with edgeiq.Streamer(
            queue_depth=len(image_paths), inter_msg_time=3) as streamer:
        for image_path in image_paths:
            # Load image from disk
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            r, g, b = cv2.split(image)
            fig = plt.figure()
            axis = fig.add_subplot(1, 1, 1, projection="3d")

            pixel_colors = image.reshape((np.shape(image)[0]*np.shape(image)[1], 3))
            norm = colors.Normalize(vmin=-1., vmax=1.)
            norm.autoscale(pixel_colors)
            pixel_colors = norm(pixel_colors).tolist()

            axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
            axis.set_xlabel("Red")
            axis.set_ylabel("Green")
            axis.set_zlabel("Blue")
            plt.show()

             #convert from rgb to hsv and pick out 2 shades
            hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv_drot = (18, 24, 61)
            hsv_lrot = (13, 203, 55)

            #build the color mask
            mask = cv2.inRange(hsv_image, hsv_lrot, hsv_drot)
            res = cv2.bitwise_and(image, image, mask=mask)
            plt.subplot(1, 2, 1)
            plt.imshow(mask, cmap="gray")
            plt.subplot(1, 2, 2)
            plt.imshow(res)
            plt.show()

            #2nd layer mask, did not display
            hsv_olive = (34, 32, 120)
            hsv_dolive = (37, 240, 27)
            mask_ol = cv2.inRange(hsv_image, hsv_olive, hsv_dolive)
            res_w = cv2.bitwise_and(image, image, mask=mask_ol)
            plt.subplot(1, 2, 1)
            plt.imshow(mask_ol, cmap="gray")
            plt.subplot(1, 2, 2)
            plt.imshow(res_w)
            plt.show()

            #final mask
            final_mask = mask + mask_ol
            final_result = cv2.bitwise_and(image, image, mask=final_mask)
            plt.subplot(1, 2, 1)
            plt.imshow(final_mask, cmap="gray")
            plt.subplot(1, 2, 2)
            plt.imshow(final_result)
            plt.show()

            #testing .shape and typecast image
            print("The type of this input is {}".format(type(image)))
            print("Shape: {}".format(image.shape))


            #piee
            ##text.append(get_colors(get_image(image_path), 4, True))

            # Generate text to display on streamer
            text = ["Model: {}".format(obj_detect.model_id)]
            text.append("Inference time: {:1.3f} s".format(results.duration))

            #need to convert from bgr to rgb
            swapped_colors = swap(obj_detect.colors)
            text.append("Colors printed!")
            # text.append(swapped_colors)

            print(swapped_colors)

            # print(obj_detect.colors)

            # converted = np.array([np.array(rgb) for rgb in swapped_colors]) // numpy arrays with lists (like numpy contained within itself, list of lists)

            # print(converted.shape)

            results = obj_detect.detect_objects(image, confidence_level=.5)

            image = edgeiq.markup_image(
                    image, results.predictions, colors=obj_detect.colors)
            # print(rgb2hex(swapped_colors))

            # print(converted)

            # iterate through tuple list and convert
            # for x in obj_detect.colors:
            #     text.append(rgb2hex(swapped_colors))
            #     text.append(format(x))
            
            text.append("Objects:")

            for prediction in results.predictions:
                text.append("{}: {:2.2f}%".format(
                    prediction.label, prediction.confidence * 100))

            streamer.send_data(image, text) 

        streamer.wait()

    print("Program Ending")


if __name__ == "__main__":
    main()
