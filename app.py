import cv2
import edgeiq

import numpy as np

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

def rgb2hex(rgb: np.ndarray) -> str:
    # rgb = rgb.reshape(3, ...) //don't need cuz swap already formats as needed
    return "#{:02x}{:02x}{:02x}".format(*rgb)

#bgr to rgb converter
def swap(inlist): 
    # list[pos1], list[pos2] = list[pos2], list[pos1] //only swaps indexes, not the actual object of the list to located objects
    ret = [[hex(int(tup[2])), hex(int(tup[1])), hex(int(tup[0]))] for tup in inlist] #typecasting tuples within a list of lists

    return ret

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

            results = obj_detect.detect_objects(image, confidence_level=.5)

            image = edgeiq.markup_image(
                    image, results.predictions, colors=obj_detect.colors)

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
