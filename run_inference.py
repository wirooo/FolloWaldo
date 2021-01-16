import tensorflow as tf
import numpy as np
import os
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import argparse
import os


DEFAULT_MODEL_PATH = "./exported-models/my_model/saved_model"
DEFAULT_LABEL_MAP_PATH = "./annotations/label_map.pbtxt"
DEFAULT_IMAGE_INPUT_PATH = "./images/input"
DEFAULT_IMAGE_OUTPUT_PATH = "./images/output"


def inference(inputPath, modelPath, labelPath, outputPath):
    print("Loading model... ", end='')

    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(modelPath)

    # Loading the label_map
    category_index = label_map_util.create_category_index_from_labelmap(labelPath, use_display_name=True)

    print("Done!")

    def load_image_into_numpy_array(path):
        # Load an image from file into a numpy array.
        return np.array(Image.open(path))

    def predict(inputPath, imageName, modelPath, labelPath, outputPath):
        print("Running inference for {}... ".format(os.path.join(inputPath, imageName)), end='')
        image_np = load_image_into_numpy_array(os.path.join(inputPath, imageName))

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`
        input_tensor = tf.convert_to_tensor(image_np)
        # The model expects a batch of images, so add an axis with `tf.newaxis`
        input_tensor = input_tensor[tf.newaxis, ...]

        # input_tensor = np.expand_dims(image_np, 0)
        try:
            detections = detect_fn(input_tensor)
        except ValueError:
            print("Oops!")
            continue

        # Convert to numpy arrays, and take index [0] to remove the batch dimension
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # Detection_classes should be ints
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=1,
            min_score_thresh=.08,
            agnostic_mode=False)

        image = Image.fromarray(image_np_with_detections)
        image.save(os.path.join(outputPath, imageName))
        print("Done!")


    if os.path.isdir(inputPath):
        for imageName in [i for i in os.listdir(inputPath) if os.path.isfile(os.path.join(inputPath, i))]:
            predict(inputPath, imageName, modelPath, labelPath, outputPath)
    else:
        predict("", inputPath, modelPath, labelPath, outputPath)        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Let's find Waldo!")
    parser.add_argument("--input", type=str, nargs="?", default=DEFAULT_IMAGE_INPUT_PATH)
    parser.add_argument("--model", type=str, nargs="?", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--labels", type=str, nargs="?", default=DEFAULT_LABEL_MAP_PATH)
    parser.add_argument("--output", type=str, nargs="?", default=DEFAULT_IMAGE_OUTPUT_PATH)
    args = parser.parse_args()
    inference(inputPath=args.input, modelPath=args.model, labelPath=args.labels, outputPath=args.output)
