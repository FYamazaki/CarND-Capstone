import numpy as np
import os
import sys
import tensorflow as tf
from collections import defaultdict
import cv2

# This is needed since the notebook is stored in the object_detection folder.
#sys.path.append("..")
#from object_detection.utils import ops as utils_ops
#from object_detection.utils import label_map_util

# What model to download.
TOP_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
#MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = TOP_DIR + '/' + MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

class ObjectDetector():
    def __init__(self):
        
        # Load a (frozen) Tensorflow model into memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

                config = tf.ConfigProto()
                # dynamically grow the memory used on the GPU
                config.gpu_options.allow_growth = True

                self.sess = tf.Session(graph=self.detection_graph, config=config)
                self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
    
    def run_inference_for_single_image(self, image):
        with self.detection_graph.as_default():
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image, axis=0)
            # Actual detection.
            (boxes, scores, classes, num_detections) = self.sess.run(
                  [self.boxes, self.scores, self.classes, self.num_detections],
                  feed_dict={self.image_tensor: image_np_expanded})

            return num_detections[0], boxes[0], classes[0].astype(np.uint8), scores[0]

    def __del__(self):
        sess.close()
