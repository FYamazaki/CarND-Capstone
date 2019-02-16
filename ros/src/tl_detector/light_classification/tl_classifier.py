import tensorflow as tf
import keras
import numpy as np
import rospy
import h5py
import cv2
from cv_bridge import CvBridge
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from styx_msgs.msg import TrafficLight
from object_detector import ObjectDetector
import os

MODEL_FILE = 'model.h5'
MODEL_SITE_FILE = 'model_site.h5'
DEF_SIZE = (60, 160)
DEF_THRESHOLD = 0.2
MIN_THRESHOLD = 0.0001
FILE_DIR = os.path.dirname(os.path.realpath(__file__))

class TLClassifier(object):
    def __init__(self, is_site):
        self.count = 0
        self.object_detector = ObjectDetector()

        self.bridge = CvBridge()

        #TODO load classifier
        self.is_site = is_site  # False: simulator

        config = tf.ConfigProto()
        # dynamically grow the memory used on the GPU
        config.gpu_options.allow_growth = True
        # to log device placement (on which device the operation ran)
        # config.log_device_placement = True
        # (nothing gets printed in Jupyter, only if you run it standalone)
        sess = tf.Session(config=config)
        # set this TensorFlow session as the default session for Keras
        set_session(sess)

        # check that model Keras version is same as local Keras version
        if is_site:
            # site
            rospy.loginfo("site mode")
            f = h5py.File(FILE_DIR + '/' + MODEL_SITE_FILE, mode='r')
            self.model = load_model(FILE_DIR + '/' + MODEL_SITE_FILE)
        else:
            # simulator
            rospy.loginfo("simulator mode")
            f = h5py.File(FILE_DIR + '/' + MODEL_FILE, mode='r')
            self.model = load_model(FILE_DIR + '/' + MODEL_FILE)

        model_version = f.attrs.get('keras_version')
        keras_version = str(keras.__version__).encode('utf8')

        if model_version != keras_version:
            rospy.loginfo('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

        self.graph = tf.get_default_graph()

    def get_detection(self, image):
        num_detections, boxes, classes, scores = self.object_detector.run_inference_for_single_image(image)
        threshold = DEF_THRESHOLD

        im_height, im_width, color = image.shape
        for i in range(0, num_detections):
          if classes[i] == 10 and scores[i] >= threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            left = int(xmin * im_width)
            right = int(xmax * im_width)
            top = int(ymin * im_height)
            bottom = int(ymax * im_height)
            tl_image = image[top:bottom, left:right, :]
            return tl_image
        return None

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        with self.graph.as_default():
            #cv2.imwrite('/home/futaya/tl_image/tl_' + str(self.count) + '.jpg', image)
            tl_image = self.get_detection(image)
            self.count += 1
            if tl_image is None:
              #cv2.imwrite('/home/futaya/tl_image/tl_' + str(self.count) + '.jpg', image)
              return TrafficLight.UNKNOWN
            else:
              re_tl_image = cv2.resize(tl_image, DEF_SIZE)
              img = cv2.cvtColor(re_tl_image, cv2.COLOR_BGR2HSV)
              image_array = np.asarray(img[:,:,2]).astype(np.uint8)
              tl = self.model.predict(image_array[None, :, :], batch_size=1)
              #rospy.loginfo("predict %s", str(tl[0]))
              index = np.argmax(tl[0])
              #cv2.imwrite('/home/futaya/tl_image/tl_' + str(self.count) + '_' + str(index) + '.jpg', re_tl_image)
              return index

        return TrafficLight.UNKNOWN
