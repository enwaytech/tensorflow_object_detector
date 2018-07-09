#!/usr/bin/env python
# # Author: Rohit
# # Date: July, 25, 2017
# Purpose: Ros node to detect objects using tensorflow

# Modified by Thanuja Ambegoda, Enway GmbH - May 13th, 2018
# Added config.yaml to paraeterize model name, model path, label file path...

import os
import sys
import cv2
import numpy as np
try:
    import tensorflow as tf
except ImportError:
    print("unable to import TensorFlow. Is it installed?")
    print("  sudo apt install python-pip")
    print("  sudo pip install tensorflow")
    sys.exit(1)

# ROS related imports
import rospy
import rospkg
from std_msgs.msg import String , Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from enway_msgs.srv import *

# Object detection module imports
import object_detection
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

rospy.init_node('tensorflow_object_detector_node')
debug_image_topic = rospy.get_param('~debug_image_pub_name')
object_pub_topic = rospy.get_param('~objects_detected_pub_name')
model_name = rospy.get_param('~model_name')
gpu_fraction = rospy.get_param('~gpu_fraction')
rospack = rospkg.RosPack()
pkg_path_name = rospy.get_param('~package_name_for_models')
pkg_path = rospack.get_path(pkg_path_name);
model_root = rospy.get_param('~model_root')
model_path = os.path.join(model_root, 'models', model_name)
frozen_graph_file_name = rospy.get_param('~frozen_graph_file_name')
path_to_ckpt = os.path.join(model_path, frozen_graph_file_name)
label_map_file = rospy.get_param('~label_map_file_name')
path_to_label_maps = os.path.join(model_root, 'labels', label_map_file)
num_classes = rospy.get_param('~num_classes')
patch_size = rospy.get_param('~patch_size')
patch_stride = rospy.get_param('~patch_stride')
min_score = rospy.get_param('~min_score')

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# # Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`,
# we know that this corresponds to `airplane`.  Here we use internal utility functions,
# but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(path_to_label_maps)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Setting the GPU options to use fraction of gpu that has been set
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction

# Detection
with detection_graph.as_default():
  with tf.Session(graph=detection_graph, config=config) as sess:
    class detector:

      def __init__(self):
        self.image_pub = rospy.Publisher(debug_image_topic, Image, queue_size=1)
        self.object_pub = rospy.Publisher(object_pub_topic, Detection2DArray, queue_size=1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("image", Image, self.image_cb, queue_size=1, buff_size=2 ** 24)
        self.serviceServer = rospy.Service('SetObjectDetectionMode', SetObjectDetectionMode, self.object_detection_service_cb)
        self.object_detection_activated = False

      def generate_im_patch(self, img_in, patch_size, patch_stride):
        self.rows = img_in.shape[0]
        self.cols = img_in.shape[1]
        print("img_in rows = %d, cols = %d", (self.rows, self.cols))
        # get the bottom half of the image
        img = img_in[(self.rows / 2 + 1) : self.rows, :]
        img_cv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        save_name = '/home/thanuja/test/outputs/in.png'
        print("output saved!**************")
        cv2.imwrite(save_name, img_cv)
        print("sub image img.shape = ")
        print(img.shape)
        patch_id = 0
        for i in range(0, img.shape[0] - patch_size, patch_stride):
          for j in range(0, img.shape[1] - patch_size, patch_stride):
            patch_id = patch_id + 1
            im_patch = img[i : i + patch_size, j : j + patch_size, ...]
            yield im_patch, i, j

      def results_aggregator(self, boxes, scores, classes, num_detections, \
                    boxes_all_t, scores_all_t, classes_all_t, num_detections_all, \
                    y0, x0):
        # box_coords = ymin, xmin, ymax, xmax
        num_predictions = len(scores)
        for i in range(0, num_predictions):
          ymin = (boxes[i][0] * patch_size + y0 + self.rows / 2) / self.rows
          ymax = (boxes[i][2] * patch_size + y0 + self.rows / 2) / self.rows
          xmin = (boxes[i][1] * patch_size + x0) / self.cols
          xmax = (boxes[i][3] * patch_size + x0) / self.cols 
          print(ymin, ymax, xmin, xmax)
          box_adjusted = np.array([ymin, ymax, xmin, xmax]).astype(np.float32)
          print("box_adjusted.shape=")
          print(box_adjusted.shape)
          print("before concat boxes_all_t.shape=")
          print(boxes_all_t.shape)
          boxes_all_t = tf.concat([boxes_all_t, box_adjusted],0)
          print("after concat boxes_all_t.shape=")
          print(boxes_all_t.shape)
          
        scores_all_t = tf.concat([scores_all_t, scores],1)
        print("scores.shape=")
        print(scores.shape)
        print("scores_all_t.shape=")
        print(scores_all_t.shape)
        classes_all_t = tf.concat([classes_all_t, classes],1)
        print("classes.shape")
        print(classes.shape)
        print("classes_all_t.shape")
        print(classes_all_t.shape)
        print("num_detections=")
        print(num_detections)
        num_detections_all += num_detections

        return boxes_all_t, scores_all_t, classes_all_t, num_detections_all
      
      def get_objects_in_image(self, image_in):
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        object_array = []
        coord_array = []
        image_array = []
        image_np_array = []
        patch_id = 0
        # Split image and run the following code for each image in parallel
        for image, y0, x0 in self.generate_im_patch(image_in, patch_size, patch_stride):
          patch_id = patch_id + 1
          # the array based representation of the image will be used later in order to prepare the
          # result image with boxes and labels on it.
          # debug start
          save_name = '/home/thanuja/test/inputs/patch_%d.png' % (patch_id)
          print("patch saved!************** i=%d" % (patch_id))
          cv2.imwrite(save_name, image)
          # debug stop
          image_np = np.asarray(image)
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)              
          image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
          # Each box represents a part of the image where a particular object was detected.
          boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
          # Each score represent how level of confidence for each of the objects.
          # Score is shown on the result image, together with the class label.
          scores = detection_graph.get_tensor_by_name('detection_scores:0')
          classes = detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = detection_graph.get_tensor_by_name('num_detections:0')
          (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], \
              feed_dict={image_tensor: image_np_expanded})
          
          '''
          # put together prediction results into one image representation
          (boxes_all_t, scores_all_t, classes_all_t, num_detections_all) = self.results_aggregator(\
                                boxes, scores, classes, num_detections, \
                                boxes_all_t, scores_all_t, classes_all_t, num_detections_all, \
                                y0, x0)
          '''

          objects = vis_util.visualize_boxes_and_labels_on_image_array(
              image,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=2)
          
          object_array.append(objects)
          coord_array.append([x0, y0])
          image_array.append(image)
          image_np_array.append(image_np)
        return object_array, coord_array, image_array, image_np_array

      def image_cb(self, data):
        if(self.object_detection_activated):
          objArray = Detection2DArray()
          try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
          except CvBridgeError as e:
            print(e)
          image_in = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
          image_in_np = np.asarray(image_in)
          object_array, coords_array, image_array, image_np_array = self.get_objects_in_image(image_in)
          
          objArray.detections = []
          objArray.header = data.header
          object_count = 1
          
          num_patches = len(object_array)
          for patch_id in range(0,num_patches):
            objects = object_array[patch_id]
            [x0, y0] = coords_array[patch_id]
            image = image_array[patch_id]
            image_np = image_np_array[patch_id]
        
            for i in range(len(objects)):
              object_count += 1
              objArray.detections.append(self.object_predict_patch(objects[i], data.header, image_np, image, x0, y0))

          self.object_pub.publish(objArray)
          
          image_np_out = self.aggregate_patches(image_in_np, image_np_array, coords_array)
          image_np_out=image_np_out.astype(np.uint8)
          img = cv2.cvtColor(image_np_out, cv2.COLOR_BGR2RGB)
          save_name = '/home/thanuja/test/outputs/out.png'
          print("output saved!**************")
          cv2.imwrite(save_name, img)
          image_out = Image()
          try:
            image_out = self.bridge.cv2_to_imgmsg(img, "bgr8")
          except CvBridgeError as e:
            print(e)
          image_out.header = data.header
          self.image_pub.publish(image_out)

      def object_predict(self, object_data, header, image_np, image):
        image_height, image_width, channels = image.shape
        obj = Detection2D()
        obj_hypothesis = ObjectHypothesisWithPose()

        object_id = object_data[0]
        object_score = object_data[1]
        dimensions = object_data[2]

        obj.header = header
        obj_hypothesis.id = object_id
        obj_hypothesis.score = object_score
        obj.results.append(obj_hypothesis)
        obj.bbox.size_y = int((dimensions[2] - dimensions[0]) * image_height)
        obj.bbox.size_x = int((dimensions[3] - dimensions[1]) * image_width)
        obj.bbox.center.x = int((dimensions[1] + dimensions [3]) * image_width / 2)
        obj.bbox.center.y = int((dimensions[0] + dimensions[2]) * image_height / 2)

        return obj
      
      def object_predict_patch(self, object_data, header, image_np, image, x0, y0):
        image_height, image_width, channels = image.shape
        obj = Detection2D()
        obj_hypothesis = ObjectHypothesisWithPose()

        object_id = object_data[0]
        object_score = object_data[1]
        dimensions = object_data[2]

        obj.header = header
        obj_hypothesis.id = object_id
        obj_hypothesis.score = object_score
        obj.results.append(obj_hypothesis)
        obj.bbox.size_y = int((dimensions[2] - dimensions[0]) * image_height)
        obj.bbox.size_x = int((dimensions[3] - dimensions[1]) * image_width)
        obj.bbox.center.x = int((dimensions[1] + dimensions [3]) * image_width / 2) + x0
        obj.bbox.center.y = int((dimensions[0] + dimensions[2]) * image_height / 2) + y0

        return obj

      def object_detection_service_cb(self, req):
        # if (req.mode == enway_msgs.SetObjectDetectionModeRequest.IDLE):
        if (req.mode == 0):
          # stop the object detector if running
          self.object_detection_activated = False

        # elif(req.mode == enway_msgs.SetObjectDetectionModeRequest.ACTIVE):
        elif(req.mode == 1):
          # start the service if idle
          self.object_detection_activated = True

        return True
      
      def aggregate_patches(self, image_np_in, image_np_array, coords_array):

        im_out = image_np_in
        print("im_out.shape")
        print(im_out.shape)
        for i in range(len(image_np_array)):
          image_patch = image_np_array[i]
          [start_col, start_row] = coords_array[i]
          start_row = start_row + self.rows / 2
          stop_col = start_col + image_patch.shape[1]
          stop_row = start_row + image_patch.shape[0]
          print("i, start_col, stop_col, start_row, stop_row")
          print(i, start_col, stop_col, start_row, stop_row)
          im_out[start_row: stop_row, start_col : stop_col, :] = image_patch[:,:,:]
          
        return im_out

def main(args):
  obj = detector()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("ShutDown")
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main(sys.argv)
