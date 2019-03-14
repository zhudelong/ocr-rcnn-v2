#!/usr/bin/env python
import os
import PIL.Image
import imageio
import numpy as np
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util

class ButtonDetector:
  def __init__(self, graph_path=None, label_path=None, verbose=False):
    self.graph_path = graph_path
    self.label_path = label_path
    self.category_index = None

    self.session = None
    self.input = None
    self.output = []
    self.class_num = 1
    self.verbose = verbose
    self.image_show = None

    self.init_detector()
    print('button detector initialized!')

  def init_detector(self):

    # load graph and label map from default folder
    if self.graph_path is None:
      self.graph_path = './frozen_model/detection_graph.pb'
    if self.label_path is None:
      self.label_path = './frozen_model/button_label_map.pbtxt'

    # check existence of the two files
    if not os.path.exists(self.graph_path):
      raise IOError('Invalid detector_graph path! {}'.format(self.graph_path))
    if not os.path.exists(self.label_path):
      raise IOError('Invalid label path! {}'.format(self.label_path))

    # load frozen graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(self.graph_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    self.session = tf.Session(graph=detection_graph)

    # prepare input and output request
    self.input = detection_graph.get_tensor_by_name('image_tensor:0')
    self.output.append(detection_graph.get_tensor_by_name('detection_boxes:0'))
    self.output.append(detection_graph.get_tensor_by_name('detection_scores:0'))
    self.output.append(detection_graph.get_tensor_by_name('detection_classes:0'))
    self.output.append(detection_graph.get_tensor_by_name('num_detections:0'))

    # Load label map
    label_map = label_map_util.load_labelmap(self.label_path)
    categories = label_map_util.convert_label_map_to_categories(
      label_map, max_num_classes=self.class_num, use_display_name=True)
    self.category_index = label_map_util.create_category_index(categories)

  def clear_session(self):
    if self.session is not None:
      self.session.close()

  def predict(self, image_np, draw=False):
    img_in = np.expand_dims(image_np, axis=0)
    boxes, scores, classes, num = self.session.run(self.output, feed_dict={self.input: img_in})
    boxes, scores, classes, num = [np.squeeze(x) for x in [boxes, scores, classes, num]]

    if self.verbose:
      self.visualize_detection_result(image_np, boxes, classes, scores, self.category_index)
    if draw:
      self.image_show = np.copy(image_np)
      self.draw_result(self.image_show, boxes, classes, scores, self.category_index)

    return boxes, scores, num

  @staticmethod
  def visualize_detection_result(image_np, boxes, classes, scores, category):
    vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      np.squeeze(boxes),
      np.squeeze(classes).astype(np.int32),
      np.squeeze(scores),
      category,
      max_boxes_to_draw=100,
      use_normalized_coordinates=True,
      line_thickness=5,
      predict_chars=None
    )
    img_show = PIL.Image.fromarray(image_np)
    img_show.show()


  @staticmethod
  def draw_result(image_np, boxes, classes, scores, category, predict_chars=None):
    vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      np.squeeze(boxes),
      np.squeeze(classes).astype(np.int32),
      np.squeeze(scores),
      category,
      max_boxes_to_draw=100,
      use_normalized_coordinates=True,
      line_thickness=5,
      predict_chars=predict_chars
    )

if __name__ == '__main__':
  detector = ButtonDetector(verbose=True)
  image = imageio.imread('./test_panels/image0.jpg')
  detector.predict(image)
  detector.clear_session()
