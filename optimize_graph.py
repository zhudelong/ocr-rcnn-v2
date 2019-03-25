#!/usr/bin/env python
import os
import sys
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.tools import optimize_for_inference_lib
# from tensorflow.contrib.lite.toco.python

class GraphOptimizer:
  def __init__(self, rcnn_path= None, ocr_path=None, inferlib_opt=False, tflite_opt=False):
    self.ocr_graph_path = ocr_path
    self.rcnn_graph_path = rcnn_path
    self.inferlib_opt = inferlib_opt
    self.tflite_opt = tflite_opt
    self.rcnn_input = ['image_tensor']
    self.rcnn_output = ['detection_boxes', 'detection_scores', 'num_detections']
    self.ocr_input = ['ocr_input']
    self.ocr_output = ['predicted_chars', 'predicted_scores']

  def optimize_rcnn(self, input_graph_def):
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
      input_graph_def,
      self.rcnn_input,
      self.rcnn_output,
      tf.uint8.as_datatype_enum,
      False)
    output_path = self.rcnn_graph_path.replace('.pb','_optimized.pb')
    if os.path.exists(output_path):
      raise IOError('Model already exist: {}'.format(output_path))
    f = gfile.FastGFile(output_path, "w")
    f.write(output_graph_def.SerializeToString())
    print('Finish optimize rcnn graph!')

  def optimize_ocr(self, input_graph_def):
    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
      input_graph_def,
      self.ocr_input,
      self.ocr_output,
      tf.uint8.as_datatype_enum,
      False)
    output_path = self.ocr_graph_path.replace('.pb','_optimized.pb')
    if os.path.exists(output_path):
      raise IOError('Model already exist: {}'.format(output_path))
    f = gfile.FastGFile(output_path, "w")
    f.write(output_graph_def.SerializeToString())
    print('Finish optimize ocr graph!')

  def tflite_rcnn(self):
    converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(
      self.rcnn_graph_path, self.rcnn_input, self.rcnn_output)
    tflite_model = converter.convert()
    model_name = self.rcnn_graph_path.replace('.pb', '.tflite')
    open(model_name, "wb").write(tflite_model)

  def tflite_ocr(self):
    converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(
      self.ocr_graph_path, self.ocr_input, self.ocr_output)
    tflite_model = converter.convert()
    model_name = self.ocr_graph_path.replace('.pb', '.tflite')
    open(model_name, "wb").write(tflite_model)

  def optimize_graph(self):
    # check graph paths
    if self.ocr_graph_path is None:
      self.ocr_graph_path = './frozen_model/ocr_graph.pb'
    if self.rcnn_graph_path is None:
      self.rcnn_graph_path = './frozen_model/detection_graph_640x480.pb'
    assert os.path.exists(self.ocr_graph_path) and os.path.exists(self.rcnn_graph_path)

    # load button detection graph definition
    with tf.gfile.GFile(self.rcnn_graph_path, 'rb') as fid:
      detection_graph_def = tf.GraphDef()
      serialized_graph = fid.read()
      detection_graph_def.ParseFromString(serialized_graph)
      if self.inferlib_opt:
        self.optimize_rcnn(detection_graph_def)
      if self.tflite_opt:
        self.tflite_rcnn()

    # load character recognition graph definition
    with tf.gfile.GFile(self.ocr_graph_path, 'rb') as fid:
      recognition_graph_def = tf.GraphDef()
      serialized_graph = fid.read()
      recognition_graph_def.ParseFromString(serialized_graph)
      if self.inferlib_opt:
        self.optimize_ocr(recognition_graph_def)
      if self.tflite_opt:
        self.tflite_ocr()

if __name__ == '__main__':
  optimizer = GraphOptimizer(inferlib_opt=True, tflite_opt=False)
  optimizer.optimize_graph()