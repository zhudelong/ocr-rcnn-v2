#!/usr/bin/env python
import os
import cv2
import rospy
import numpy as np
import PIL.Image as Image
import PIL.ImageOps as ImageOps
from sensor_msgs.msg import CompressedImage
from button_recognition.srv import *

VIDEO_PATH = '../samples/sample-2.MOV'


class ButtonTracker:
  def __init__(self):
    self.detected_box = None
    self.tracker = None

  def init_tracker(self, image, box_list):
    self.tracker = None
    self.tracker = cv2.MultiTracker_create()
    for box_item in box_list:
      self.tracker.add(cv2.TrackerKCF_create(), image, tuple(box_item))

  @staticmethod
  def call_for_service(image):
    rospy.wait_for_service('recognition_service')
    compressed_image = CompressedImage()
    compressed_image.header.stamp = rospy.Time.now()
    compressed_image.data = np.array(cv2.imencode('.jpg', image)[1]).tostring()
    try:
      recognize = rospy.ServiceProxy('recognition_service', recog_server)
      response = recognize(compressed_image)
      if response is None:
        print("None service response!")
      boxes, scores, texts, beliefs = [], [], [], []

      for pred in response.box.data:
        boxes.append([pred.x_min,  pred.y_min, pred.x_max, pred.y_max])
        scores.append(pred.score)
        text = pred.text
        texts.append(text.replace(' ', ''))
        beliefs.append(pred.belief)
      return boxes, scores, texts, beliefs
    except rospy.ServiceException, e:
      print "recognition service failed: {}".format(e)

  @staticmethod
  def visualize_recognitions(frame, box, text):
    # draw bounding boxes
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(frame, p1, p2, (50, 220, 100), thickness=2)
    # draw text at a proper location
    btn_width = (box[2] - box[0]) / 2.0
    btn_height = (box[3] - box[1]) / 2.0
    font_size = min(btn_width, btn_height) * 0.6
    text_len = len(text)
    font_pose = int(0.5*(box[0]+box[2]) - 0.5 * text_len * font_size), int(0.5*(box[1]+box[3]) + 0.5 * font_size)
    # font_pose is the bottom_left of the text
    cv2.putText(frame, text, font_pose, cv2.FONT_HERSHEY_SIMPLEX, 0.6, thickness=2, color=(255, 0, 255))

  @staticmethod
  def resize_to_480x680(img):
    if img.shape != (480, 640):
      img_pil = Image.fromarray(img)
      img_thumbnail = img_pil.thumbnail((640, 480), Image.ANTIALIAS)
      delta_w, delta_h= 640 - img_pil.size[0], 480 - img_pil.size[1]
      padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
      new_im = ImageOps.expand(img_pil, padding)
      img = np.copy(np.asarray(new_im))
    return img


def read_video_and_recognize(video_name):
  if not os.path.exists(video_name):
    raise IOError('Invalid video path or device number!')

  video = cv2.VideoCapture(video_name)
  if not video.isOpened():
    rospy.logwarn('Cannot open the video or device!')
    sys.exit()
  rospy.loginfo('Initialize the tracker ...')

  # initialize tracking process
  button_tracker = ButtonTracker()
  (state, frame) = video.read()
  frame = button_tracker.resize_to_480x680(frame)
  (boxes, scores, texts, beliefs) = button_tracker.call_for_service(frame)
  # button_tracker.init_tracker(frame, tuple(boxes))

  counter = 0
  while state:
    counter += 1
    (state, frame) = video.read()
    if not state: sys.exit()
    frame = button_tracker.resize_to_480x680(frame)

    # update tracker using recognition service every 10 frames
    # todo: separate tracker and recognizer to different threads
    # ok, boxes = button_tracker.tracker.update(frame)
    #if counter % 10 == 0:
    (boxes, scores, texts, beliefs) = button_tracker.call_for_service(frame)
      # button_tracker.init_tracker(frame, boxes)

    # display recognition result
    for box, text in zip(boxes, texts):
      button_tracker.visualize_recognitions(frame, box, text)
    cv2.imshow('button_tracker', frame)
    k = cv2.waitKey(1)
    if k == 27:
      break  # esc pressed


def read_image_and_recognize(image_list):
  button_tracker = ButtonTracker()
  for image_name in image_list:
    if not os.path.exists(image_name):
      raise IOError('Image path {} not exist!'.format(image_name))
    frame = cv2.imread(image_name, cv2.IMREAD_COLOR)
    frame = button_tracker.resize_to_480x680(frame)
    (boxes, scores, texts, beliefs) = button_tracker.call_for_service(frame)

    # display recognition result
    for box, text in zip(boxes, texts):
      button_tracker.visualize_recognitions(frame, box, text)
    cv2.imshow('button_recognition', frame)
    cv2.waitKey(0)


if __name__ == '__main__':
  rospy.init_node('button_tracker', anonymous=True)
  img_only = rospy.get_param('button_tracker/img_only', True)
  if img_only:
    image_path = rospy.get_param('button_tracker/image_path', '../../ocr_rcnn_lib/test_panels')
    image_number = rospy.get_param('button_tracker/image_number', 4)
    img_list = [os.path.join(image_path, '{}.jpg'.format(i)) for i in range(0, image_number)]
    for img_item in img_list:
      if not os.path.exists(img_item):
        raise IOError('Image path not exist: {}'.format(img_item))
    read_image_and_recognize(img_list)
  else:
    video_path = rospy.get_param('button_tracker/video_path', '../test_samples/sample-4.MOV')
    if not os.path.exists(video_path):
      raise IOError('Video path not exist!')
    read_video_and_recognize(video_path)

  rospy.loginfo('Process Finished!')
