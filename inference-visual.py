#!/usr/bin/env python
from __future__ import print_function
import os
import cv2
import PIL.Image
import numpy as np
import tensorflow as tf
from button_detection import ButtonDetector
from character_recognition import CharacterRecognizer

def button_candidates(boxes, scores, image):
  img_height = image.shape[0]
  img_width = image.shape[1]

  button_scores = []
  button_patches = []
  button_positions = []

  for box, score in zip(boxes, scores):
    if score < 0.5: continue

    y_min = int(box[0] * img_height)
    x_min = int(box[1] * img_width)
    y_max = int(box[2] * img_height)
    x_max = int(box[3] * img_width)

    button_patch = image[y_min: y_max, x_min: x_max]
    button_patch = cv2.resize(button_patch, (180, 180))

    button_scores.append(score)
    button_patches.append(button_patch)
    button_positions.append([x_min, y_min, x_max, y_max])
  return button_patches, button_positions, button_scores


def get_image_name_list(target_path):
    assert os.path.exists(target_path)
    image_name_list = []
    file_set = os.walk(target_path)
    for root, dirs, files in file_set:
      for image_name in files:
        image_name_list.append(image_name.split('.')[0])
    return image_name_list

if __name__ == '__main__':
    data_dir = './test_panels'
    data_list = get_image_name_list(data_dir)
    detector = ButtonDetector()
    recognizer = CharacterRecognizer()
    overall_time = 0
    for data in data_list:
      img_path = os.path.join(data_dir, data+'.jpg')
      img_np = np.asarray(PIL.Image.open(tf.gfile.GFile(img_path)))
      t0 = cv2.getTickCount()

      boxes, scores, _ = detector.predict(img_np, True)
      button_patches, button_positions, _ = button_candidates(boxes, scores, img_np)

      for button_img, button_pos in zip(button_patches, button_positions):
        button_text, button_score, button_draw =recognizer.predict(button_img, draw=True)
        x_min, y_min, x_max, y_max = button_pos
        button_rec = cv2.resize(button_draw, (x_max-x_min, y_max-y_min))
        detector.image_show[y_min+6:y_max-6, x_min+6:x_max-6] = button_rec[6:-6, 6:-6]

      t1 = cv2.getTickCount()
      time = (t1-t0)/cv2.getTickFrequency()
      overall_time += time
      print('Time elapsed: {}'.format(time))
      cv2.imshow('panels', detector.image_show)
      cv2.waitKey(0)
      #result_show = PIL.Image.fromarray(detector.image_show)
      #result_show.show()
      #result_show.save('./demos/'+data+'.jpg')

    average_time = overall_time / len(data_list)
    print('Average_used:{}'.format(average_time))
    detector.clear_session()
    detector.clear_session()

