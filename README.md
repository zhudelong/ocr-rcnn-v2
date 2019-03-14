# Accurate and Efficient Elevator Button Localization

OCR-RCNN-v2 is designed for autonomous elevator manipulation, the goal of which is to enable the robot to autonomously operate elevators that are previously unvisited. This repository contains the perception part of this project.  We published the initial version in paper  [A Novel OCR-RCNN for Elevator Button Recognition](https://ieeexplore.ieee.org/abstract/document/8594071) and this version improves the accuracy by 20% and achieves a real-time running speed (640*480 in gtx1070ti).  Current version can also run in  notebooks with at least 2GB  GPU memory.  The Nvidia TX-2 compatible version will be soon released with the dataset, as well as the post-processing code. 

### Requirements

1.  Ubuntu == 16.04
2.  TensorFlow == 1.9.0
3. Python == 2.7
4.  2GB GPU (or shared) memory 

### Installation

For notebooks and desktops:

1. `sudo apt install libjpeg-dev libpng12-dev libfreetype6-dev libxml2-dev libxslt1-dev`
2. `pip install pillow, matplotlib, lxml` 
3. `git clone https://github.com/zhudelong/ocr-rcnn-v2.git`
4. `cd ocr-rcnn-v2`
5. `python ocr-rcnn-v2-infer.py` 
6. `python ocr-rcnn-v2-visual.py` (for visualization)

For Nvidia TX-2 platform:

1. soon be available.
2. if you are interested in converting the model by yourself, please check [here](https://jkjung-avt.github.io/tf-trt-models/)

### Demonstrations

Two demo-images are listed as follows. They are screenshots from two Youtube videos. The character recognition results are visualized at the center of each bounding box. 

  <p align="center">
    <img src="./demos/image3.jpg" width=960 height=540>
    Image Source: [https://www.youtube.com/watch?v=bQpEYpg1kLg&t=8s]
  </p>
  <p align="center">
    <img src="./demos/image2.jpg" width=960 height=540>
    Image Source: [https://www.youtube.com/watch?v=k1bTibYQjTo&t=9s]
  </p>