# OCR-RCNN: An Accurate and Efficient Framework for Elevator Button Recognition

  <p align="center">
    <img src="./src/button_recognition/scripts/ocr_rcnn_lib/demos/demo_10.jpg">


Cascaded OCR-RCNN is designed for autonomous elevator manipulation, the goal of which is to enable the robot to autonomously operate elevators that are previously unvisited. This repository contains the perception part of this project.  We published the initial version in paper  [A Novel OCR-RCNN for Elevator Button Recognition](https://ieeexplore.ieee.org/abstract/document/8594071) and this version improves the accuracy by 20% and achieves a real-time running speed ~10FPS (640*480)  on a graphical card (>=GTX950).  We have also tested on a laptop installed with a GTX950M (2G memory). It can achieves a running speed of ~6FPS. We are working on optimizing the TX2 version to make it faster,  which will be soon released with the dataset, as well as the post-processing code. 

### Requirements

1.  Ubuntu == 16.04
2.  TensorFlow == 1.12.0
3.  Python == 2.7
4.  Tensorrt == 4.0 (optional)
5.  2GB GPU (or shared) memory 

### Inference

Note: the inference code is moved to **perception** branch and the master branch is used for holding the whole system code, so please check out perception branch for inference!

Before running the code, please first download the [model](https://drive.google.com/file/d/1FVXI-G-EsCrkKbknhHL-9Y1pBshY7JCv/view?usp=sharing) file into the code folder and unzip it. There are five frozen tensorflow models:

1. *detection_graph.pb*: a general detection model that can handle panel images with arbitrary size.
2.  *ocr_graph.pb*: a character recognition model that can handle button images with a size of 180x180.
3. *detection_graph_640x480.pb*: a detection model  fixed-size image as input.
4. *detection_graph_640x480_optimized.pb*: an optimized version of the detection model.
5. *ocr_graph_optimized.pb*:  an optimized version of the recognition model.

For running on laptops and desktops (x86_64), you may need to install some packages :

1. `sudo apt install libjpeg-dev libpng12-dev libfreetype6-dev libxml2-dev libxslt1-dev `
2. `sudo apt install ttf-mscorefonts-installer`
3. `pip install pillow matplotlib lxml imageio --user` 
4. `git clone https://github.com/zhudelong/ocr-rcnn-v2.git`
5. `cd ocr-rcnn-v2`
6. ``mv frozen/ ocr-rcnn-v2/``
7. `python inference.py`  (slow version with two models loaded separately) 
8. ``python inference_640x480.py`` (fast version with two models merged)
9. `python ocr-rcnn-v2-visual.py` (for visualization)

For Nvidia TX-2 platform:

1. Flash your system with [JetPack 4.2](<https://developer.nvidia.com/embedded/jetpack>).

2. We have to install tensorflow-1.12.0 by compiling source code, but if you want to try our [wheel](https://drive.google.com/file/d/1HVXrPZO29loYVdoaPOZDRlB-lB92OuKC/view?usp=sharing), just ignore the following procedure. 

   1. Start TX2 power mode.

      ```bash
      sudo nvpmodel -m 0
      ```

   2. Install some dependencies

      ```bash
      sudo apt-get install openjdk-8-jdk
      sudo apt-get install libhdf5-dev libblas-dev gfortran
      sudo apt-get install libfreetype6-dev libpng-dev pkg-config 
      pip install six mock h5py enum34 scipy numpy --user
      pip install keras --user
      ```

   3. Download Bazel building tool

      ```bash
      cd ~/Downloads
      wget https://github.com/bazelbuild/bazel/releases/download/0.15.2/bazel-0.15.2-dist.zip
      mkdir -p ~/src
      cd ~/src
      unzip ~/Downloads/bazel-0.15.2-dist.zip -d bazel-0.15.2-dist
      cd bazel-0.15.2-dist
      ./compile.sh
      sudo cp output/bazel /usr/local/bin
      bazel help
      ```

   4. Download tensorflow source code and check out r1.12

      ```bash
      cd ~/src
      git clone https://github.com/tensorflow/tensorflow.git
      git checkout r1.12
      ```

   5. Before compiling please apply this [patch](<https://github.com/peterlee0127/tensorflow-nvJetson/blob/master/patch/tensorflow1.12.patch>). 

   6. Configure tensorflow-1.12,  please refer to [item-9](https://jkjung-avt.github.io/build-tensorflow-1.8.0/) and the official [doc](<https://www.tensorflow.org/install/source>).

      ```bash
      cd ~/src/tensorflow
      ./configure
      ```

   7. Start compiling tensorflow-1.12

      ```bash
      bazel build --config=opt --config=cuda --local_resources 4096,2.0,1.0 //tensorflow/tools/pip_package:build_pip_package --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"
      ```

   8. Make the pip wheel file, which will be put in wheel folder.

      ```bash
      ./bazel-bin/tensorflow/tools/pip_package/build_pip_package wheel/tensorflow_pkg
      ```

   9. Install tensorflow with pip.

      ```
      cd wheel/tensorflow_pkg
      pip tensorflow-1.12.1-cp27-cp27mu-linux_aarch64.whl --user
      ```

3. Run the python code in TX2 platform.

   1. ``python inference_tx2.py`` (~0.7s per image, without optimization)
   2. The model can be converted to tensorrt engine for faster inference. If you are interested in converting the model by yourself, please check [here](https://jkjung-avt.github.io/tf-trt-models/)

If you find this work is helpful to your project, please consider cite our paper:

```
@inproceedings{zhu2018novel,
  title={A Novel OCR-RCNN for Elevator Button Recognition},
  author={Zhu, Delong and Li, Tingguang and Ho, Danny and Zhou, Tong and Meng, Max QH},
  booktitle={2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={3626--3631},
  year={2018},
  organization={IEEE}
}
```

### Demonstrations

Two demo-images are listed as follows. They are screenshots from two Youtube videos. The character recognition results are visualized at the center of each bounding box. 

  <p align="center">
    <img src="./src/button_recognition/scripts/ocr_rcnn_lib/demos/image3.jpg" >
    Image Source: [https://www.youtube.com/watch?v=bQpEYpg1kLg&t=8s]
  </p>
  <p align="center">
    <img src="./src/button_recognition/scripts/ocr_rcnn_lib/demos/image2.jpg">
    Image Source: [https://www.youtube.com/watch?v=k1bTibYQjTo&t=9s]
  </p>


