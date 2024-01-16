# Custom Object Detection with TensorFlow
This repository describes how to detect, label, and localize objects in videos using TensorFlow's Object Detection and OpenCV. For my particular application,the first part of the tutorial shows how to use a pre-trained model, and the second part shows how to train your own model to detect whatever object(s) you would like.

![](readme_gifs/output_frisbee_catch_faster_rcnn.gif)

## 1. Installation/Setup
**1.1.** Install the other necessary packages by running the following commands in the command prompt:
```
pip install jupyter
pip install pandas
pip install opencv-python
```

**1.7.** Simply run run_me.py,the command will automatically download the pretrained model.
```
sudarshsnan@Sudarshan-MacBook-Air Custom_Object_Detection % /usr/local/bin/python3 /Users/sudarshsnan/De
sktop/Projects/Custom_Object_Detection/run_me.py
Loading modelmask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8....
modelmask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8is loaded successfully!
tf.Tensor([0 1 2 3 4 5 6 7 8], shape=(9,), dtype=int32)
2024-01-16 16:11:01.980 Python[48542:2334426] WARNING: Secure coding is not enabled for restorable state! Enable secure coding by implementing NSApplicationDelegate.applicationSupportsSecureRestorableState: and returning YES.
```

## 2. Using a pre-trained model
**2.1.** Clone this repository and extract the files to `C:\tensorflow\models\research\object_detection` directory.

**2.2.** This repo uses the `faster_rcnn_inception_v2_coco` model. Download the model [here]([http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz]). You can also choose to use a different model from TensorFlow's [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for your object detection application based on whether you prefer speed or accuracy. I prefer accuracy for my model, that is why I chose the faster_rcnn model. 


**3.7. Use your custom object detector**

To use your custom-trained object detector, open `video_object_detection.py` and change the `MODEL_NAME` variable from `faster_rcnn_inception_v2_coco_2018_01_28` to `inference_graph`. Change the `PATH_TO_LABELS` variable from `os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')` to `os.path.join(CWD_PATH,'training','custom_label_map.pbtxt')`. Change the `NUM_CLASSES` variable from 90 to the number of object classes you are trying to detect. 

Its difficult to tell, but my resulting custom fine-tuned model produces slightly better detection accuracy of a frisbee than the pre-trained model. 

![](readme_gifs/custom_model_output.gif)
