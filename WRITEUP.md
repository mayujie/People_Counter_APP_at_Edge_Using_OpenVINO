# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## 1.Explaining Custom Layers

The process behind converting custom layers involves...
Adding extensions to both the Model Optimizer and the Inference Engine.
- Model Optimizer searches for each layer of the input model in the list of known layers before building the model's internal representation, optimizing the model, and producing the Intermediate Representation.
- To actually add custom layers, there are a few differences depending on the original model framework. In both TensorFlow and Caffe, the first option is to register the custom layers as extensions to the Model Optimizer.
- For Caffe, the second option is to register the layers as Custom, then use Caffe to calculate the output shape of the layer. You’ll need Caffe on your system to do this option.
- For TensorFlow, its second option is to actually replace the unsupported subgraph with a different subgraph. The final TensorFlow option is to actually offload the computation of the subgraph back to TensorFlow during inference.

**Detail:**
1. Using Model Optimizer to Generate IR Files Containing the Custom Layer
    - Edit the Extractor Extension Template File
    - Edit the Operation Extension Template File
    - Generate the Model IR Files

2. Inference Engine Custom Layer Implementation for the Intel® CPU
    - Edit the CPU extension template files.
    - Compile the CPU extension library.
    - Execute the Model with the custom layer.
3. Generate the Extension Template Files Using the Model Extension Generator

Some of the potential reasons for handling custom layers are...
- Due to OpenVINO support various frameworks(TensorFlow, Caffe, MXNet, ONNX, etc.), so toolkit has a list of all supported layers from these framework. In case of model uses layer that not exists in list, it will be automatically classified as a custom layer by the Model Optimizer. 
- Custom layers are layers that are not included into a list of known layers. If your topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom. [Here from Custom Layers documentation](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer.html).
- The [list of supported layers](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html) from earlier very directly relates to whether a given layer is a custom layer. Any layer not in that list is automatically classified as a custom layer by the Model Optimizer.
## 2.Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The size of the model pre- and post-conversion was...
Go to each model directory and use command below to check size
```
ls -lh *
```
|Model(Framework)|Size (MB) Before conversion (TF/.pd)|Size (MB) After conversion (OpenVINO/.xml + .bin)|
| ------ | ------ | ------ |
|ssd_mobilenet_v2_coco|67|65.11 (65 + 0.11)|
|ssd_inception_v2_coco|98|96.146 (96 + 0.146)|
|ssdlite_mobilenet_v2_coco|19|18.124 (18 + 0.124)|
|person-detection-retail-0013 (FP32)|/|2.953 (2.8 + 0.153)|

The inference time of the model pre- and post-conversion was...

|Model(Framework)|Inference time OpenVINO (ms)|
| ------ | ------ |
|ssd_mobilenet_v2_coco|69|
|ssd_inception_v2_coco|156|
|ssdlite_mobilenet_v2_coco|32|
|person-detection-retail-0013 (FP32)|45|

The difference of model total count
|Model(OpenVINO)|Total count|
| ------ | ------ | 
|ssd_mobilenet_v2_coco|26|
|ssd_inception_v2_coco|28|
|ssdlite_mobilenet_v2_coco|23|
|person-detection-retail-0013 (FP32)|9|

## 3.Assess Model Use Cases

Some of the potential use cases of the people counter app are...
Each of these use cases would be useful because...

This application could use to help monitor the number of customer in specific scenario such as in store or shop. 
It could help owner to know which time and season customers would like to visit the store and how long they will stay, also could help monitor the restriction of covid-19 in store or shop, which is check if number of people in store is above the limited number. 

## 4.Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

- Lighting: lighting condition is the one of most impact factor in Computer Vision. Since the concept of video detection is actually process by each frame. The lighting condition is changing when object move, it's hard to keep good lighting. Bad lighting cause the model cannot properly detect object. Such as the light is different when person in the center of screen and in the coner. In such case, good preprocess of video is concerned.
- Model accuracy: The model accuracy must be highly accurate. With low accuracy, it's hard to ensure detect object in each frame well. In this project, adjust the threshold of accuracy is one of important step. For different model the accuracy is different, so good model accuracy will be easier to set a good threshold which ensure great performance.
- Camera focal length: Camera focal length depends on requirements of the user. The high focal length is required if user wants monitor the scene with wide angle. The low camera focal length is concerned, when user wants to monitor the narrow scene, like corner. The segmentation of object will be affect by camera focal length. In some scenario, the object could be small for high camera focal length, vice versa.
- Image size: Image size depends on how much quality of output of image that user wants. For image with high resolution the image size will be larger. Of course, the model will give output with high resolution. However, the high resolution input also causes the model takes more time and memory to process. So it's better to achieve a good balance of cost and performance.

## 5.Model Research
[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

The three models i tried were selected from [TensorFlow Object Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). In the end, i've tried one model which provided from @Intel.

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [ssd_mobilenet_v2_coco]
  - [http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz]
  - I converted the model to an Intermediate Representation with the following arguments...
    ```
    cd /home/workspace/models
    wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
    tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
    cd ssd_mobilenet_v2_coco_2018_03_29
    ```
    ```
    python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
    ```
  - The model was insufficient for the app because...it somestimes lost object when detect 2nd person, and result of total count is not acceptable. The detection was unstable during person leave the screen.
  - I tried to improve the model for the app by...adjusting the -pt threshold value, then fianlly find with 0.3. But still not perform good result.
    ```
    python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.3 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
    ```

- Model 2: [ssd_inception_v2_coco]
  - [http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz]
  - I converted the model to an Intermediate Representation with the following arguments...
    ```
    cd /home/workspace/models
    wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
    tar -xvf ssd_inception_v2_coco_2018_01_28.tar.gz
    cd ssd_inception_v2_coco_2018_01_28
    ```
    ```
    python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
    ```
  - The model was insufficient for the app because... it takes a lot of inference time, and monitor seems with high latency. The detection was unstable during person leave the screen.
  - I tried to improve the model for the app by...adjusting the -pt threshold to 0.2, then found minimum total count but still not acceptable result. The high latency problem still not solved.
    ```
    python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.2 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
    ```

- Model 3: [ssdlite_mobilenet_v2_coco]
  - [http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz]
  - I converted the model to an Intermediate Representation with the following arguments...
    ```
    cd /home/workspace/models
    wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
    tar -xvf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
    cd ssdlite_mobilenet_v2_coco_2018_05_09
    ```
    ```
    python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
    ```
  - The model was insufficient for the app because...it's not stable when person leave the screen, the accuracy was not good.
  - I tried to improve the model for the app by...adjusting the -pt threshold to 0.2 and found the minimum total count in three model which i selected. The inference time is the most fast out of all model i've tried. But the result still not good.
    ```
    python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.2 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
    ```

- Final Model from OpenVINO: [[person-detection-retail-0013](https://docs.openvinotoolkit.org/2018_R5/_docs_Retail_object_detection_pedestrian_rmnet_ssd_0013_caffe_desc_person_detection_retail_0013.html)]

Finally, i used the pretrained model person-detection-retail-0013 (FP32) which @Intel provided. The performance was much better than the three model i have tried before. With -pt threshold 0.4, i found the best accuracy which total count is 9 (Very close to actual number of person shown in whole video which is 6). 

    use downloader.py
    ```
    cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader
    sudo ./downloader.py --name person-detection-retail-0013 -o /home/workspace
    ```
    ```
    python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.4 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
    ```
