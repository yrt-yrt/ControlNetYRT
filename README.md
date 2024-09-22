# ControlNetYRT
The main content of this code repository:
---
The preprocessing network of the ControlNet model has been modified to respond to numerical inputs instead of the previous conditional images (such as Canny), controlling the image generation process. After completing model training, the network parameters are frozen, and results images A are generated using randomly initialized numerical input data. These are compared with target image B to calculate the error. Backpropagation and the SGD optimizer are applied to perform gradient descent optimization on the input data, inferring the numerical data that can generate target image B, thus obtaining the state change values from image A to image B. Additionally, the image generation process has been optimized, speeding up the backpropagation update of input values and improving overall work efficiency.<br>
<br>
Next are the steps for training the model and implementing backpropagation for the input.<br>
<br>
## Model Training
The specific training process can be referenced in the training procedure of the ControlNet source code below, which is essentially consistent in steps:<br>
https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md<br>
It’s important to note that, according to the above process, the trained model is not saved. However, the training scripts in this repository include operations for saving the model files, so you don’t need to handle this yourself.<br>
<br>
The biggest difference from the previous ControlNet training process lies in the creation of the dataset. Due to changes in the preprocessing network structure of ControlNet in this repository, the original multi-layer CNN structure has been changed to a 3-layer MLP structure, meaning that the input data has shifted from conditional images to numerical data. The specific changes can be referenced in `./dataForTrain/prompt.json`, where the main change is in the 'source' section, which has switched from image file paths to numerical values.<br>
<br>
To verify the feasibility of updating input values through backpropagation in the latter half of the process, the numerical data in the dataset consists of simple one-dimensional data representing rotation angles centered around images. From the perspective of the forward execution of the generated model, the trained model can generate images based on the input angle values and the information from the prompts, altering the images according to the input angles. From the perspective of updating input values through backpropagation, for example, if the initial input angle value is -4 and there is a target image with the same content rotated by 2 degrees, the error between the initially generated image and the target image can be used. By freezing the network model parameters and continuously applying gradient descent backpropagation optimization to the input value, the input value can ultimately converge to a value near 2 degrees.<br>
<br>
## Backpropagation to update input values
The code for backpropagation to update the input is executed as follows:<br>
<br>
`python test5.py`<br>
<br>
You can change the input_image_path and target_image_path in it to your own input image and output image paths, and the control variable represents the input numeric data, and you can randomize the initial value yourself. After the code runs control will gradually update to the rotation angle corresponding to the target image.<br>
Note: The results run on the training set data are basically as expected, for example, training with angle values within the interval -5 to +5, the results obtained within this range are basically as expected. However, as the values move away from this range, it is difficult to converge towards the target value. It is recommended to train with more datasets so that it can be adapted to a larger range of data<br>
<br>

## Accelerated image generation
The following code accelerates image generation during back propagation.<br>
<br>
`python test5_s1.py`<br>
The main principle is to update the input values of the previous step of the image obtained as the next image generation plus noise denoising starting point, before the image generation of noise reduction requires 32 steps, after the change of 10 steps to generate the target image, but also reduces the time of the input values converge to the target data.

Translated with DeepL.com (free version)
