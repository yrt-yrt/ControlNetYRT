# ControlNetYRT
The main content of this code repository:
---
The preprocessing network of the ControlNet model has been modified to respond to numerical inputs instead of the previous conditional images (such as Canny), controlling the image generation process. After completing model training, the network parameters are frozen, and results images A are generated using randomly initialized numerical input data. These are compared with target image B to calculate the error. Backpropagation and the SGD optimizer are applied to perform gradient descent optimization on the input data, inferring the numerical data that can generate target image B, thus obtaining the state change values from image A to image B. Additionally, the image generation process has been optimized, speeding up the backpropagation update of input values and improving overall work efficiency.<br>
<br>
Next are the steps for training the model and implementing backpropagation for the input.<br>
<br>
## Model Training
The specific training process can be referenced in the training procedure of the ControlNet source code below, which is essentially consistent in steps:<br>
https://github.com/lllyasviel/ControlNet/blob/main/docs/train.md\<br>
It’s important to note that, according to the above process, the trained model is not saved. However, the training scripts in this repository include operations for saving the model files, so you don’t need to handle this yourself.
