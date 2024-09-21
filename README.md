# ControlNetYRT
The main content of this code repository:
---
The preprocessing network of the ControlNet model has been modified to respond to numerical inputs instead of the previous conditional images (such as Canny), controlling the image generation process. After completing model training, the network parameters are frozen, and results images A are generated using randomly initialized numerical input data. These are compared with target image B to calculate the error. Backpropagation and the SGD optimizer are applied to perform gradient descent optimization on the input data, inferring the numerical data that can generate target image B, thus obtaining the state change values from image A to image B. Additionally, the image generation process has been optimized, speeding up the backpropagation update of input values and improving overall work efficiency.<br>
<br>
Next are the steps for training the model and implementing backpropagation for the input.<br>
<br>
##Model Training
