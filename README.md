# Image-Colorization_Group8
The goal of this project is to use convolutional neural networks to colorize grayscale images. 

The database used to train the model was MIT places 2: https://www.kaggle.com/datasets/nickj26/places2-mit-dataset 
The code was adapted from: https://lukemelas.github.io/image-colorization.html 
The main code for building the model is the file: Image_Colorization_Group8_Code.py

We wanted to compare how colorization would be affected if we trained 12 images vs 500 images. The processing time for 12 images was about 10 minutes whereas the processing time for 500 images was about 2 hours. We also looked two different regression models and compared their loss. That being the mean squared error and the Hubor loss. 

# Results from Image Colorization 


# Analysis 
(Two different Criterions were used)

When looking at the loss graphs for a mean squared error regression, we see that the overall pattern is that the loss decreases as the epoch increases for both the training and validation dataset. This means that as the model is going through the iterations of the images, it is minimizing the error between the predicted and actual values. However, for the validation curve, the scale of the y-axis is 1e6 meaning that the loss is quite large. 

When the Huber loss was used, the same pattern was observed where the loss decreases for increasing epoch. It is important to note that the scale of the y-axis is much smaller for this regression than it is for the mean squared error. This means that it was able to reduce the error between predicted and actual values much better than the previous regression model.
