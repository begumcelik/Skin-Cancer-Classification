# Skin Cancer Classification

The purpose of the project is to classify five different types of skin cancer given the skin lesion dataset. The class labels are Melanoma (MEL), Melanocytic nevus (NV), Basal cell carcinoma (BCC), Actinic keratosis (AK), and Benign keratosis (BKL).
Firstly, the data file which contains images within the image IDs was downloaded and right after unzipped in order to extract photos using PyDrive without loss. Photos were loaded into an image array with respect to their IDs given in the train.csv file. After the realization of the difference in image sizes, the target input shape was determined and all images were resized before they were added to the image array. Normalization was also performed in this step. Related to the available RAM limitations of the notebooks, there was a trade-off between image shape and image resolution. In that case, the visualization of the dataset played an important role in observing the image quality related to input shape and images were used as high quality as possible within the given RAM usage. In addition to those, the distribution of samples per class was visualized in order to increase knowledge of the given dataset. As a result, an imbalance in the dataset was realized due to the big amount of data corresponding to class 2 which means melanocytic nevus cancer type. In training data, instance counts per class are shown below:

![alt text](https://github.com/begumcelik/Skin-Cancer-Classification/blob/main/Screen%20Shot%202021-05-04%20at%2015.29.33.png)


         2	: 4489
         1	: 2204
         3	: 1592
         5	: 1288
         4        : 427




Secondly, the need for image augmentation has emerged after the realization of an imbalanced dataset. To reach the aim of increasing the number of images that do not belong to the second class, rotation, flip from left to right, and upside down and random noise functions of skimage library were used. Therefore, images had reiterated with its variations in order to balance the dataset. The number of images per class in the augmented dataset is given below:

               Augmented dataset shape({1: 11020, 3: 7960, 5: 6440, 2: 4489, 4: 2135})
![alt text](https://github.com/begumcelik/Skin-Cancer-Classification/blob/main/Screen%20Shot%202021-05-04%20at%2015.29.48.png)

After that this new data set was split into two as training data and validation data. Training data contains 80% of the dataset and validation is 20%.
The process is continued with searching for image processors. As a library, Tensorflow was initially preferred. Due to the big amount of errors, Keras library is replaced with the Tensorflow. 
In the CNN model, there were 3 2D Convolution layers and 2 dense layers. The input shape was set to (28,28,3) for the first layer. The activation function Relu was preferred for these layers except for the last layer, which has the activation function of Softmax for classification. For the first layer, the kernel matrix with width 3, height 3 and 32 for the dimensionality of the output space was defined. After the second 2D Convolution Layer the process was continued with the max-pooling layer, which returns the pixel with maximum value from a kernel. Max operation for 2D spatial data aims to downsample the input representation. The pool size is fixed to (2,2) and is shifted by strides for each dimension. The MaxPooling2D layer takes the maximum value over this pool in each iteration. In the model, two dropouts with values 0.25 and 0.5 are used. The dropout regularization gives the model the opportunity to learn independent representations. Therefore, the performance of the model increases. Additionally, Flatten function, which prepares a vector for the fully connected layers, was used to flatten the input. The last part of the model consists of two dense layers to obtain the output.



After the model is implemented, it is tried with both original data and the augmented data. The accuracy with the original data is higher than the augmented dataset. It may be caused by the overuse of the image samples. 
 
# Challenges

One of the difficulties that are dealt with is RAM insufficiency. During the trial of k fold the size on n taken as 10. K-Fold was tried in Google Colaboratory with GPU and also TPU. However, the result of this cross-validation stays unfinished due to high memory requiring calculations. Nonetheless, it is expected to increase the validation and test accuracy by using cross-validation.

Another challenge in the process is using the images with their actual sizes. As it is observed, most of the images have dimensions bigger than 500x500. In order to have a more comprehensive machine learning model, which has increased performance, the sizes of the images are tried to preserve as much as possible. However, due to the memory issues, the trial of using the actual image sizes ended with a failure.

# Conclusion

Finally, the model has given 64% test accuracy in validation data and %61.8 accuracy on test.csv which is higher than many of the medical based tests. To reach the aim of increasing the accuracy, cross-validation can be implemented but it requires a higher amount of RAM. Moreover, there is no doubt that with more training data, the model would be more accurate.

# Contributors
Begüm Çelik	

Ekin Oskay

Ece Alptekin

Güren İçim
