# Kelz-Mask-Class
This repository contains the code for a Convolutional Neural Network (CNN) model trained to detect whether a person is wearing a mask correctly, incorrectly, or not at all. The model is built using TensorFlow/Keras and the data is preprocessed to extract masked faces using YOLO annotations. 

# Convolutional Neural Network for Mask Detection

This repository contains the code for a Convolutional Neural Network (CNN) model trained to detect whether a person is wearing a mask correctly, incorrectly, or not at all. The model is built using TensorFlow/Keras and the data is preprocessed to extract masked faces using YOLO annotations. This project is designed to be easily integrated with a Streamlit application for interactive mask detection.

## Project Overview

The project follows a standard machine learning workflow:

1.  **Data Preprocessing:**
    *   Loading image and YOLO label data from specified directories (`Downloads/ML Test/CNN/Mask/train/images` and `Downloads/ML Test/CNN/Mask/train/labels` for training, and similar for testing).
    *   Cropping the regions of interest (faces) based on the YOLO bounding box annotations. The cropped images are saved to `Downloads/ML Test/CNN/Mask/output` (and `Downloads/ML Test/CNN/Mask/output/test` for the test set).
    *   Organizing the cropped images into directories corresponding to their class labels ("mask_correct", "mask_incorrect", "no_mask").
    *   Applying data augmentation techniques (rescaling, shear range, zoom range, horizontal flip) to the training set using `ImageDataGenerator` to increase the dataset size and improve model generalization.
    *   Rescaling pixel values to the range [0, 1] for both training and testing sets.

2.  **Building the CNN Model:**
    *   Initializing a Sequential Keras model.
    *   Adding multiple 2D Convolutional layers with 32 filters, a 3x3 kernel size, and ReLU activation. These layers extract features from the input images.
    *   Adding MaxPooling layers with a 2x2 pool size and appropriate strides to reduce the spatial dimensions of the feature maps and help the model focus on the most important features.
    *   Flattening the output of the convolutional layers into a 1D vector to prepare it for the fully connected layers.
    *   Adding two Dense (fully connected) layers with ReLU activation (200 and 100 units respectively) to learn complex patterns from the flattened features.
    *   Adding the output layer with 3 units (corresponding to the three classes) and a Softmax activation function to output probability distributions over the classes.

3.  **Training the CNN Model:**
    *   Compiling the model using the Adam optimizer, which is an efficient algorithm for gradient descent.
    *   Using `categorical_crossentropy` as the loss function, suitable for multi-class classification problems with one-hot encoded labels.
    *   Using 'accuracy' as the evaluation metric.
    *   Calculating class weights using `sklearn.utils.class_weight.compute_class_weight` to address the imbalance in the dataset, ensuring that the model doesn't become biased towards the majority class.
    *   Training the model on the preprocessed training set (`training_set`) for 50 epochs, using the calculated `class_weight_dict`.
    *   Evaluating the model on a separate test set (`test_set`) during training to monitor performance on unseen data.
    *   Implementing Early Stopping with a patience of 40 epochs to stop training if the validation accuracy doesn't improve, preventing overfitting.
    *   Implementing ReduceLROnPlateau with a factor of 0.5 and patience of 30 to reduce the learning rate when the validation loss plateaus, allowing for finer tuning of the model weights.

4.  **Evaluation and Prediction:**
    *   Loading the trained Keras model from the saved `.h5` file (`mask_model.h5`).
    *   Making predictions on new, unseen images by preprocessing them (resizing, converting to array, expanding dimensions, and rescaling) and using the loaded model.
    *   Generating a Confusion Matrix and Classification Report to evaluate the model's performance in detail on the test set.
