Rasmus Barfod Prætorius
Henning Wiberg
Lukas Bjornvad
# Facial recognition project

# Description
This framework provides cli commands for facial recognition and tracking functions to detect users based on webcam and other media footage such as people in photos and videos. The framework utilizes stored facial images for the recognition, and it is possible for the user to add more through the tracking feature. The project has a set of cli commands for each main function. The project requires a moderate amount of technical knowledge to operate.
 
Within the process of recognition, we have all so trained own Convolutional, Siamese, and re-trained model neural network models as examples.  This primarily involves examples on how to train our own neural networks with Keras and TenserFlow deep learning. For the recognition itself, we have made use of dlib’s pre-trained models for face detectors, facial landmark predictors and recognition models for higher recognition accuracy within the actual framework demo.

# Technologies

# Installation
Clone the project, cd into the project.
Run "pip install --editable ."

To use the the facial classification feature, please download and extract the vgg_face.mat file from [here](https://www.robots.ox.ac.uk/~vgg/data/vgg_face/), and place it in the face_recog_project\face_learning_model\vgg_face_matconvnet folder. You will afterwards be able to train the classifier in the command line using ‘frecog trainer -tr large 2’. This classifier can be used to train a k-nearest neighbor model for specific faces to obtain more accurate classification results.

# Disposition:

The project will be focusing on the development of a facial recognition framework that with a high certainty can detect users based on webcam and other media footage.

Summary:

We will develop our own facial recognition framework that can receive a picture of a person, then through a neural network match the face with known faces to recognize and confirm the person in the picture.

Goals for functionality:
- Detect faces in a picture/frame
- Make out own neural network examples that with somewhat high certainty can recognize faces by matching them with known faces. This part requires a lot of machine learning and training with a large dataset. Try Convolutional, Siamese, and re-trained model types.
- Use pre-trained models by dlib to display a more accurate use of facial recognition within the framework itself. Here pre-trained models for face detectors, facial landmark predictors and recognition models will be used.
- Use facial recognition to recognize faces within webcam footage as well as videos and pictures from local directories.
- Use a benchmark to test the recognition values and probabilities.
	- If the face in question is recognized, save the match accuracy to a csv file, and return the correct name of the verified person's face.
	- If the face is not recognized, run the function again and save the failed match accuracy to a csv file.
	- Visualize/plot the verification data from the csv.
- Make a feature that can detect a face and then save samples if that face to known faces.
- Create a facial classifier, that can be trained on top of the dlib models and provide enhanced probability readings on specific classified faces.

Concepts and Focus Areas:

The concepts involved in this project regarding the python course and related technologies include the following entries:
- OpenCV and image processing
- CLI
- Neural networks, Deep learning
- Machine Learning
- Data processing
- Data wrangling
- Data collection, working with video capture
- Working with CSV and plotting

