Rasmus Barfod Prætorius
Henning Wiberg
Lukas Bjornvad
# Facial recognition project

# Description
For this project we have developed a facial recognition framework. It's capable of recognizing the people in photos, videos and live through a webcam. It utilizes stored faces for the recognition, it's possible for the user to add more through the tracking feature. The project has a set of cli commands for each main function. The project requires a moderate amount of technical knowledge to operate. 

# Technologies

# Installation
Clone the project, cd into the project.
Run "pip install --editable ."
To utilize the training models get matlab at MATLAB

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

