# Facial recognition project
dat4sem2020spring-python

Agreeable Government:

Rasmus Barfod Prætorius,
Henning Wiberg and
Lukas Bjornvad

# Description
This framework provides cli commands for facial recognition and tracking functions to detect users based on webcam and other media footage such as people in photos and videos. The framework utilizes stored facial images for the recognition, and it is possible for the user to add more through the tracking feature. The project has a set of cli commands for each main function. The project requires a moderate amount of technical knowledge to operate.
 
Within the process of recognition, we have all so trained own Convolutional, Siamese, and using Convolutional, Siamese, and VGG-Face pre-calibrated neural network models as examples. This primarily involves examples on how to train our own neural networks with Keras and TenserFlow deep learning. For the recognition itself, we have made use of dlib’s pre-trained models for face detectors, facial landmark predictors and recognition models for higher recognition accuracy within the actual framework demo.

# Technologies
- OpenCV and image processing
- CLI
- Neural networks, Deep learning
- Machine Learning
- Data processing
- Data wrangling
- Data collection, working with video capture
- Working with CSV and plotting

For specific requirements see:  [requirements.txt](https://github.com/Mutestock/face_recog_project/blob/master/requirements.txt).

# Installation
Clone the project, cd into the project.
Run "pip install --editable ."

Project is compatible with pipenv and can be activated with 'pipenv shell', 'pipenv lock', 'pipenv sync'. Change the python version in the pipfile so that it matches yours.

Project cli commands can be run with an 'frecog' or a 'python main.py' prefix, depending on how the requirements were installed.
cli commands can be found in cli.py and contains examples.

Please be advised, that dlib can be rather sensitive. Especially so on Windows. If errors pop up please check the load order with cmake and imutils. If this doesn't work, you may have to go on a long journey through the VS IDE installer.


To use the the facial classification feature, please download and extract the vgg_face.mat file from [here](https://www.robots.ox.ac.uk/~vgg/data/vgg_face/), and place it in the face_recog_project\face_learning_model\vgg_face_matconvnet folder. You will afterwards be able to train the classifier in the command line using ‘frecog trainer -tr large 2’. This classifier can be used to train a k-nearest neighbor model for specific faces to obtain more accurate classification results.



# Disposition:

The project will be focusing on the development of a facial recognition framework that with a high certainty can detect users based on webcam and other media footage.

Summary:

We will develop our own facial recognition framework that can receive a picture of a person, then through a neural network match the face with known faces to recognize and confirm the person in the picture.

Goals for functionality:
- Detect faces in a picture/frame
- Make out own neural network examples that with somewhat high certainty can recognize faces by matching them with known faces. This part requires a lot of machine learning and training with a large dataset. Try Convolutional, Siamese, and VGG-Face pre-calibrated model types.
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

# How to use.
- Install everything you need by following the installation guide.
- Open up a cmd, bash or terminal based in the root of the project.
- All functions can be found through a "frecog --help" command. Note that all windows that pop up can be closed by pressing the X at the top or pressing "q". To force a stop presse "ctrl C"
- If you want the framework to work with your face run the tracking command by typing "frecog run -t" and then pressing "r". This will prompt a pop-up that you have to fill out with your name.
- The recognition command is run by writing "frecog run -r" followed by either "small" or "large", this will determine the size of the model that will be used. If nothing is added at the end of the command, the default which is "large" will be run with, thus making the process slower.
- The trainer command is used to train the models that does the recogniton.

