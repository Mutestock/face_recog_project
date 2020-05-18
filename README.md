# Facial recognition project

Disposition:

The project will be focusing on the development of a facial recognition service/framework that with a high certainty can securely authenticate users based on webcam footage. 

Summary:

We will develop our own facial recognition service/framework that can receive a picture of a person, then through our own neural network match the face with known faces from a database to recognize and authenticate the person in the picture.
- Facial detect faces in a picture
- Make a neural network that with a high certainty can recognize faces by matching them with our database of known faces. This part requires a lot of machine learning and training with a large dataset of known faces.
- (Stretch Goal) Plot validation loss and validation accuracy during training/testing.
- If the face in question is recognized, save the match accuracy to a csv file, and return the correct name of the verified person's face.
- If the face is not recognized, run the function again and save the failed match accuracy to a csv file.
- (Stretch Goal) Visualize/plot the verification data from the csv.
- Make a feature that can detect a face and then persist samples of the face in question to a database.
- (Stretch Goal) If we have time, we could deploy our framework to a droplet running flask so it functions as a cloud service. Here we could also allocate our database.

Concepts and Focus Areas:

The concepts involved in this project in regards to the python course and related technologies include the following entries:
- Data persistence in python
- Logging
- OpenCV
- Working with video capture
- Neural networks, Deep learning
- Machine Learning

Stretch Goals:
- Python web services with flask
- Requests, Headers and Authentication
- Host the neural network training on an external server
- Deployment of framework to a droplet
