#AUTHORS: Henrik Tambo Buhl & Alexander Stæhr Johansen 
#CREATED: 01-05-2022

import time
import numpy as np
from homomorphic_eigenfaces_module import EigenfacesServer

class TestSuite():
    server: EigenfacesServer
    labels: np.array([])

    def __init__(self, input_server: EigenfacesServer):
        self.server = input_server
        self.labels = []

    def computation_time_training(self, normalized_training_images: np.array([]), vectorized_training_images: np.array([])):
        print("Testing training time...")
        start = time.time()
        self.server.Train(normalized_training_images, vectorized_training_images)
        duration = (time.time() - start)
        print(f"Training time: {(duration)/60:0.0f} minutes and {(duration)%60:0.0f} seconds.")

    def computation_time_classification(self, test_images: np.array([]), training_image_labels: np.array([])):
        print("Testing classification time...")
        start = time.time()
        self.labels = self.server.Classify(test_images, training_image_labels)
        duration = time.time() - start
        print(f"Classification time: {(duration)/60:0.0f} minutes and {(duration)%60:0.0f} seconds.")

    def prediction_accuracy(self, expected_labels: np.array([])):
        print("Calculating prediction accuracy...")
        correct = 0
        for i in range (0, len(expected_labels)): 
            if self.labels[i] == expected_labels[i]:
                correct += 1
        print("Expected labels: ", expected_labels)
        print("Predicted labels: ", self.labels)
        print(f"Correctly classified: {correct / len(expected_labels) * 100}%")