#AUTHORS: Henrik Tambo Buhl & Alexander Stæhr Johansen 
#CREATED: 01-05-2022

from PIL import Image #Used in preprocesser
from eigenfaces_module import EigenfacesClient, EigenfacesServer
import numpy as np
import os
from tests import TestSuite

TRAINING_IMAGES_PATH = "training_images"
TEST_IMAGE_PATH = "test_images"

def load_images(image_root: str) -> np.array([]):
    images = [] #List for images
    image_directories = [] #List for image directories
    image_names = [] #List for image names
    image_labels = [] #List for image labels
    image_name_suffix = '.' #File extension suffix

    #Store all image directories, if it does not start with a '.': 
    image_directories = [image for image in os.listdir(image_root) if not image.startswith(image_name_suffix)]
            
    #Within each directory, store all image names, if it does not start with a '.':
    for index, image_directory in enumerate(image_directories):
        in_directory_image_names = ([image_name for image_name in os.listdir(os.path.join(image_root, image_directory)) if not image_name.startswith(image_name_suffix)])
        image_names.append(in_directory_image_names)

    #Greyscale, rescale to default size, if necessary, and store in image list:
    for index, image_directory in enumerate(image_directories): 
        for image_name in image_names[index]:
            image = Image.open(os.path.join(image_root, image_directory, image_name)) #Open image
            images.append(image) #Append image to list of images
            image_labels.append(image_directory)
    #Return images and image names: 
    return images, image_labels

if __name__ == '__main__':
    # Create an Eigenfaces client:
    Client = EigenfacesClient()
    # Create the Eigenfaces server and inject the client-sided functionality into its initializer:
    Server = EigenfacesServer(Client._n_components_comparison, Client._distance_comparison, 
    Client._goldschmidt_initializer)

    # Load training images and test image from paths: 
    training_images, training_image_labels = load_images(TRAINING_IMAGES_PATH)
    test_images, test_image_labels = load_images(TEST_IMAGE_PATH)

    # Preprocess the training images, using the client:
    normalized_training_images = Client.Image_preprocesser(training_images)

    # Represent the training images as vectors, using the client:
    vectorized_training_images = Client.Image_vector_representation(normalized_training_images)
    
    # Preprocess the test images, using the client:
    normalized_test_images = Client.Image_preprocesser(test_images)

    #Test the module: 
    tests = TestSuite(Server)
    tests.computation_time_training(normalized_training_images, vectorized_training_images)
    tests.computation_time_classification(normalized_test_images, training_image_labels)
    tests.prediction_accuracy(test_image_labels)