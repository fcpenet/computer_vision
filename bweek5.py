from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.models import load_model
import numpy as np

# load an image from file
image = load_img('clothes2.jpg', target_size=(28, 28), grayscale=True)
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2])) # prepare the image for the VGG model
model = load_model('model.h5')
# predict the probability across all output classes
yhat = model.predict_classes(image)
print(yhat)

