import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#https://www.tensorflow.org/tutorials/keras/classification

print(tf.__version__)
print()

print("Loading the datasets......")
fashion_minist = tf.keras.datasets.fashion_mnist
(train_images, train_labels),(test_images, test_labels) = fashion_minist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(train_images.shape)
print(len(train_labels))
print(train_labels)

print()
print("Start training the model with the datasets...")
train_images = train_images/255.0
test_images  = test_images/255.0

# build the model
print()
print("Build the model")
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10)
])

# Before the model for training, related settings in compiling step
# 1. Loss function —This measures how accurate the model is during training.
# 2. Optimizer —This is how the model is updated based on the data it sees and its loss function.
# 3. Metrics —Used to monitor the training and testing steps.
print()
print("Model settings before the training......")
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Feed the model
print()
print("Feed the model (or train the model)......")
model.fit(train_images, train_labels, epochs=10)

# Evaluate accuracy
print()
print("Evaluate the accuracy......")
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

print()
print("Make predictions......")
# Make predictions
# With the model trained, you can use it to make predictions about some images. The model's linear outputs, logits.
# Attach a softmax layer to convert the logits to probabilities, which are easier to interpret.
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
# The model has predicted the label for each image in the testing set.

print(predictions[0])
# argmax 沿轴axis最大值的索引值, cooresponding to on specific class
print(np.argmax(predictions[0]))
print(class_names[np.argmax(predictions[0])])

# Verify predictions
# pass


# Use the trained model
print()
print("Use the trained model......")
img = test_images[1]
print(img.shape)
img = (np.expand_dims(img,0))
print(img.shape)
predictions_single = probability_model.predict(img)
print(predictions_single) # probabilities
print(np.argmax(predictions_single[0]))
