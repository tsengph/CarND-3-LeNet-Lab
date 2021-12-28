# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[img_visualize_dataset]: ./test_images_output/visualize_dataset.png
[img_label_frequency]: ./test_images_output/label_frequency.png
[img_rgb_vs_grayscale]: ./test_images_output/rgb_vs_grayscale.png
[img_original_vs_normalized]: ./test_images_output/original_vs_normalized.png
[img_augment_image]: ./test_images_output/augment_image.png
[img_label_frequency_after_filling_agumented_images]: ./test_images_output/label_frequency_after_filling_agumented_images.png
[img_test_model_on_new_images]: ./test_images_output/test_model_on_new_images.png
[img_test_model_on_new_images_top_5_guess]: ./test_images_output/test_model_on_new_images_top_5_guess.png
[img_test_model_on_new_images_softmax_probability]: ./test_images_output/test_model_on_new_images_softmax_probability.png


---
**File Location**

* Notebook: Traffic\_Sign_Classifier.ipynb
* Test images output: test\_images_output/*
* Data Set Training: traffic-signs-data/train.p
* Data Set Validation: traffic-signs-data/valid.p
* Data Set Testing: traffic-signs-data/test.p
* New Images: traffic-signs-data/custom/*

---
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

## Step 0: Load The Data

```
# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = "./traffic-signs-data/train.p"
validation_file= "./traffic-signs-data/valid.p"
testing_file = "./traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

print("X_train(features): %s, y_train(labels): %s" % (X_train.shape, y_train.shape))
print("X_valid(features): %s, y_valid(labels): %s" % (X_valid.shape, y_valid.shape))
print("X_test(features): %s, y_test(labels): %s" % (X_test.shape, y_test.shape))
```
```
X_train(features): (34799, 32, 32, 3), y_train(labels): (34799,)
X_valid(features): (4410, 32, 32, 3), y_valid(labels): (4410,)
X_test(features): (12630, 32, 32, 3), y_test(labels): (12630,)
```

---

## Step 1: Dataset Summary & Exploration

The pickled data is a dictionary with 4 key/value pairs:

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**


### (1) Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas
```
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results
import numpy as np

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```
```
Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
```

### (2) Include an exploratory visualization of the dataset
```
### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import random
# Visualizations will be shown in the notebook.
%matplotlib inline

# show image of 15 random data points
fig, axs = plt.subplots(3,5, figsize=(15, 6))
fig.subplots_adjust(hspace = .2, wspace=.001)
axs = axs.ravel()
for i in range(15):
    index = random.randint(0, len(X_train))
    image = X_train[index]
    axs[i].axis('off')
    axs[i].imshow(image)
    axs[i].set_title(y_train[index])
    print (index, y_train[index], signName[y_train[index]])

fig.savefig('./test_images_output/visualize_dataset.png')
```
```
ID sign_id sign_name
4017 1 Speed limit (30km/h)
912 36 Go straight or right
10304 32 End of all speed and passing limits
29120 12 Priority road
8821 11 Right-of-way at the next intersection
34608 25 Road work
887 31 Wild animals crossing
24119 7 Speed limit (100km/h)
12871 5 Speed limit (80km/h)
12934 5 Speed limit (80km/h)
20319 34 Turn left ahead
923 36 Go straight or right
9993 0 Speed limit (20km/h)
19517 35 Ahead only
6662 19 Dangerous curve to the left
```
![alt text][img_visualize_dataset]

### (3) Show the label frequency of the dataset

```
# histogram of label frequency
hist, bins = np.histogram(y_train, bins=n_classes)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.savefig('./test_images_output/label_frequency.png')
plt.show()
```
![alt text][img_label_frequency]


## Step 2: Design and Test a Model Architecture

### Pre-process the Data Set (normalization, grayscale, etc.)


I use 7 steps to pre-process the data set.

-  1. Convert train and test datasets from RGB to grayscale - This could help to reduce the training time.
-  2. Normalize the data to the range (-1,1) - This helps data to have mean zero and equal variance. 
-  3 - 5. Augment the images and make each sign type has at least 1000 examples - This helps to reduce the bias of the imbalance of the training set and increase accuracy of the model.
-  6. Shuffle the training dataset - This help to reduce the overfitting of the training.
-  7. Split validation dataset off from training dataset - I used the SciKit Learn train_test_split function to create a validation set out of the training set. I used 20% of the testing set to create the validation set.


### (1) Convert train and test datasets from RGB to grayscale

```
### Preprocess the data here.
### Feel free to use as many code cells as needed.

# Convert to grayscale
X_train_rgb = X_train
X_train_gry = np.sum(X_train/3, axis=3, keepdims=True)

X_test_rgb = X_test
X_test_gry = np.sum(X_test/3, axis=3, keepdims=True)

print('RGB shape:', X_train_rgb.shape)
print('Grayscale shape:', X_train_gry.shape)

X_train = X_train_gry
X_test = X_test_gry
```
```
RGB shape: (34799, 32, 32, 3)
Grayscale shape: (34799, 32, 32, 1)
```
![alt text][img_rgb_vs_grayscale]


### (2) Normalize the train and test datasets to (-1,1)

```
## Normalize the train and test datasets to (-1,1)

X_train_normalized = (X_train - 128)/128 
X_test_normalized = (X_test - 128)/128

print(np.mean(X_train_normalized))
print(np.mean(X_test_normalized))

# print(X_train_normalized)
```
![alt text][img_original_vs_normalized]

### (3) Create augment_image function to add a new training image by applying 4 processing on an existing image 
1. random_translate: random translate 2 pixels in x and y directions
2. random_scale: random sacle up to 2 pixels in x and y directions
3. random_warp: random warp the degree on x and y directions
4. random_brightness: random brightness on (0,2) range


```
import cv2
import scipy.ndimage as scnd

def random_translate(img):
    rows,cols,_ = img.shape
    
    # allow translation up to px pixels in x and y directions
    px = 2
    dx,dy = np.random.randint(-px,px,2)

    M = np.float32([[1,0,dx],[0,1,dy]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    
    dst = dst[:,:,np.newaxis]
    
    return dst

def random_scaling(img):   
    rows,cols,_ = img.shape

    # transform limits
    px = np.random.randint(-2,2)

    # ending locations
    pts1 = np.float32([[px,px],[rows-px,px],[px,cols-px],[rows-px,cols-px]])

    # starting locations (4 corners)
    pts2 = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(img,M,(rows,cols))
    
    dst = dst[:,:,np.newaxis]
    
    return dst

def random_warp(img):
    
    rows,cols,_ = img.shape

    # random scaling coefficients
    rndx = np.random.rand(3) - 0.5
    rndx *= cols * 0.06   # this coefficient determines the degree of warping
    rndy = np.random.rand(3) - 0.5
    rndy *= rows * 0.06

    # 3 starting points for transform, 1/4 way from edges
    x1 = cols/4
    x2 = 3*cols/4
    y1 = rows/4
    y2 = 3*rows/4

    pts1 = np.float32([[y1,x1],
                       [y2,x1],
                       [y1,x2]])
    pts2 = np.float32([[y1+rndy[0],x1+rndx[0]],
                       [y2+rndy[1],x1+rndx[1]],
                       [y1+rndy[2],x2+rndx[2]]])

    M = cv2.getAffineTransform(pts1,pts2)

    dst = cv2.warpAffine(img,M,(cols,rows))
    
    dst = dst[:,:,np.newaxis]
    
    return dst

def random_brightness(img):
    shifted = img + 1.0   # shift to (0,2) range
    img_max_value = max(shifted.flatten())
    max_coef = 2.0/img_max_value
    min_coef = max_coef - 0.1
    coef = np.random.uniform(min_coef, max_coef)
    dst = shifted * coef - 1.0
    
    return dst

def augment_image(img):
    img = random_translate(img)
    img = random_scaling(img)
    img = random_warp(img)
    img = random_brightness(img)
    
    return img

print('done')
```
![alt text][img_augment_image]


### (4) Add augmented images to make each sign type has at least 1000 examples.

```
print('X, y shapes:', X_train_normalized.shape, y_train.shape)

input_indices = []
output_indices = []

for class_n in range(n_classes):
    print(class_n, ': ', end='')
    class_indices = np.where(y_train == class_n)
    n_samples = len(class_indices[0])
    if n_samples < 1000:
        for i in range(1000 - n_samples):
            input_indices.append(class_indices[0][i%n_samples])
            output_indices.append(X_train_normalized.shape[0])
            new_img = X_train_normalized[class_indices[0][i % n_samples]]
            new_img = augment_image(new_img)
            X_train_normalized = np.concatenate((X_train_normalized, [new_img]), axis=0)
            y_train = np.concatenate((y_train, [class_n]), axis=0)
            if i % 50 == 0:
                print('|', end='')
            elif i % 10 == 0:
                print('-',end='')
    print('')
            
print('X, y shapes:', X_train_normalized.shape, y_train.shape)
```

### (5) Show the histogram of label frequency after filling augmented images

```
# histogram of label frequency
hist, bins = np.histogram(y_train, bins=n_classes)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.savefig('./test_images_output/label_frequency_after_filling_agumented_images.png')
plt.show()
```
![alt text][img_label_frequency_after_filling_agumented_images]

### (6) Shuffle the training dataset

```
## Shuffle the training dataset

from sklearn.utils import shuffle

X_train_normalized, y_train = shuffle(X_train_normalized, y_train)

print('done')
```

### (7) Split validation dataset off from training dataset

```
## Split validation dataset off from training dataset

from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(X_train_normalized, y_train, 
                                                                test_size=0.20, random_state=42)

print("Old X_train size:",len(X_train_normalized))
print("New X_train size:",len(X_train))
print("X_validation size:",len(X_validation))
```
```
Old X_train size: 51690
New X_train size: 41352
X_validation size: 10338
```

### Model Architecture

### Use modified LeNet Model Architecture

Original LeNet Model Architecture
![LeNet Architecture](lenet.png)
Source: Yan LeCun


First, I began the original LeNet-5 implementation shown in the classroom at the end of the CNN lesson. I got a validation set accuracy of about 0.89.

Then, I changed the second covolution layer output to 26, added additional convolution layer (no max pool) flattened and concatenated with flattened layer 2 output. i.e. 5x5x16 branched off to 5x5 convolution (output of 1x1x400), each flattened (to 400) and concatenated (650). I got a  validation set accuracy of about 0.92.

Finally, I add dropout layer in the end. I got a validation set accuracy of about 0.94.



My Modified Model Architecture:

- (1) Input : image 32x32x1 ==> memory = 32x32x1 ~ 1K, weights = 0

- (2) Convolution 5x5 layer 1. The output shape is 28x28x16. ==> memory : 28x28x16 ~ 13K, weights = (5x5x1)x16 = 400

- (3) Activation 1: Relu.

- (4) Pooling layer 1 : The output shape is 14x14x16. ==> memory : 14x14x16 ~ 3K, weights = 0

- (5) Convolution 5x5 layer 2. The output shape is 10x10x26. ==> memory : 10x10x26 ~ 2.6K, weights = (5x5x16)x26 = 10400

- (6) Activation 2. Relu

- (7) Pooling layer 2. The output shape should be 5x5x26. ==> 5x5x26 = 650, weights = 0

- (8) Convolution 5x5 layer 3. The output shape is 1x1x400. ==> memory : 1x1x400 ~ 0.4K, weights = (5x5x26)x400 = 260000

- (9) Activation 3. Relu

- (10) Flatten layers from numbers 8 (1x1x400 -> 400) and 6 (5x5x26 -> 650)

- (11) Concatenate flattened layers to a single size-1050 layer

- (12) dopout layer

- (13) Fully connected layer (1050 in, 43 out)





```
from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x16.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 16), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(16))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x16. Output = 14x14x16.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x26.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 26), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(26))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x26. Output = 5x5x26.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # Fully Connected Layer 2. Input = 5x5x26. Output = 1x1x400.
    conv3_W  = tf.Variable(tf.truncated_normal(shape=(5, 5, 26, 400), mean = mu, stddev = sigma))
    conv3_b  = tf.Variable(tf.zeros(400))
    conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b
    
    # SOLUTION: Activation.
    conv3 = tf.nn.relu(conv3)
    
    # SOLUTION: Flatten. Input = 400. Output = 400.
    layer3flat = flatten(conv3)
    print("layer3flat shape:",layer3flat.get_shape())
    
     # SOLUTION: Flatten. Input = 5x5x26. Output = 650.
    layer2flat = flatten(conv2)
    print("layer2flat shape:",layer2flat.get_shape())
    
    # Concat layer2flat and layer3flat. Input = 400 + 650. Output = 1050
    x = tf.concat([layer3flat, layer2flat], 1)
    print("x shape:",x.get_shape())
    
    # Dropout
    x = tf.nn.dropout(x, 0.5)
    
    # TODO: Layer 4: Fully Connected. Input = 1050. Output = 43.
    fc4_W = tf.Variable(tf.truncated_normal(shape=(1050, 43), mean = mu, stddev = sigma))
    fc4_b = tf.Variable(tf.zeros(43))    
    logits = tf.add(tf.matmul(x, fc4_W), fc4_b)
    
    return logits
print('done')
```

### Train, Validate and Test the Model
A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

```
import tensorflow as tf

EPOCHS = 60
BATCH_SIZE = 100

print('done')
```
```
tf.reset_default_graph() 

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32) # probability to keep units
one_hot_y = tf.one_hot(y, 43)

print('done')
```
```
rate = 0.0009

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

print('done')
```
```
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

print('done')
```
```
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")
```
```
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver2 = tf.train.import_meta_graph('./lenet.meta')
    saver2.restore(sess, "./lenet")
    test_accuracy = evaluate(X_test_normalized, y_test)
    print("Test Set Accuracy = {:.3f}".format(test_accuracy))
```
```
INFO:tensorflow:Restoring parameters from ./lenet
Test Set Accuracy = 0.947
```


---

## Step 3: Test a Model on New Images

To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.

You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.


I tried out trained model on manually captured images (in Google Maps). You can find them in traffic-signs-data/custom/. 



### Load and Output the Images

```
# Reinitialize and re-import if starting a new kernel here
import matplotlib.pyplot as plt
%matplotlib inline

import tensorflow as tf
import numpy as np
import cv2

print('done')
```
```
### Load the images and plot them here.
### Feel free to use as many code cells as needed.

#reading in an image
import glob
import os
from skimage import io
import matplotlib.image as mpimg

fig, axs = plt.subplots(4,6, figsize=(15, 6))
fig.subplots_adjust(hspace = .2, wspace=.001)
axs = axs.ravel()

my_images = []
my_labels = [
    21, # "example_00001"
    39, # "example_00002"
    17, # "example_00003"
    17, # "example_00004"
    17, # "example_00005"
    39, # "example_00006"
    39, # "example_00007"
    40, # "example_00008"
    40, # "example_00009"
    34, # "example_00010"
    28, # "example_00011"
    39, # "example_00012"
    0, # "example_00013"
    17, # "example_00014"
    38, # "example_00015"
    13, # "example_00016"
    40, # "example_00017"
    13, # "example_00018"
    38, # "example_00019"
    38, # "example_00020"
    11, # "example_00021"
    0, # "example_00022"
    28, # "example_00023"
    0, # "example_00024"
]

for index in range(0, 24):
    image = io.imread(os.getcwd() + '/traffic-signs-data/custom/' + "example_{0:0>5}".format(index + 1) + '.png')
#     plt.imshow(image)
    axs[index].axis('off')
    axs[index].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    my_images.append(image)
    print (index, signName[my_labels[index]])

my_images = np.asarray(my_images)

my_images_gry = np.sum(my_images/3, axis=3, keepdims=True)

my_images_normalized = (my_images_gry - 128)/128 

print(my_images_normalized.shape)

fig.savefig('./test_images_output/test_model_on_new_images.png')
```
```
0 Double curve
1 Keep left
2 No entry
3 No entry
4 No entry
5 Keep left
6 Keep left
7 Roundabout mandatory
8 Roundabout mandatory
9 Turn left ahead
10 Children crossing
11 Keep left
12 Speed limit (20km/h)
13 No entry
14 Keep right
15 Yield
16 Roundabout mandatory
17 Yield
18 Keep right
19 Keep right
20 Right-of-way at the next intersection
21 Speed limit (20km/h)
22 Children crossing
23 Speed limit (20km/h)
(24, 32, 32, 1)
```

![alt text][img_test_model_on_new_images]

### Predict the Sign Type for Each Image

My images appear to be more easily distinguishable than quite a few images from the original dataset. The test accuracy is 0.95.

```
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver3 = tf.train.import_meta_graph('./lenet.meta')
    saver3.restore(sess, "./lenet")
    my_accuracy = evaluate(my_images_normalized, my_labels)
    print("Test Set Accuracy = {:.3f}".format(my_accuracy))
```

```
INFO:tensorflow:Restoring parameters from ./lenet
Test Set Accuracy = 0.958
```

### Analyze Performance


I noticed that my images tend to be quite a bit blur and darker and might occupy a different range in the color space, possibly a range that the model was not trained on.

In addition, the GTSRB dataset states that the images "contain a border of 10 % around the actual traffic sign (at least 5 pixels) to allow for edge-based approaches" and the images that I used do not all include such a border. This could be another source of confusion for the model.


```
### Visualize the softmax probabilities here.
### Feel free to use as many code cells as needed.

softmax_logits = tf.nn.softmax(logits)
top_k = tf.nn.top_k(softmax_logits, k=3)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('./lenet.meta')
    saver.restore(sess, "./lenet")
    my_softmax_logits = sess.run(softmax_logits, feed_dict={x: my_images_normalized, keep_prob: 1.0})
    my_top_k = sess.run(top_k, feed_dict={x: my_images_normalized, keep_prob: 1.0})

    
    fig, axs = plt.subplots(len(my_images),4, figsize=(12, 14), dpi=100)
    fig.subplots_adjust(hspace = .4, wspace=.2)
    axs = axs.ravel()

    for i, image in enumerate(my_images):
        axs[4*i].axis('off')
        axs[4*i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[4*i].set_title('input: %s'% my_labels[i])
        guess1 = my_top_k[1][i][0]
        index1 = np.argwhere(y_validation == guess1)[0]
        axs[4*i+1].axis('off')
        axs[4*i+1].imshow(X_validation[index1].squeeze(), cmap='gray')
        axs[4*i+1].set_title('top guess: {} ({:.0f}%)'.format(guess1, 100*my_top_k[0][i][0]))
        guess2 = my_top_k[1][i][1]
        index2 = np.argwhere(y_validation == guess2)[0]
        axs[4*i+2].axis('off')
        axs[4*i+2].imshow(X_validation[index2].squeeze(), cmap='gray')
        axs[4*i+2].set_title('2nd guess: {} ({:.0f}%)'.format(guess2, 100*my_top_k[0][i][1]))
        guess3 = my_top_k[1][i][2]
        index3 = np.argwhere(y_validation == guess3)[0]
        axs[4*i+3].axis('off')
        axs[4*i+3].imshow(X_validation[index3].squeeze(), cmap='gray')
        axs[4*i+3].set_title('3rd guess: {} ({:.0f}%)'.format(guess3, 100*my_top_k[0][i][2]))
        
fig.savefig('./test_images_output/test_model_on_new_images_top_3_guess.png')
```

![alt text][img_test_model_on_new_images_top_5_guess]


### Output Top 5 Softmax Probabilities For Each Image Found on the Web

For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 

The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.

`tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.

Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:

```
# (5, 6) array
a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
         0.12789202],
       [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
         0.15899337],
       [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
         0.23892179],
       [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
         0.16505091],
       [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
         0.09155967]])
```

Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:

```
TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
       [ 0.28086119,  0.27569815,  0.18063401],
       [ 0.26076848,  0.23892179,  0.23664738],
       [ 0.29198961,  0.26234032,  0.16505091],
       [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
       [0, 1, 4],
       [0, 5, 1],
       [1, 3, 5],
       [1, 4, 3]], dtype=int32))
```

Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

```
fig, axs = plt.subplots(12,2, figsize=(9, 19), dpi=100)
axs = axs.ravel()

for i in range(len(my_softmax_logits)):
    if i%2 == 0:
        axs[i].axis('off')
        axs[i].imshow(cv2.cvtColor(my_images[i//2], cv2.COLOR_BGR2RGB))
        axs[i].set_title('input: %s'% my_labels[i//2])
    else:
        axs[i].bar(np.arange(n_classes), my_softmax_logits[(i-1)//2]) 
        axs[i].set_ylabel('Softmax probability')
fig.savefig('./test_images_output/test_model_on_new_images_softmax_probability.png')
```

![alt text][img_test_model_on_new_images_softmax_probability]





### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


