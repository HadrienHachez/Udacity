import cv2, os
import numpy as np
import matplotlib.image as mpimg

SHAPE = (160, 320, 3)


# load rgb image from file
def load_image(data_dir, image_file):
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))


# remove sky and car from image
def crop(img):
    cropped = img
    cropped = img[90:140, :, :]
    return cropped


# rgb values normalized to -1 and 1
def normalize(img):
    normalized = img
    for x in range(160):
        for y in range(320):
            normalized[x][y] = normalized[x][y]/127.5-1.0
    return normalized


# gaussian blur
def blur(img):
    blurred = cv2.GaussianBlur(img, (5, 5),0)
    return blurred


# canny edge detection
def canny(img):
    v = np.median(img)
    sigma = 0.2
    lower = int(max(150, (1.0 - sigma) * v))
    upper = int(max(255, (1.0 + sigma) * v))
    canny = cv2.Canny(img, lower, upper)
    return canny


# resize cropped image
def resize(img):
    resized = cv2.resize(img, (320, 160), cv2.INTER_AREA)
    return resized


# converts rgb to yuv 
def rgb2yuv(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    return yuv


# flips image and adjusts steering
def flip(image, steering_angle):
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


# image preprocessing step before CNN
def preprocess(image):
    image = crop(image)
    image = resize(image)
    #image = blur(image)
    
    #image = canny(image)
    #image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    #image = cv2.rectangle(image, (90, 90), (230, 160), (0,0,0), thickness=-1)

    image = rgb2yuv(image)

    return image


# Generate augmented images
def augment(data_dir, center, left, right, steering_angle):
    image = load_image(data_dir, center)

    image, steering_angle = flip(image, steering_angle)    # flip augmentation
    return image, steering_angle


# Generate training image
def batch_gen(data_dir, image_paths, steering_angles, batch_size, is_training):
    images = np.empty([batch_size, 160, 320, 3])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]

            # argumentation only if is in training
            if is_training and np.random.rand() < 0.6:
                image, steering_angle = augment(data_dir, center, left, right, steering_angle)
            else:
                image = load_image(data_dir, center) 

            # preprocessing
            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers
