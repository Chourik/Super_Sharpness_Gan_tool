## Imports 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from skimage.transform import resize
from skimage.transform import rescale
from keras.layers.core import Activation
from tensorflow.keras.layers import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Model
from keras.layers import ELU, PReLU, LeakyReLU
from keras.layers import add
from keras.layers import Dense
from keras.layers.core import Flatten
from keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
from keras.models import Model
from tensorflow.keras.models import load_model
import os 
import sys
# import fiftyone.zoo as foz
# import fiftyone as fo
from tqdm import tqdm
import cv2

### Image preprocessing functions

# Takes list of images and provide HR images in form of numpy array
def hr_images(images):
    images_hr = np.array(images)
    ##we normalize the images
    images_hr_norm = (images_hr - 127.5)/127.5 
    print('here_hr')
    return images_hr_norm


def downscale_image(image, n):
    return rescale(image, 1/n, anti_aliasing=True, multichannel=True)

# # Takes list of images and provide LR images in form of numpy array
# def lr_images(images_real , downscale):
#     """
#     Process a dataset of images and return a new array with low-resolution images.
    
#     Parameters:
#     X_train (numpy array): The dataset of images as a numpy array.
#     n (int): The downscaling factor.
    
#     Returns:
#     numpy array: The dataset with low-resolution images as a numpy array.
#     """
#     X_train_lowresolution = []
#     for image in X_train:
#         low_res_image = downscale_image(image, downscale)
#         X_train_lowresolution.append(low_res_image)
#     return X_train_lowresolution


# Subpixel Conv will upsample from (h, w, c) to (h/r, w/r, c/r^2)
def SubpixelConv2D(input_shape, scale=4):
    def subpixel_shape(input_shape):
        dims = [input_shape[0],input_shape[1] * scale,input_shape[2] * scale,int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape
    
    def subpixel(x):
        return tf.nn.depth_to_space(x, scale)
    return Lambda(subpixel, output_shape=subpixel_shape)


def normalize(input_data):
    return (input_data.astype(np.float32) - 127.5)/127.5 
    

def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)

new_shape = (96,96,3)

def load_images(image_folder):
    image_filenames = os.listdir(image_folder)
    images = []
    for filename in image_filenames:
        img = cv2.imread(os.path.join(image_folder, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = resize(img, new_shape, mode='reflect', anti_aliasing=True)
        ##img = img.astype(np.float32) / 255.0
        images.append(img)
    return np.array(images)


### GAN Model
# Residual block
def res_block_gen(model, kernal_size, filters, strides):    
    gen = model
    
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    # Using Parametric ReLU
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    model = add([gen, model])
    return model
    
    
def up_sampling_block(model, kernal_size, filters, strides):
    
    # In place of Conv2D and UpSampling2D we can also use Conv2DTranspose (Both are used for Deconvolution)
    # Even we can have our own function for deconvolution (i.e one made in Utils.py)
    #model = Conv2DTranspose(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = UpSampling2D(size = 2)(model)
    model = LeakyReLU(alpha = 0.2)(model)
    
    return model
  
class Generator(object):
    def __init__(self, noise_shape):
        
        self.noise_shape = noise_shape
    def generator(self):
        
        gen_input = Input(shape = self.noise_shape)
     
        model = Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same")(gen_input)
        model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
        
        gen_model = model
        
        # Using 16 Residual Blocks
        for index in range(16):
            model = res_block_gen(model, 3, 64, 1)
     
        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)
        model = BatchNormalization(momentum = 0.5)(model)
        model = add([gen_model, model])
     
        # Using 2 UpSampling Blocks
        for index in range(2):
            model = up_sampling_block(model, 3, 256, 1)
     
        model = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same")(model)
        model = Activation('tanh')(model)
    
        generator_model = Model(inputs = gen_input, outputs = model)
        return generator_model
    
def discriminator_block(model, filters, kernel_size, strides):
    
    model = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    model = LeakyReLU(alpha = 0.2)(model)
    
    return model
  
class Discriminator(object):

    def __init__(self, image_shape):
        
        self.image_shape = image_shape
    
    def discriminator(self):
        
        dis_input = Input(shape = self.image_shape)
        
        model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(dis_input)
        model = LeakyReLU(alpha = 0.2)(model)
        
        model = discriminator_block(model, 64, 3, 2)
        model = discriminator_block(model, 128, 3, 1)
        model = discriminator_block(model, 128, 3, 2)
        model = discriminator_block(model, 256, 3, 1)
        model = discriminator_block(model, 256, 3, 2)
        model = discriminator_block(model, 512, 3, 1)
        model = discriminator_block(model, 512, 3, 2)
        
        model = Flatten()(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha = 0.2)(model)
       
        model = Dense(1)(model)
        model = Activation('sigmoid')(model) 
        
        discriminator_model = Model(inputs = dis_input, outputs = model)
        
        return discriminator_model
    
def vgg_loss(y_true, y_pred, model):
    return K.mean(K.square(model(y_true) - model(y_pred)))

def vgg_loss_wrapper(image_shape):
    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=image_shape)
    vgg19.trainable = False
    for l in vgg19.layers:
        l.trainable = False
    model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
    model.trainable = False

    def loss_function(y_true, y_pred):
        return vgg_loss(y_true, y_pred, model)
    
    return loss_function

def get_gan_network(discriminator, shape, generator, optimizer, vgg_loss):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=[x,gan_output])
    gan.compile(loss=[vgg_loss, "binary_crossentropy"], loss_weights=[1., 1e-3], optimizer=optimizer)

    return gan

def get_optimizer():
    
    adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    return adam

# Better to use downscale factor as 4
downscale = 4
image_shape = (384,384,3)

# def train(epochs, batch_size, output_dir, model_save_dir):
#     # Loads training and test data
#     batch_count = int(x_train_hr.shape[0] / batch_size)
#     shape = (image_shape[0]//downscale, image_shape[1]//downscale, image_shape[2])
    
#     generator = Generator(shape).generator()
#     discriminator = Discriminator(image_shape).discriminator()

#     optimizer = get_optimizer()
#     loss_function = vgg_loss_wrapper(image_shape)
#     generator.compile(loss=loss_function, optimizer=optimizer) #loss = vgg_loss
#     discriminator.compile(loss="binary_crossentropy", optimizer=optimizer)
    
#     gan = get_gan_network(discriminator, shape, generator, optimizer, loss_function)
    
#     loss_file = open(model_save_dir + 'losses.txt' , 'w+')
#     loss_file.close()

#     for e in range(1, epochs+1):
#         print ('-'*15, 'Epoch %d' % e, '-'*15)
#         for _ in tqdm(range(batch_count)):
            
#             rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size).astype(int)
#             image_batch_hr = np.array([x_train_hr[i] for i in rand_nums])
#             image_batch_lr = np.array([x_train_lr[i] for i in rand_nums])
#             generated_images_sr = generator.predict(image_batch_lr)

#             real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
#             fake_data_Y = np.random.random_sample(batch_size)*0.2
            
#             discriminator.trainable = True
            
#             d_loss_real = discriminator.train_on_batch(image_batch_hr, real_data_Y)
#             d_loss_fake = discriminator.train_on_batch(generated_images_sr, fake_data_Y)
#             discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
            
#             rand_nums = np.random.randint(0, len(x_train_hr), size=batch_size).astype(int)
#             image_batch_hr = np.array([x_train_hr[i] for i in rand_nums])
#             image_batch_lr = np.array([x_train_lr[i] for i in rand_nums])

#             gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
#             discriminator.trainable = False
#             gan_loss = gan.train_on_batch(image_batch_lr, [image_batch_hr,gan_Y])
            
            
#         print("discriminator_loss : %f" % discriminator_loss)
#         print("gan_loss :", gan_loss)
#         gan_loss = str(gan_loss)
        
#         loss_file = open(model_save_dir + 'losses.txt' , 'a')
#         loss_file.write('epoch%d : gan_loss = %s ; discriminator_loss = %f\n' %(e, gan_loss, discriminator_loss) )
#         loss_file.close()

#         if e == 1 or e % 10 == 0:
#             #plot_generated_images(output_dir, e, generator, x_test_hr, x_test_lr)
#             generator.save(model_save_dir + 'gen_model_2_%d.h5' % e)
#             discriminator.save(model_save_dir + 'dis_model_2_%d.h5' % e)