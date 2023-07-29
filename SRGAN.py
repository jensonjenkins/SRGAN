import cv2
import numpy as np
import tensorflow as tf
from keras.applications import VGG19


"""
GENERATOR
"""
def res_unit(input):
    x = tf.keras.layers.Conv2D(64, 3, padding="same")(input)
    x = tf.keras.layers.BatchNormalization(momentum=0.5)(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.5)(x)
    x = tf.keras.layers.Add()([input, x])
    return x


def upscale(input):
    x = tf.keras.layers.Conv2D(256, 3, padding="same")(input)
    x = tf.keras.layers.UpSampling2D(size = 2)(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)

    return x

B = 16

def create_gen(input, num_res_block):
    x = tf.keras.layers.Conv2D(64, 9, padding="same")(input)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    residual = x

    for i in range(num_res_block):
        x = res_unit(x)
    
    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.5)(x)
    x = tf.keras.layers.Add()([x, residual])

    for i in range(2):
        x = upscale(x)

    output = tf.keras.layers.Conv2D(3, 9, padding="same")(x)

    return tf.keras.Model(inputs=input, outputs=output)


"""
DISCIMINATOR
"""

def d_block(input, filters, strides=1, bn=True):
    x = tf.keras.layers.Conv2D(filters, 3, strides=strides, padding="same")(input)
    if bn:
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    return x

def create_dis(input):
    df = 64
    x = d_block(input, df, bn=False)
    x = d_block(x, df, strides=2)

    for i in range(1, 4):
        x = d_block(x, df*(2**i))
        x = d_block(x, df*(2**i), strides=2)
    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(df*16)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    validity = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs=input, outputs=validity)


"""
Combined Model (GAN)
"""

def create_gan(gen, dis, vgg, lr_i, hr_i):
    gen_img = gen(lr_i)

    gen_features = vgg(gen_img)

    dis.trainable=False
    validity = dis(gen_img)

    return tf.keras.Model(inputs=[lr_i, hr_i], outputs=[validity, gen_features])

"""
Build VGG Model
"""
def build_vgg(hr_shape):
  vgg = VGG19(weights="imagenet", include_top=False, input_shape=hr_shape)
  return tf.keras.Model(inputs=vgg.inputs, outputs=vgg.layers[10].output)

"""
Compile GAN
"""
def compile_gan(weights="SRGAN_10e_extended_(40l).h5"):
    hr_shape = (128, 128, 3)
    lr_shape = (32, 32, 3)

    lr_ip = tf.keras.layers.Input(shape=lr_shape)
    hr_ip = tf.keras.layers.Input(shape=hr_shape)

    generator = create_gen(lr_ip, 16)

    discriminator = create_dis(hr_ip)
    discriminator.compile(loss="binary_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])

    vgg = build_vgg((128, 128, 3))
    vgg.trainable=False
    gan_model = create_gan(generator, discriminator, vgg, lr_ip, hr_ip)
    gan_model.compile(loss=["binary_crossentropy", "mse"],
                        loss_weights=[1e-3, 1],
                        optimizer="adam",
                        metrics=["accuracy"])
    generator.load_weights(weights)
    return gan_model

"""
Compile only generator 
"""
def compile_gen(weights="SRGAN_10e_extended_(40l).h5"):
    lr_shape = (32, 32, 3)
    lr_ip = tf.keras.layers.Input(shape=lr_shape)

    generator = create_gen(lr_ip, 16)
    temp = tf.zeros(shape=(1, 32, 32, 3))
    _ = generator(temp)

    generator.load_weights(weights)
    return generator

