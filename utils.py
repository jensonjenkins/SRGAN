from tqdm.auto import tqdm
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2


def train(gan_model, generator, discriminator, vgg, 
            epochs, train_lr, train_hr, batch_size):

    for e in tqdm(range(epochs)):
        fake_label = np.zeros((batch_size, 1))
        real_label = np.ones((batch_size, 1))

        g_losses, d_losses = [], []

        cur_loss = 41
        for b in tqdm(range(len(train_hr))):
            lr_img, hr_img = train_lr[b], train_hr[b]

            fake_img = generator.predict_on_batch(lr_img)

            #train discriminator
            discriminator.trainable=True
            d_loss_gen, _ = discriminator.train_on_batch(fake_img, fake_label)
            d_loss_real, _ = discriminator.train_on_batch(hr_img, real_label)
            discriminator.trainable=False

            # average only for reporting purposes
            d_loss = 0.5*np.add(d_loss_gen, d_loss_real)

            img_features = vgg.predict(hr_img, verbose=0)

            #train generator
            g_loss, _, _, _, _ = gan_model.train_on_batch([lr_img, hr_img], [real_label, img_features])

            if g_loss < cur_loss:
                cur_loss = g_loss
                generator.save_weights(f"SRGAN_extended_({cur_loss}l).h5")

            print(f"\rg_loss:{g_loss}, d_loss:{d_loss}", end="")
            d_losses.append(d_loss)
            g_losses.append(g_loss)

        g_losses, d_losses = np.array(g_losses), np.array(d_losses)
        g_loss = np.sum(g_losses, axis=0)/len(g_losses)
        d_loss = np.sum(d_losses, axis=0)/len(d_losses)

        print(f"epoch:{e+1} g_loss:{g_loss}, d_loss:{d_loss}")

        generator.save_weights(f"SRGAN_{e+11}e_extended.h5")

"""
Normalization function
"""
def min_max_norm(input):
  min_value = tf.reduce_min(input)
  max_value = tf.reduce_max(input)

  return (input - min_value) / (max_value - min_value)


"""
Compare Generated Images
input:
    - im_path: image path (string format)-> image must be of size 32x32 
    - generator: generator model

returns: generated image by model in ndarray format
"""
def compare_generated(im_path, generator):

    im_demo = cv2.imread(im_path)
    im_demo = cv2.cvtColor(im_demo, cv2.COLOR_BGR2RGB)
    im_demo = im_demo/255

    im_demo = np.expand_dims(im_demo, axis = 0)

    im_gen = generator.predict(im_demo, verbose=0)

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title("low res image")
    plt.imshow(im_demo[0])
    plt.subplot(1, 2, 2)
    plt.title("SRGAN generated")
    plt.imshow(min_max_norm(im_gen[0]))
    plt.show()

    return im_gen[0]

"""
Save image to local directory
"""

def save_img(im_path, img):
    cv2.imwrite(im_path, cv2.cvtColor(255*img, cv2.COLOR_RGB2BGR))