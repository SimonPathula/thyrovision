import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.cm as cm 
from utils.logger import logger

def make_gradcam_heatmap(img_array, model):

    with tf.GradientTape() as tape:

        x1 = model.block1(img_array, training=False)
        x2 = model.block2(x1, training=False)

        x3 = model.block3(x2, training=False)

        x = model.pcb1(x2, x3)

        x = model.block4(x, training=False)

        x = model.pcb2(x3, x, training=False)

        x = model.block5(x, training=False)

        x = model.dwsc1(x, training=False)

        last_conv_output = model.dwsc2(x, training=False)

        x = model.gap(last_conv_output)
        x = model.dropout(x, training=False)

        preds = model.classifier(x)

        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, last_conv_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    last_conv_output = last_conv_output[0]

    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, alpha = 0.4):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    
    # Always convert to RGB to ensure 3 channels
    if img.mode != "RGB":
        img = img.convert("RGB")
        
    # Now convert to numpy array (H, W, 3)
    img = np.array(img)
    
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img