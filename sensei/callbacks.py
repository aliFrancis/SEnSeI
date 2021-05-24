import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
from tensorflow.keras.callbacks import Callback, TensorBoard
import tensorflow as tf

class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or "learning_rate" in logs:
            return
        logs["learning_rate"] = self.model.optimizer.lr


class ImageCallback(Callback):
    """
    Callback to write predictions of a few test images to TensorBoard each epoch.

    At start of training, the correct masks are written, so they can be referred to
    by the user.
    """
    def __init__(self, display_loader, file_writer, idxs=None):
        self.display_loader = display_loader
        self.file_writer = file_writer
        if idxs is None:
            self.idxs = list(range(len(self.display_loader)))
        else:
            self.idxs = idxs

        super(ImageCallback, self).__init__()

    def on_train_begin(self,logs=None):
        # Add groundtruth as first panel
        truth_raw = []
        disp_images = []
        for i in self.idxs:
            ins,masks = self.display_loader[i]
            truth_raw.append(masks)
            if isinstance(ins,tuple):
                disp_images.append(ins[0])
            else:
                disp_images.append(ins)
        truth = np.squeeze(np.array(truth_raw))
        disp_images = np.array(disp_images)
        if disp_images.ndim == 6:
            disp_images = np.moveaxis(disp_images,2,-1)
        disp_images = np.squeeze(disp_images)
        assert disp_images.ndim==4, 'Shape conversion failed!'
        figure = image_grid(disp_images,truth)
        image = plot_to_image(figure)
        plt.close('all')
        # Log the confusion matrix as an image summary.
        with self.file_writer.as_default():
            tf.summary.image("Test Images", image, step=0)

    def on_epoch_end(self,epoch, logs):
        # Use the model to predict the values from the validation dataset.
        test_pred_raw = []
        disp_images = []
        for i in self.idxs:
            ins,_ = self.display_loader[i]
            test_pred_raw.append(self.model.predict(ins))
            if isinstance(ins,tuple):
                disp_images.append(ins[0])
            else:
                disp_images.append(ins)
        test_pred_raw = np.array(test_pred_raw)
        disp_images = np.array(disp_images)
        test_pred = np.argmax(test_pred_raw, axis=-1)
        if disp_images.ndim == 6:
            disp_images = np.moveaxis(disp_images,2,-1)
        disp_images = np.squeeze(disp_images)
        assert disp_images.ndim==4, 'Shape conversion failed!'

        figure = image_grid(disp_images,test_pred_raw)
        image = plot_to_image(figure)
        plt.close('all')
        # Log the confusion matrix as an image summary.
        with self.file_writer.as_default():
            tf.summary.image("Test Images", image, step=epoch+1)

# Adapted from: https://www.tensorflow.org/tensorboard/image_summaries

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)

    return image

def image_grid(imgs,labels):
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(15,15))
    labels = convert_labels_for_plot(labels)
    for i,img in enumerate(imgs):
        # Start next subplot.
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        img =imgs[i,...,3:0:-1]
        label = labels[i,...]
        img = (img-img.min())/(img.max()-img.min())
        plt.imshow(exposure.equalize_adapthist(img,clip_limit=0.005))
        plt.imshow(np.squeeze(label))
    return figure

def convert_labels_for_plot(labels):
    #Converts ONE-HOT masks to plottable overlays
    plotted = np.zeros((*labels.shape[:-1],4))
    colour_list = [[1.0,1.0,1.0,0.0], # Blank
                   [1.0,1.0,0.0,0.15], # Yellow
                   [1.0,0.0,0.0,0.15]] # Red
    for i in range(labels.shape[-1]):
        plotted[np.argmax(labels,axis=-1)==i] = colour_list[i]
    return plotted
