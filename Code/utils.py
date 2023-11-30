"""
This file contains utility functions.
"""

import math
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K


# Enable dynamic memory allocation
def enable_gpu_mem_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


class Trainer(object):
    # Define training step
    @tf.function
    def train_step(self, x, y, mdl, opt):
        with tf.GradientTape() as tape:
            logits = mdl(x, training=True)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, logits, from_logits=True))
        grads = tape.gradient(loss, mdl.trainable_variables)
        opt.apply_gradients(zip(grads, mdl.trainable_variables))
        return loss

    # Define training step for bias correction
    def bias_correction_step(self, x, y, mdl, opt):
        with tf.GradientTape() as tape:
            logits = mdl(x, training=True)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, logits, from_logits=True))
        grads = tape.gradient(loss, (mdl.w, mdl.b))
        opt.apply_gradients(zip(grads, (mdl.w, mdl.b)))
        return loss

    # Define training and distillation step
    @tf.function
    def train_distill_step(self, x, y, x_r, y_r, l_r, mdl, opt, task, T=2.0):
        with tf.GradientTape() as tape:
            # Compute classification loss
            logits = mdl(x, training=True)
            logits_r = mdl(x_r, training=True)
            class_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(tf.concat((y, y_r), axis=0), tf.concat((logits, logits_r), axis=0), from_logits=True))
            # Compute distillation loss
            mask = tf.math.less(tf.range(0, y.shape[-1]), tf.multiply(tf.constant(2, dtype=tf.int32), task))
            masked_l_r = tf.boolean_mask(tf.divide(l_r, tf.constant(T, dtype=tf.float32)), mask, axis=1)
            masked_logits_r = tf.boolean_mask(tf.divide(logits_r, tf.constant(T, dtype=tf.float32)), mask, axis=1)
            dist_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(tf.nn.softmax(masked_l_r), masked_logits_r, from_logits=True))
            # Combine losses
            num_old_classes = tf.multiply(tf.constant(2.0, dtype=tf.float32), task)
            weight = tf.divide(num_old_classes, tf.add(num_old_classes, tf.constant(2.0, dtype=tf.float32)))
            loss = tf.add(tf.multiply(tf.subtract(tf.constant(1.0, dtype=tf.float32), weight), class_loss), tf.multiply(weight, dist_loss))
        grads = tape.gradient(loss, mdl.trainable_variables)
        opt.apply_gradients(zip(grads, mdl.trainable_variables))
        return loss


# Standardization
def standardize(batch):
    x = tf.divide(tf.cast(batch["image"], tf.float32), tf.constant(255.0, tf.float32))
    y = tf.one_hot(tf.cast(batch["label"], tf.int32), 10)
    return x, y


def combine_img(x_plt):
    dim = int(np.ceil(np.sqrt(x_plt.shape[0])))
    img = np.zeros((x_plt.shape[1] * dim, x_plt.shape[2] * dim, x_plt.shape[3]))
    for i in range(dim):
        for j in range(dim):
            idx = j * dim + i
            if idx < x_plt.shape[0]:
                img[i * x_plt.shape[1]:(i + 1) * x_plt.shape[1], j * x_plt.shape[2]:(j + 1) * x_plt.shape[1]] = x_plt[idx].numpy()
            else:
                img[i * x_plt.shape[1]:(i + 1) * x_plt.shape[1], j * x_plt.shape[2]:(j + 1) * x_plt.shape[1]] = np.zeros_like(x_plt[0])
    return img


def plot(x_plt, name, path):
    img = combine_img(x_plt)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.savefig(fname=path+"/{}.png".format(name), format="png")


def reinitialize_model(mdl):
    for layer in mdl.layers:
        if layer.trainable:
            if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
                if hasattr(layer, "kernel"):
                    layer.kernel.assign(layer.kernel_initializer(layer.kernel.shape))
                if hasattr(layer, "bias"):
                    if layer.bias is not None:
                        layer.bias.assign(layer.bias_initializer(layer.bias.shape))
            if isinstance(layer, tfa.layers.InstanceNormalization) or isinstance(layer, tf.keras.layers.BatchNormalization):
                if hasattr(layer, "gamma"):
                    layer.gamma.assign(layer.gamma_initializer(layer.gamma.shape))
                if hasattr(layer, "beta"):
                    layer.beta.assign(layer.beta_initializer(layer.beta.shape))

def make_grid(images, N_COLUMNS=10, PADDING=2):
    '''
    Make grid images frpm tf.tensors of multiple images
    '''
    nmaps = images.shape[0]
    xmaps = min(N_COLUMNS, nmaps) # number of element images in x-axis
    ymaps = int(math.ceil(float(nmaps) / xmaps)) # number of element images in y-axis

    paddings = tf.constant([[1, 1,], [1, 1], [0, 0]])

    index = 0 
    columns = []
    blank = tf.pad(tf.zeros(images.shape[1:]), paddings)
    for y in range(ymaps + 1):
        if index >= nmaps:
            grid = tf.concat(columns, axis=0)
            break
        column = tf.pad(images[index, :, :, :], paddings)
        index += 1
        for _ in range(xmaps - 1):
            if index >= nmaps:
                column = tf.concat([column, blank], axis=1)
            else:
                column = tf.concat([column, tf.pad(images[index, :, :, :], paddings)], axis=1)
            index += 1
        columns.append(column)

    pil_grid = tf.keras.utils.array_to_img(grid)

    return pil_grid


class SupConLoss(tf.keras.layers.Layer):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    Self-authored Tensorflow implementation
    """
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    @tf.function
    def call(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Here, labels are always avaliable.

        Args:
            features: hidden vector of shape [bsz, n_views, ...]. tf.tensor
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.reshape(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = tf.eye(batch_size)
        elif labels is not None:
            labels = tf.reshape(labels, (-1, 1))
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = tf.cast(tf.math.equal(labels, tf.transpose(labels)), dtype=tf.dtypes.float32)
        else:
            mask = mask.float()

        contrast_count = features.shape[1]
        contrast_feature = tf.concat(tf.unstack(features, axis=1), axis=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = tf.math.divide(
            tf.linalg.matmul(anchor_feature, tf.transpose(contrast_feature)), 
            self.temperature)

        # for numerical stability
        logits_max = tf.math.reduce_max(anchor_dot_contrast, axis=1, keepdims=True)
        logits = anchor_dot_contrast - tf.stop_gradient(logits_max)

        # tile mask
        mask = tf.tile(mask, tf.constant([anchor_count, contrast_count], dtype=tf.dtypes.int32))

        # mask-out self-contrast cases
        logits_mask = tf.ones_like(mask) - tf.eye(mask.shape[0])
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = tf.math.exp(logits) * logits_mask
        log_prob = logits - tf.math.log(tf.math.reduce_sum(exp_logits, axis=1, keepdims=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = tf.math.reduce_sum(mask * log_prob, axis=1) / tf.math.reduce_sum(mask, axis=1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = tf.math.reduce_mean(tf.reshape(loss, (anchor_count, batch_size)))

        return loss


def get_batch_size(batch_size, current_data_proportion, current_class):
    if current_data_proportion == 0:
        batch_size_current_classes = int(2 * batch_size * (2 / (current_class[0] + 2)))
    elif current_data_proportion > 0.5:
        raise "current_data_proportion must not greater than 0.5"
    else:
        batch_size_current_classes = int(2 * batch_size * current_data_proportion)
    batch_size_previous_classes = 2 * batch_size - batch_size_current_classes
    return batch_size_previous_classes, batch_size_current_classes


def sample_batch(batch_x, batch_y, batch_size):
    indices = np.random.permutation(batch_size)
    x_ds = tf.gather(batch_x, indices)
    y_ds = tf.gather(batch_y, indices)
    return x_ds, y_ds


def def_beta(beta=1.0, gamma=0.0, comp_beta=1.0, comp_gamma=0.0):
    global global_beta
    global_beta = beta
    global global_gamma
    global_gamma = gamma
    global global_comp_beta
    global_comp_beta = comp_beta
    global global_comp_gamma
    global_comp_gamma = comp_gamma


def adjusted_sigmoid(x):
    return K.sigmoid(global_beta * (x - global_gamma))

def comp_sigmoid(x):
    return K.sigmoid(global_comp_beta * (x - global_comp_gamma))