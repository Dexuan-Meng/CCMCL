"""
This file contains models.
"""

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_transform as tft
import keras
from keras import layers
import matplotlib.pyplot as plt
import abc
import numpy as np
import wandb
from tqdm import tqdm
import utils
from utils import get_batch_size, sample_batch, adjusted_sigmoid, comp_sigmoid, adjusted_tanh


class CNN(tf.keras.Model):
    """
    Simple and small CNN.
    """

    def __init__(self, n, activation_0, activation_1, activation_2, activation_3):
        super(CNN, self).__init__()
        self.n = n
        self.activation_fn_0 = activation_0
        self.activation_fn_1 = activation_1
        self.activation_fn_2 = activation_2
        self.activation_fn_3 = activation_3
        self.activation_layer = None
        self.relu = None
        self.conv0 = None
        self.norm0 = None
        self.conv1 = None
        self.norm1 = None
        self.conv2 = None
        self.norm2 = None
        self.pool = None
        self.flatten = None
        self.dense = None

    def build(self, input_shape):

        # Adjustable activations function
        self.activation_layer_0 = tf.keras.layers.Activation(self.activation_fn_0)
        self.activation_layer_1 = tf.keras.layers.Activation(self.activation_fn_1)
        self.activation_layer_2 = tf.keras.layers.Activation(self.activation_fn_2)

        self.conv0 = tf.keras.layers.Conv2D(128, 3, activation="linear", padding="SAME")
        self.norm0 = tfa.layers.InstanceNormalization()
        self.conv1 = tf.keras.layers.Conv2D(128, 3, activation="linear", padding="SAME")
        self.norm1 = tfa.layers.InstanceNormalization()
        self.conv2 = tf.keras.layers.Conv2D(128, 3, activation="linear", padding="SAME")
        self.norm2 = tfa.layers.InstanceNormalization()

        self.pool = tf.keras.layers.AveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(self.n, activation=self.activation_fn_3)
        super(CNN, self).build(input_shape)
        
    def call(self, inputs, training=None):
        output = self.conv0(inputs)
        output = self.norm0(output)
        output = self.activation_layer_0(output)
        output = self.pool(output)
        output = self.conv1(output)
        output = self.norm1(output)
        output = self.activation_layer_1(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.norm2(output)
        output = self.activation_layer_2(output)
        output = self.pool(output)
        output = self.flatten(output)
        output = self.dense(output)
        return output
    
    def model(self):
        x = tf.keras.Input(shape=(32, 32 ,3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


class DistCNN(tf.keras.Model):
    """
    Simple and small CNN.
    """

    def __init__(self, n):
        super(DistCNN, self).__init__()
        self.n = n
        self.relu = None
        self.sigmoid = None
        self.conv0 = None
        self.norm0 = None
        self.conv1 = None
        self.norm1 = None
        self.conv2 = None
        self.norm2 = None
        self.pool = None
        self.flatten = None
        self.dense = None

    def build(self, input_shape):
        self.relu = tf.keras.layers.Activation("relu")
        self.sigmoid = tf.keras.layers.Activation("sigmoid")
        self.conv0 = tf.keras.layers.Conv2D(128, 3, activation="linear", padding="SAME")
        self.norm0 = tfa.layers.InstanceNormalization()
        self.conv1 = tf.keras.layers.Conv2D(128, 3, activation="linear", padding="SAME")
        self.norm1 = tfa.layers.InstanceNormalization()
        self.conv2 = tf.keras.layers.Conv2D(128, 3, activation="linear", padding="SAME")
        self.norm2 = tfa.layers.InstanceNormalization()
        self.pool = tf.keras.layers.AveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(self.n, activation="linear")
        super(DistCNN, self).build(input_shape)
        
    def call(self, inputs, training=None):
        output = self.conv0(inputs)
        output = self.norm0(output)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv1(output)
        output = self.norm1(output)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.norm2(output)
        output = self.sigmoid(output)
        output = self.pool(output)
        output = self.flatten(output)
        output = self.dense(output)
        return output


class ValCNN(tf.keras.Model):
    """
    Simple and small CNN.
    """

    def __init__(self, n):
        super(ValCNN, self).__init__()
        self.n = n
        self.relu = None
        self.conv0 = None
        self.norm0 = None
        self.conv1 = None
        self.norm1 = None
        self.conv2 = None
        self.norm2 = None
        self.pool = None
        self.flatten = None
        self.dense = None

    def build(self, input_shape):
        self.relu = tf.keras.layers.Activation("relu")
        self.conv0 = tf.keras.layers.Conv2D(128, 3, activation="linear", padding="SAME")
        self.norm0 = tfa.layers.InstanceNormalization()
        self.conv1 = tf.keras.layers.Conv2D(128, 3, activation="linear", padding="SAME")
        self.norm1 = tfa.layers.InstanceNormalization()
        self.conv2 = tf.keras.layers.Conv2D(128, 3, activation="linear", padding="SAME")
        self.norm2 = tfa.layers.InstanceNormalization()
        self.pool = tf.keras.layers.AveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(self.n, activation="linear")
        super(ValCNN, self).build(input_shape)
        
    def call(self, inputs, training=None):
        output = self.conv0(inputs)
        output = self.norm0(output)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv1(output)
        output = self.norm1(output)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.norm2(output)
        output = self.relu(output)
        output = self.pool(output)
        output = self.flatten(output)
        output = self.dense(output)
        return output


class DataCompressor(object):
    """
    Compresses data into a smaller set of synthetic examples.
    """

    def __init__(self, batch_size, train_learning_rate, dist_learning_rate, K, T, mdl, I=10):
        super(DataCompressor, self).__init__()
        self.batch_size = batch_size
        self.mdl = mdl
        self.K = K
        self.T = T
        self.I = I
        self.dist_opt = tf.keras.optimizers.RMSprop(dist_learning_rate)
        self.train_opt = tf.keras.optimizers.SGD(train_learning_rate)

    @tf.function
    def distill_step(self, x, y, x_s, y_s):
        # Minimize cosine similarity between gradients
        with tf.GradientTape() as inner_tape:
            logits_x = self.mdl(x, training=False)
            loss_x = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, logits_x, from_logits=True))
        grads = inner_tape.gradient(loss_x, self.mdl.trainable_variables)
        with tf.GradientTape() as tape:
            # Make prediction using model
            with tf.GradientTape() as inner_tape:
                logits_s = self.mdl(x_s, training=False)
                loss_s = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_s, logits_s, from_logits=True))
            grads_s = inner_tape.gradient(loss_s, self.mdl.trainable_variables)
            # Compute cosine similarity
            dist_loss = tf.constant(0.0, dtype=tf.float32)
            for g, gs in zip(grads, grads_s):
                if len(tf.shape(g)) == 2:
                    g_norm = tf.math.l2_normalize(g, axis=0)
                    gs_norm = tf.math.l2_normalize(gs, axis=0)
                    inner = tf.reduce_sum(tf.multiply(g_norm, gs_norm), axis=0)
                if len(tf.shape(g)) == 4:
                    g_norm = tf.math.l2_normalize(g, axis=(0, 1, 2))
                    gs_norm = tf.math.l2_normalize(gs, axis=(0, 1, 2))
                    inner = tf.reduce_sum(tf.multiply(g_norm, gs_norm), axis=(0, 1, 2))
                dist_loss += tf.reduce_sum(tf.subtract(tf.constant(1.0, dtype=tf.float32), inner))
        dist_grads = tape.gradient(dist_loss, [x_s])
        self.dist_opt.apply_gradients(zip(dist_grads, [x_s]))
        return dist_loss

    # Define loss and gradients function
    @tf.function
    def calc_grads(self, x, y, mdl, training=True):
        with tf.GradientTape() as tape:
            logits = mdl(x, training=training)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, logits, from_logits=True))
        gradients = tape.gradient(loss, mdl.trainable_variables)
        return loss, gradients

    # Define training step
    @tf.function
    def train_step(self, x, y, mdl, opt):
        loss, grads = self.calc_grads(x, y, mdl, training=True)
        opt.apply_gradients(zip(grads, mdl.trainable_variables))
        return loss

    def compress(self, ds, c, img_shape, num_synth, buf=None, verbose=False):
        # Create and initialize synthetic data
        x_s = tf.Variable(tf.random.uniform((num_synth, img_shape[0], img_shape[1], img_shape[2]), maxval=tf.constant(1.0, dtype=tf.float32)))
        y_s = tf.Variable(tf.one_hot(tf.constant(c, shape=(num_synth,), dtype=tf.int32), 10), dtype=tf.float32)

        # Compress
        ds_iter = ds.as_numpy_iterator()
        for k in range(self.K):
            # Reinitialize model
            utils.reinitialize_model(self.mdl)
            for t in range(self.T):
                x_ds, y_ds = next(ds_iter)
                # Perform distillation step
                for i in range(self.I):
                    dist_loss = self.distill_step(x_ds, y_ds, x_s, y_s)
                # Perform training step
                x_t, y_t = buf.sample(self.batch_size)
                if x_t is not None:
                    x_comb = tf.concat((x_ds, x_t), axis=0)
                    y_comb = tf.concat((y_ds, y_t), axis=0)
                else:
                    x_comb = x_ds
                    y_comb = y_ds
                train_loss = self.train_step(x_comb, y_comb, self.mdl, self.train_opt)
            if verbose:
                print("Iter: {} Dist loss: {:.3} Train loss: {:.3}".format(k, dist_loss, train_loss))
        return x_s, y_s


class AbstractBuffer(abc.ABC):
    """
    Abstract base class for buffers
    """

    def __init__(self, max_buffer_size=1000):
        self.x_buffer = None
        self.y_buffer = None
        self.max_buffer_size = max_buffer_size
        self.samples_seen = 0
        super(AbstractBuffer, self).__init__()

    def add_samples(self, x, y):
        # Check if buffer is empty
        if self.x_buffer is None:
            if self.max_buffer_size >= x.shape[0]:
                self.x_buffer = np.copy(x)
                self.y_buffer = np.copy(y)
            else:
                self.x_buffer = np.copy(x[0:self.max_buffer_size])
                self.y_buffer = np.copy(y[0:self.max_buffer_size])
        else:
            # Check how many samples can be added to buffer
            add_samples = self.max_buffer_size - self.x_buffer.shape[0]
            if add_samples >= x.shape[0]:
                self.x_buffer = np.concatenate((self.x_buffer, np.copy(x)), axis=0)
                self.y_buffer = np.concatenate((self.y_buffer, np.copy(y)), axis=0)
            else:
                self.x_buffer = np.concatenate((self.x_buffer, np.copy(x[0:add_samples])), axis=0)
                self.y_buffer = np.concatenate((self.y_buffer, np.copy(y[0:add_samples])), axis=0)
        self.samples_seen += x.shape[0]

    def is_full(self):
        # Check if buffer is full or not
        if self.x_buffer is not None:
            if self.x_buffer.shape[0] == self.max_buffer_size:
                return True
            else:
                return False
        else:
            return False

    @abc.abstractmethod
    def update_buffer(self, x, y):
        pass

    def summary(self):
        print("+======================================+")
        print("| Summary                              |")
        print("+======================================+")
        print("| Number of samples in memory: {}".format(self.x_buffer.shape[0]))
        print("+--------------------------------------+")
        cl, counts = np.unique(np.argmax(self.y_buffer, axis=-1), return_counts=True)
        for i, j in zip(cl, counts):
            print("| Class {}: {}".format(i, j))
        print("+--------------------------------------+")

    def sample(self, k):
        # Randomly select and return k examples with their labels from the buffer
        if self.x_buffer is not None:
            sel_idx = np.random.choice(np.arange(self.x_buffer.shape[0]), k)
            data = self.x_buffer[sel_idx]
            labels = self.y_buffer[sel_idx]
            return data, labels
        else:
            return None, None

    def free_space(self, new_classes=2):
        # Free buffer space and keep examples per class
        x_buffer = None
        y_buffer = None
        # Get classes and number of examples per class
        cl, counts = np.unique(np.argmax(self.y_buffer, axis=-1), return_counts=True)
        idx = np.arange(self.x_buffer.shape[0])
        for c in cl:
            # Randomly select the examples to keep
            num_examples = int(np.asarray(self.x_buffer.shape[0], dtype=np.float32)/np.asarray(len(cl)+new_classes, dtype=np.float32))
            sel_idx = np.random.choice(idx[np.argmax(self.y_buffer, axis=-1) == c], num_examples, replace=False)
            # Build new buffer
            if x_buffer is None:
                x_buffer = self.x_buffer[sel_idx]
                y_buffer = self.y_buffer[sel_idx]
            else:
                x_buffer = np.concatenate((x_buffer, self.x_buffer[sel_idx]), axis=0)
                y_buffer = np.concatenate((y_buffer, self.y_buffer[sel_idx]), axis=0)
        self.x_buffer = x_buffer
        self.y_buffer = y_buffer


class BalancedBuffer(AbstractBuffer):
    """
    Buffer that always replaces examples from the majority class at random
    """

    def __init__(self, max_buffer_size=1000):
        super(BalancedBuffer, self).__init__(max_buffer_size)

    def update_buffer(self, x, y):
        for i in range(x.shape[0]):
            # Estimate entropy of label distribution in the buffer
            classes, counts = np.unique(self.y_buffer, return_counts=True)
            majority_class = classes[np.argmax(counts)]
            # Compute minimum distance of all examples from the majority class to every other example from this class
            majority_idx = np.arange(0, self.x_buffer.shape[0])[self.y_buffer == majority_class]
            # Randomly select a sample of the majority class to be replaced
            repl_idx = np.random.choice(majority_idx, 1)
            self.x_buffer[repl_idx] = x[i]
            self.y_buffer[repl_idx] = y[i]


class CompressedBalancedBuffer(BalancedBuffer):
    """
    Buffer that adds a compression to the balanced buffer
    """

    def __init__(self, max_buffer_size=1000):
        super(CompressedBalancedBuffer, self).__init__(max_buffer_size)

    def compress_add(self, ds, c, batch_size, train_learning_rate, dist_learning_rate, img_shape, num_synth, K, T, mdl, verbose=False):
        # Create compressor
        comp = DataCompressor(batch_size, train_learning_rate, dist_learning_rate, K, T, mdl)
        # Compress data
        print("Compressing class {} down to {} samples...".format(c, num_synth))
        x_c, y_c = comp.compress(ds, c, img_shape, num_synth, self, verbose=False)
        # Add compressed data to buffer
        if self.x_buffer is not None:
            self.x_buffer = np.concatenate((self.x_buffer, x_c.numpy()), axis=0)
            self.y_buffer = np.concatenate((self.y_buffer, y_c.numpy()), axis=0)
        else:
            self.x_buffer = x_c.numpy()
            self.y_buffer = y_c.numpy()


class CompositionalCompressor(DataCompressor):
    """
    Compresses data into a smaller set of synthetic examples.
    """

    def __init__(self, batch_size, train_learning_rate, dist_learning_rate, K, T, mdl, I=10, sigmoid_grad=False, 
                 sigmoid_comp=True, sigmoid_input=False, sigmoid_logits=False, tanh_logits=False):
        super(CompositionalCompressor, self).__init__(batch_size, train_learning_rate, dist_learning_rate, K, T, mdl, I)
        self.sigmoid_grad = sigmoid_grad
        self.sigmoid_comp = sigmoid_comp
        self.sigmoid_input = sigmoid_input
        self.sigmoid_logits = sigmoid_logits
        self.tanh_logits = tanh_logits
        self.gradients = None
        self.grads_conv3 = None

    @tf.function
    def distill_step(self, x, y, c_s, w_s, y_s):
        # Minimize cosine similarity between gradients
        with tf.GradientTape() as inner_tape:
            logits_x = self.mdl(x, training=False)
            if self.sigmoid_logits:
                logits_x = adjusted_sigmoid(logits_x)
            loss_x = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, logits_x, from_logits=True))
        grads = inner_tape.gradient(loss_x, self.mdl.trainable_variables)
        # wandb.log({"unsigmoided grads":wandb.Histogram(tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0), num_bins=512)})
        if self.sigmoid_grad:
            for i in range(len(grads)):
                # grads[i] = adjusted_sigmoid(grads[i])
                grads[i] = adjusted_tanh(grads[i])
            # wandb.log({"sigmoided grads":wandb.Histogram(tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0), num_bins=512)})
        with tf.GradientTape() as tape:
            # Make prediction using model
            with tf.GradientTape() as inner_tape:
                # comp = tf.nn.relu(tf.reduce_sum(tf.multiply(w_s, tf.expand_dims(c_s, axis=0)), axis=1))
                if self.sigmoid_comp:
                    comp = comp_sigmoid(tf.reduce_sum(tf.multiply(w_s, tf.expand_dims(c_s, axis=0)), axis=1))
                    # comp = translated_sigmoid(tf.reduce_sum(tf.multiply(w_s, tf.expand_dims(c_s, axis=0)), axis=1))
                    
                else:
                    comp = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(w_s, tf.expand_dims(c_s, axis=0)), axis=1))
                    # wandb.log({"sigmoided comp":wandb.Histogram(comp, num_bins=512)})
                    comp = tf.reduce_sum(tf.multiply(w_s, tf.expand_dims(c_s, axis=0)), axis=1)
                    # wandb.log({"unscaled comp":wandb.Histogram(comp, num_bins=512)})
                    comp = (comp - tf.math.reduce_min(comp)) / (tf.math.reduce_max(comp) - tf.math.reduce_min(comp))
                    # wandb.log({"scaled comp":wandb.Histogram(comp, num_bins=512)})
                logits_s = self.mdl(comp, training=False)
                if self.sigmoid_logits:
                    loss_s_1 = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_s, logits_s, from_logits=True))
                    logits_s = adjusted_sigmoid(logits_s)
                elif self.tanh_logits:
                    loss_s_1 = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_s, logits_s, from_logits=True))
                    logits_s = tf.math.tanh(logits_s)
                loss_s = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_s, logits_s, from_logits=True))
            grads_s = inner_tape.gradient(loss_s, self.mdl.trainable_variables)
            # wandb.log({"unsigmoided grads_s":wandb.Histogram(tf.concat([tf.reshape(g, [-1]) for g in grads_s], axis=0), num_bins=512),})
            if self.sigmoid_grad:
                for i in range(len(grads_s)):
                    # grads_s[i] = adjusted_sigmoid(grads_s[i])
                    grads_s[i] = adjusted_tanh(grads_s[i])
                # wandb.log({"sigmoided grads_s":wandb.Histogram(tf.concat([tf.reshape(g, [-1]) for g in grads_s], axis=0), num_bins=512)})
            # Compute cosine similarity
            dist_loss = tf.constant(0.0, dtype=tf.float32)
            for g, gs in zip(grads, grads_s):
                if len(g.shape) == 2:
                    g_norm = tf.math.l2_normalize(g, axis=0)
                    gs_norm = tf.math.l2_normalize(gs, axis=0)
                    inner = tf.reduce_sum(tf.multiply(g_norm, gs_norm), axis=0)
                if len(g.shape) == 4:
                    g_norm = tf.math.l2_normalize(g, axis=(0, 1, 2))
                    gs_norm = tf.math.l2_normalize(gs, axis=(0, 1, 2))
                    inner = tf.reduce_sum(tf.multiply(g_norm, gs_norm), axis=(0, 1, 2))
                dist_loss += tf.reduce_sum(tf.subtract(tf.constant(1.0, dtype=tf.float32), inner))
        dist_grads = tape.gradient(dist_loss, [c_s, w_s])
        self.dist_opt.apply_gradients(zip(dist_grads, [c_s, w_s]))

        if self.sigmoid_logits:
            return dist_loss, grads[-2], g_norm, grads[8], tf.math.l2_normalize(grads[8], axis=0), loss_s_1, loss_s
        else:
            return dist_loss, grads[-2], g_norm, grads[8], tf.math.l2_normalize(grads[8], axis=0), loss_s, None

    def compress(self, ds, c, img_shape, num_synth, k, buf=None, log_histogram=False, verbose=False):
        # Create and initialize synthetic data
        # num_synth = num_weight = 2 * int(BUFFER_SIZE / len(CLASSES)) = 20
        # k = num_components = int(BUFFER_SIZE / len(CLASSES)) = 10
        c_s = tf.Variable(tf.random.uniform((k, img_shape[0], img_shape[1], img_shape[2]), maxval=tf.constant(1.0, dtype=tf.float32)))
        y_s = tf.Variable(tf.one_hot(tf.constant(c, shape=(num_synth,), dtype=tf.int32), 10), dtype=tf.float32)
        w_s = tf.Variable(tf.random.normal((num_synth, k, 1, 1, 1), dtype=tf.float32))

        # Preparation for log
        starting_step = self.K * self.T * self.I * c
        distill_step = 0
        update_step = 0

        # Compress
        ds_iter = ds.as_numpy_iterator()
        for k in tqdm(range(self.K)):
            # Reinitialize model
            utils.reinitialize_model(self.mdl)
            for t in range(self.T):
                # y_ds = next(ds_iter)
                x_ds, y_ds = next(ds_iter)
                # if t == 0:
                #     wandb.log({"Pixel Distribution/class {}/Input images".format(c):wandb.Histogram(x_ds, num_bins=512)})
                if self.sigmoid_input:
                    x_ds = tf.math.sigmoid(x_ds)
                    # wandb.log({"sigmoided input":wandb.Histogram(x_ds, num_bins=512)})
                # Perform distillation step
                for i in range(self.I): # one batch of dataset distills the components I iterations
                    dist_loss, grads_dense, grads_dense_norm, grads_conv3, grads_conv3_norm, loss_1, loss = self.distill_step(x_ds, y_ds, c_s, w_s, y_s)
                    distill_step += 1
                wandb.log({"Distill/Matching Loss": dist_loss, 'Distill_step': starting_step + distill_step})
                self.gradients = grads_dense
                self.grads_conv3 = grads_conv3
                
                # Perform training step
                x_t, y_t = buf.sample(self.batch_size)
                if x_t is not None:
                    x_comb = tf.concat((x_ds, x_t), axis=0)
                    y_comb = tf.concat((y_ds, y_t), axis=0)
                else:
                    x_comb = x_ds # Compress at first, then train with real data or real+syn data.
                    y_comb = y_ds # batch size of real data and synthetic data are both 256
                train_loss = self.train_step(x_comb, y_comb, self.mdl, self.train_opt)
                loss_name = 'InnerLoop/Class ' + str(c)
                wandb.log({loss_name: train_loss, 'update_step_'+ str(c): update_step})
                update_step += 1
                # Train T iters. However, when T=1, this train doesn't contributes to the algorithms.
                # Training is still necessary for the verbose.
            wandb.log({"Distill/categorical loss (unsigmoided logits)": loss_1, 'Distill_step': starting_step + distill_step})
            if self.sigmoid_logits:
                wandb.log({"Distill/categorical loss (sigmoided logits)": loss, 'Distill_step': starting_step + distill_step})
            if verbose:
                print("Iter: {} Dist loss: {:.3} Train loss: {:.3}".format(k, dist_loss, train_loss))
        if log_histogram:
            wandb.log({
                "Pixel Distribution/class {}/composed images".format(c):
                wandb.Histogram(tf.nn.sigmoid(tf.reduce_sum(tf.multiply(w_s, tf.expand_dims(c_s, axis=0)), axis=1)), num_bins=512),
                "Pixel Distribution/class {}/unsigmoided images".format(c):
                wandb.Histogram(tf.reduce_sum(tf.multiply(w_s, tf.expand_dims(c_s, axis=0)), axis=1), num_bins=512),
                "Pixel Distribution/class {}/base images".format(c): wandb.Histogram(c_s, num_bins=512),
                })

            # Histo_input = np.histogram(x_ds, bins=128)
            # unsigmoided_comp = tf.reduce_sum(tf.multiply(w_s, tf.expand_dims(c_s, axis=0)), axis=1)
            # comp = comp_sigmoid(unsigmoided_comp)
            # Histo_comp = np.histogram(comp, bins=128)
            # y_input = Histo_input[0] / np.sum(Histo_input[0])
            # y_comp = Histo_comp[0] / np.sum(Histo_comp[0])
            # bins_comp = ((Histo_comp[1][:-1] + Histo_comp[1][1:]) / 2)

            # plt.figure()
            # plt.bar(bins_comp, y_input, label="input")
            # plt.bar(bins_comp, y_comp, label="comp")
            # plt.ylim(0)
            # plt.legend()
            # wandb.log({"Pixel Distribution/class {}/Input and composed images - class {}".format(c,c): plt})

            for i in range(grads_dense.numpy().shape[1]):
                if i == c:
                    wandb.log({"Pixel Distribution/Gradients of class logits".format(c):wandb.Histogram(grads_dense.numpy()[:,i], num_bins=512),
                            'Gradients_step': 20 * c + i})
                else:
                    wandb.log({"Pixel Distribution/Gradients of non-class logits".format(c):wandb.Histogram(grads_dense.numpy()[:,i], num_bins=512),
                            'Gradients_step': 20 * c + i})
                wandb.log({"Pixel Distribution/Norm gradients of dense layer by dimension1".format(c):wandb.Histogram(grads_dense_norm.numpy()[:,i], num_bins=512),
                           'Norm_gradients_step': 20 * c + i})

        return c_s, w_s, y_s


class CompositionalBalancedBuffer(object):
    """
    Buffer that adds a compression to the balanced buffer
    """

    def __init__(self):
        self.c_buffer = []
        self.w_buffer = []
        self.y_buffer = []
        self.buffer_box = {}
        
        self.image_params_count_exp = 0
        self.weight_params_count_exp = 0
        self.image_params_count = 0
        self.weight_params_count = 0

        self.syn_images = None # all synthetic images in one tf Tensor
        self.syn_labels = None # all labels for synthetic images in one tf Tensor
        self.dense_layer_gradients = []
        self.conv3_layer_gradients = []

        super(CompositionalBalancedBuffer, self).__init__()

    def compress_add(self, ds, c, mdl, batch_size=128, train_learning_rate=0.01, dist_learning_rate=0.05, 
                     img_shape=(28, 28, 1), num_bases=10, K=20, T=10, I=10, log_histogram=False, verbose=False,
                     sigmoid_grad=False, sigmoid_comp=True, sigmoid_input=False, sigmoid_logits=False, tanh_logits=False):
        # Create compressor
        comp = CompositionalCompressor(batch_size, train_learning_rate, dist_learning_rate, K, T, mdl, I=I,
                                       sigmoid_grad=sigmoid_grad, sigmoid_comp=sigmoid_comp, sigmoid_input=sigmoid_input,
                                       sigmoid_logits=sigmoid_logits, tanh_logits=tanh_logits)
        self.sigmoid_comp = sigmoid_comp
        # Compress data
        num_weights = int(2*num_bases)
        num_components = int(num_bases)
        print("Compressing class {} down to {} weights and {} components...".format(c, num_weights, num_components))
        c_s, w_s, y_s = comp.compress(ds, c, img_shape, num_weights, num_components, self, log_histogram=log_histogram, 
                                      verbose=False)
        self.dense_layer_gradients.append(comp.gradients.numpy())
        self.conv3_layer_gradients.append(comp.grads_conv3.numpy())
        self.get_storage(c_s, w_s)
        wandb.log({'image_params_count': self.image_params_count,
                   'weight_params_count': self.weight_params_count,
                   'Total_params_count': self.weight_params_count + self.image_params_count})
        
        # Add compressed data to buffer
        self.c_buffer.append(c_s)
        self.w_buffer.append(w_s)
        self.y_buffer.append(y_s)
        
        # buffer_box is build for grid making and logging images
        self.buffer_box[0] = self.c_buffer
        self.buffer_box[1] = self.w_buffer
        self.buffer_box[2] = self.y_buffer

        syn_images_c = self.compose_image(self.c_buffer, self.w_buffer, c) # synthetic images of last class
        if self.syn_images == None:
            self.syn_images = syn_images_c
            self.syn_labels = y_s
        else:
            self.syn_images = tf.concat([self.syn_images, syn_images_c], axis=0)
            self.syn_labels = tf.concat([self.syn_labels, y_s], axis=0)

    @staticmethod
    def compose_image(c_buffer, w_buffer, cl, idx=None):
        if idx is None:
            comp = comp_sigmoid(tf.reduce_sum(tf.multiply(w_buffer[cl], tf.expand_dims(c_buffer[cl], axis=0)), axis=1))
        else:
            comp = comp_sigmoid(tf.reduce_sum(tf.multiply(w_buffer[cl][idx], tf.expand_dims(c_buffer[cl], axis=0)), axis=1))
        return comp

    def sample_old(self, k):
        # Randomly select and return k examples with their labels from the buffer
        num_classes = len(self.w_buffer)
        if num_classes > 0:
            data = np.zeros((k, tf.shape(self.c_buffer[0])[1], tf.shape(self.c_buffer[0])[2], tf.shape(self.c_buffer[0])[3]), dtype=np.single)
            labels = np.zeros((k, tf.shape(self.y_buffer[0])[1]), dtype=np.single)
            for i in range(k):
                # Sample class
                cl = np.squeeze(np.random.randint(0, num_classes, 1))
                # Sample instance
                idx = np.squeeze(np.random.randint(0, tf.shape(self.w_buffer[cl])[0]))
                # Compose image
                comp = self.compose_image(self.c_buffer, self.w_buffer, cl, idx)
                # comp = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(self.w_buffer[cl][idx], tf.expand_dims(self.c_buffer[cl], axis=0)), axis=1))
                data[i] = comp
                labels[i] = self.y_buffer[cl][idx]
            return data, labels
        else:
            return None, None
    
    def sample(self, batch_size):
        # Randomly select and return k examples with their labels from the buffer
        num_classes = len(self.w_buffer)
        if num_classes > 0:
            data = np.zeros((batch_size, tf.shape(self.c_buffer[0])[1], tf.shape(self.c_buffer[0])[2], tf.shape(self.c_buffer[0])[3]), dtype=np.single)
            labels = np.zeros((batch_size, tf.shape(self.y_buffer[0])[1]), dtype=np.single)
            
            indices = np.random.permutation(tf.shape(self.w_buffer[0])[0].numpy() * tf.shape(self.c_buffer[0])[0].numpy() * 10) # Max number of self.syn_images
            indices = list(indices[:batch_size] % tf.shape(self.syn_images)[0,].numpy())
            data = tf.gather(self.syn_images, indices).numpy()
            labels = tf.gather(self.syn_labels, indices).numpy()

            return data, labels
        else:
            return None, None

    def get_storage(self, c_s, w_s):
        self.image_params_count_exp = keras.backend.count_params(c_s)
        self.weight_params_count_exp = keras.backend.count_params(w_s)
        self.image_params_count += self.image_params_count_exp
        self.weight_params_count += self.weight_params_count_exp

    def summary(self):
        print("+======================================+")
        print("| Summary                              |")
        print("+======================================+")
        for i in range(len(self.w_buffer)):
            print("| Class {}: {} Instances {} components".format(i, 
                                                                  tf.shape(self.w_buffer[i])[0], 
                                                                  tf.shape(self.w_buffer[i])[1]))
        print("+--------------------------------------+")