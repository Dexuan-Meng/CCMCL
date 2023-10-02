"""
This file contains models.
"""

import tensorflow as tf
import tensorflow_addons as tfa
import keras
import abc
import numpy as np
import wandb
from tqdm import tqdm
import utils
from utils import get_batch_size, sample_batch

    
class CNN(tf.keras.Model):
    """
    Simple and small CNN.
    """

    def __init__(self, n):
        super(CNN, self).__init__()
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
        super(CNN, self).build(input_shape)
        
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
    
    def model(self):
        x = tf.keras.Input(shape=(28, 28, 1))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


class BiC_CNN(tf.keras.Model):
    """
    CNN with bias correction layer
    """

    def __init__(self, n):
        super(BiC_CNN, self).__init__()
        self.n = n
        self.mdl = CNN(n)
        self.w = None
        self.b = None
        self.task = None
        self.mask = None

    def build(self, input_shape):
        self.mdl.build(input_shape)
        self.w = tf.ones(1, dtype=tf.float32)
        self.b = tf.zeros(1, dtype=tf.float32)
        self.task = 0
        self.update_mask(self.task)
        super(BiC_CNN, self).build(input_shape)

    def update_mask(self, task):
        # Update mask and reset bias correction parameters
        self.task = task
        self.w = tf.Variable(tf.ones(1), dtype=tf.float32)
        self.b = tf.Variable(tf.zeros(1), dtype=tf.float32)
        r = tf.range(0, self.n, dtype=tf.int32)
        lower = tf.math.greater_equal(r, tf.multiply(tf.constant(2, dtype=tf.int32), self.task))
        upper = tf.math.less(r, tf.multiply(tf.constant(2, dtype=tf.int32), tf.add(self.task, tf.constant(1, dtype=tf.int32))))
        self.mask = tf.math.logical_and(lower, upper)
        
    def call(self, inputs, training=None):
        logits = self.mdl(inputs, training)
        output = tf.where(self.mask, tf.add(tf.multiply(self.w, logits), self.b), logits)
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


class BiCBuffer(AbstractBuffer):
    """
    Buffer used for BiC
    """

    def __init__(self, max_buffer_size=1000):
        self.l_buffer = None
        super(BiCBuffer, self).__init__(max_buffer_size)

    def add_samples(self, x, y, l):
        # Check if buffer is empty
        if self.x_buffer is None:
            if self.max_buffer_size >= x.shape[0]:
                self.x_buffer = np.copy(x)
                self.y_buffer = np.copy(y)
                self.l_buffer = np.copy(l)
            else:
                self.x_buffer = np.copy(x[0:self.max_buffer_size])
                self.y_buffer = np.copy(y[0:self.max_buffer_size])
                self.l_buffer = np.copy(l[0:self.max_buffer_size])
        else:
            # Check how many samples can be added to buffer
            add_samples = self.max_buffer_size - self.x_buffer.shape[0]
            if add_samples >= x.shape[0]:
                self.x_buffer = np.concatenate((self.x_buffer, np.copy(x)), axis=0)
                self.y_buffer = np.concatenate((self.y_buffer, np.copy(y)), axis=0)
                self.l_buffer = np.concatenate((self.l_buffer, np.copy(l)), axis=0)
            else:
                self.x_buffer = np.concatenate((self.x_buffer, np.copy(x[0:add_samples])), axis=0)
                self.y_buffer = np.concatenate((self.y_buffer, np.copy(y[0:add_samples])), axis=0)
                self.l_buffer = np.concatenate((self.l_buffer, np.copy(l[0:add_samples])), axis=0)
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

    def update_buffer(self, x, y):
        pass

    def sample(self, k):
        # Randomly select and return k examples with their labels from the buffer
        if self.x_buffer is not None:
            sel_idx = np.random.choice(np.arange(self.x_buffer.shape[0]), k)
            data = self.x_buffer[sel_idx]
            labels = self.y_buffer[sel_idx]
            logits = self.l_buffer[sel_idx]
            return data, labels, logits
        else:
            return None, None

    def free_space(self, new_classes=2):
        # Free buffer space and keep examples per class
        x_buffer = None
        y_buffer = None
        l_buffer = None
        # Get classes and number of examples per class
        cl, counts = np.unique(np.argmax(self.y_buffer, axis=-1), return_counts=True)
        idx = np.arange(self.x_buffer.shape[0])
        for c in cl:
            # Randomly select the examples to keep
            num_examples = int(np.asarray(self.x_buffer.shape[0], dtype=np.float32)/np.asarray(len(cl)+new_classes, dtype=np.float32))
            sel_idx = np.random.choice(idx[np.argmax(self.y_buffer, axis=-1) == c], num_examples)
            # Build new buffer
            if x_buffer is None:
                x_buffer = self.x_buffer[sel_idx]
                y_buffer = self.y_buffer[sel_idx]
                l_buffer = self.l_buffer[sel_idx]
            else:
                x_buffer = np.concatenate((x_buffer, self.x_buffer[sel_idx]), axis=0)
                y_buffer = np.concatenate((y_buffer, self.y_buffer[sel_idx]), axis=0)
                l_buffer = np.concatenate((l_buffer, self.l_buffer[sel_idx]), axis=0)
        self.x_buffer = x_buffer
        self.y_buffer = y_buffer
        self.l_buffer = l_buffer


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

    def __init__(self, batch_size, train_learning_rate, dist_learning_rate, K, T, mdl, I=10):
        super(CompositionalCompressor, self).__init__(batch_size, train_learning_rate, dist_learning_rate, K, T, mdl, I)

    @tf.function
    def distill_step(self, x, y, c_s, w_s, y_s):
        # Minimize cosine similarity between gradients
        with tf.GradientTape() as inner_tape:
            logits_x = self.mdl(x, training=False)
            loss_x = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, logits_x, from_logits=True))
        grads = inner_tape.gradient(loss_x, self.mdl.trainable_variables)
        with tf.GradientTape() as tape:
            # Make prediction using model
            with tf.GradientTape() as inner_tape:
                # comp = tf.nn.relu(tf.reduce_sum(tf.multiply(w_s, tf.expand_dims(c_s, axis=0)), axis=1))
                comp = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(w_s, tf.expand_dims(c_s, axis=0)), axis=1))
                logits_s = self.mdl(comp, training=False)
                loss_s = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_s, logits_s, from_logits=True))
            grads_s = inner_tape.gradient(loss_s, self.mdl.trainable_variables)
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
        return dist_loss

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

                # Perform distillation step
                for i in range(self.I): # one batch of dataset distills the components I iterations
                    dist_loss = self.distill_step(x_ds, y_ds, c_s, w_s, y_s)
                    wandb.log({"Distill/Matching Loss": dist_loss, 'Distill_step': starting_step + distill_step})
                    if log_histogram:
                        wandb.log({
                            "Distill/class {}/Synthetic_Pixels".format(c):
                            wandb.Histogram(tf.nn.sigmoid(tf.reduce_sum(tf.multiply(w_s, tf.expand_dims(c_s, axis=0)), axis=1)), num_bins=512),
                            "Distill/class {}/Base_Pixels".format(c): wandb.Histogram(c_s, num_bins=512)
                            })
                    distill_step += 1
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
            if verbose:
                print("Iter: {} Dist loss: {:.3} Train loss: {:.3}".format(k, dist_loss, train_loss))
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

        super(CompositionalBalancedBuffer, self).__init__()

    def compress_add(self, ds, c, mdl, batch_size=128, train_learning_rate=0.01, dist_learning_rate=0.05, 
                     img_shape=(28, 28, 1), num_synth=10, K=20, T=10, I=10, log_histogram=False, verbose=False):
        # Create compressor
        comp = CompositionalCompressor(batch_size, train_learning_rate, dist_learning_rate, K, T, mdl, I=I)
        # Compress data
        num_weights = int(2*num_synth)
        num_components = int(num_synth)
        print("Compressing class {} down to {} weights and {} components...".format(c, num_weights, num_components))
        c_s, w_s, y_s = comp.compress(ds, c, img_shape, num_weights, num_components, self, log_histogram=log_histogram, 
                                      verbose=False)
        
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

    @staticmethod
    def compose_image(c_buffer, w_buffer, cl, idx=None):
        if idx is None:
            comp = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(w_buffer[cl], tf.expand_dims(c_buffer[cl], axis=0)), axis=1))
        else:
            comp = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(w_buffer[cl][idx], tf.expand_dims(c_buffer[cl], axis=0)), axis=1))
        return comp

    def sample(self, k):
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
            print("| Class {}: {} Instances {} components".format(i, tf.shape(self.w_buffer[i])[0], tf.shape(self.w_buffer[i])[1]))
        print("+--------------------------------------+")


class FactorizationCompressor(DataCompressor):
    """
    Compresses data into a smaller set of synthetic examples.
    """

    def __init__(self, class_label, batch_size, train_learning_rate, img_learning_rate, styler_learning_rate, K, T, mdl, 
                 I=10, img_shape=(28, 28, 1), lambda_club_content= 10, lambda_cls_content = 1, lambda_likeli_content=1,
                 lambda_contrast_content=1):
        super(FactorizationCompressor, self).__init__(batch_size, train_learning_rate, img_learning_rate, K, T, mdl, I=I)
        self.class_label = class_label
        self.img_size = img_shape[0]
        self.dist_opt = tf.keras.optimizers.RMSprop(img_learning_rate)
        self.net_opt = tf.keras.optimizers.SGD(train_learning_rate)
        self.extractor_opt = tf.keras.optimizers.SGD(5 * train_learning_rate)
        self.styler_opt = tf.keras.optimizers.SGD(styler_learning_rate)
        self.extractor = Extractor(10, channel=img_shape[2], image_size=img_shape[0])
        self.cosine_similarity = tf.keras.losses.CosineSimilarity()
        self.contrastive_loss = utils.SupConLoss()
        self.lambda_club_content = lambda_club_content
        self.lambda_cls_content = lambda_cls_content
        self.lambda_likeli_content = lambda_likeli_content
        self.lambda_contrast_content = lambda_contrast_content

        self.base_image = None
        self.syn_label = None
        self.stylers = []

    @tf.function
    def matching_loss(self, x, y, base_image, syn_label):
        # Minimize cosine similarity between gradients
        with tf.GradientTape() as inner_tape:
            logits_x = self.mdl(x, training=False)
            loss_x = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, logits_x, from_logits=True))
        grads = inner_tape.gradient(loss_x, self.mdl.trainable_variables)

        # Make prediction using model
        with tf.GradientTape() as inner_tape:
            comp = self.compose_image(base_image, self.stylers)
            # comp = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(w_s, tf.expand_dims(self.base_image, axis=0)), axis=1))
            logits_s = self.mdl(comp, training=False)
            loss_s = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(syn_label, logits_s, from_logits=True))
        grads_s = inner_tape.gradient(loss_s, self.mdl.trainable_variables)

        # Compute matching loss
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

        return dist_loss

    @tf.function
    def distill_step(self, x, y):

        with tf.GradientTape() as tape_w_s:
            with tf.GradientTape() as tape_c_s:

                dist_loss = self.matching_loss(x, y, self.base_image, self.syn_label)
                # compute cosine similarity
                twin_image = self.get_twin_image()
                embed_c, _ = self.extractor(twin_image)
                club_content_loss = tf.reduce_mean(((self.cosine_similarity(embed_c[0], embed_c[1]) + 1.) / 2.))

                loss = dist_loss + self.lambda_club_content * club_content_loss

        stylers_trainable_variables = []
        for i in range(len(self.stylers)):
            stylers_trainable_variables.extend(self.stylers[i].trainable_variables)
        dist_grad_w_s = tape_c_s.gradient(loss, stylers_trainable_variables)
        dist_grad_c_s = tape_w_s.gradient(loss, [self.base_image])
        self.styler_opt.apply_gradients(zip(dist_grad_w_s, stylers_trainable_variables))
        self.dist_opt.apply_gradients(zip(dist_grad_c_s, [self.base_image]))

        return dist_loss, club_content_loss, loss

    @tf.function
    def update_extractor(self, x, y):
        # compute contrastive loss and update extractor
        sim_content_loss = 0
        with tf.GradientTape() as tape_extractor:
            
            _, ds_logits = self.extractor(x)
            cls_content_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, ds_logits, from_logits=True))

            twin_image = self.get_twin_image()
            embed_c, _ = self.extractor(twin_image)
            likeli_content_loss = tf.reduce_mean(((1. - self.cosine_similarity(embed_c[:tf.shape(self.base_image)[0]], embed_c[tf.shape(self.base_image)[0]:])) / 2.))
            embed_c_0, _ = tf.linalg.normalize(embed_c[:tf.shape(self.base_image)[0]])
            embed_c_1, _ = tf.linalg.normalize(embed_c[tf.shape(self.base_image)[0]:])
            contrast_content_loss = self.contrastive_loss(
                tf.stack([embed_c_0, embed_c_1], axis=1), 
                tf.argmax(self.syn_label, axis=1)[:tf.shape(self.base_image)[0]]
                )
            
            sim_content_loss = sim_content_loss + cls_content_loss * self.lambda_cls_content \
                                                + likeli_content_loss * self.lambda_likeli_content \
                                                + contrast_content_loss * self.lambda_contrast_content

        dist_grad_extractor = tape_extractor.gradient(sim_content_loss, self.extractor.trainable_variables)
        self.extractor_opt.apply_gradients(zip(dist_grad_extractor, self.extractor.trainable_variables))
        
        return cls_content_loss, likeli_content_loss, contrast_content_loss, sim_content_loss

    @staticmethod
    def compose_image(base_image, stylers):
        '''
        Concatenate synthetic images hallucinated by all stylers.
        '''
        syn_images = []
        for i in range(len(stylers)):
            syn_images.append(stylers[i](base_image))
        comp = tf.concat(syn_images, axis=0)
        return comp

    @tf.function
    def get_twin_image(self):
        indices = np.random.permutation(len(self.stylers))
        twin_image = tf.concat((self.stylers[indices[0]](self.base_image), self.stylers[indices[1]](self.base_image)), axis=0)
        return twin_image
    
    def compress(self, ds, c, img_shape, num_stylers, num_base, buf=None, verbose=False):

        # Create and initialize synthetic data
        # k = num_components = int(BUFFER_SIZE / len(CLASSES)) = 10
        self.base_image = tf.Variable(tf.random.uniform((num_base, img_shape[0], img_shape[1], img_shape[2]), maxval=tf.constant(1.0, dtype=tf.float32)))
        self.syn_label = tf.Variable(tf.one_hot(tf.constant(c, shape=(num_stylers * num_base,), dtype=tf.int32), 10), dtype=tf.float32)
        for _ in range(num_stylers):
            styler = StyleTranslator(in_channel=img_shape[2], mid_channel=3, out_channel=img_shape[2], image_size=img_shape, kernel_size=3)
            styler.build((None, img_shape[0], img_shape[1], img_shape[2]))
            self.stylers.append(styler)

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
                x_ds, y_ds = next(ds_iter)
                # Perform distillation step
                for i in range(self.I): # one batch of dataset distills the components I iterations
                    dist_loss, club_content_loss, loss = self.distill_step(x_ds, y_ds)
                    wandb.log({"Distill/class {}/Matching_loss".format(self.class_label): dist_loss,
                            "Distill/class {}/Club_content_loss".format(self.class_label): club_content_loss,
                            "Distill/class {}/Grand_loss".format(self.class_label): loss})
                    wandb.log({"Distill/Matching Loss": dist_loss, 'Distill_step': starting_step + distill_step})
                    cls_content_loss, likeli_content_loss, contrast_content_loss, sim_content_loss = self.update_extractor(x_ds, y_ds)
                    wandb.log({"Distill/class {}/Cls_content_loss".format(self.class_label): cls_content_loss,
                               "Distill/class {}/Likeli_content_loss".format(self.class_label): likeli_content_loss,
                               "Distill/class {}/Contrast_content_loss".format(self.class_label): contrast_content_loss,
                               "Distill/class {}/Sim_content_loss".format(self.class_label): sim_content_loss})
                    
                    distill_step += 1
                # Perform innerloop training step
                x_t, y_t = buf.sample(self.batch_size)
                if x_t is not None:
                    x_comb = tf.concat((x_ds, x_t), axis=0)
                    y_comb = tf.concat((y_ds, y_t), axis=0)
                else:
                    x_comb = x_ds # Compress at first, then train with real data or real+syn data.
                    y_comb = y_ds # batch size of real data and synthetic data are both 256
                
                innerloop = 1 if c==0 else c
                for _ in range(innerloop):
                    train_loss = self.train_step(x_comb, y_comb, self.mdl, self.train_opt)
                loss_name = 'InnerLoop/Class ' + str(c)
                wandb.log({loss_name: train_loss, 'update_step_'+ str(c): update_step})
                update_step += 1
            if verbose:
                print("Iter: {} Dist loss: {:.3} Train loss: {:.3}".format(k, dist_loss, train_loss))
        return self.base_image, self.stylers, self.syn_label


class FactorizationBalancedBuffer(CompositionalBalancedBuffer):
    """
    Buffer that adds a compression to the balanced buffer
    """

    def __init__(self):
        self.base_buffer = []
        self.styler_buffer = []
        self.label_buffer = []
        self.buffer_box = {}
        self.syn_images_buffer = {} # synthetic images stored in class
        self.syn_images = None # all synthetic images in one tf Tensor
        self.syn_labels = None # all labels for synthetic images in one tf Tensor

    def compress_add(self, ds, c, mdl, verbose=False, batch_size=128, train_learning_rate=0.01,
                     img_learning_rate=0.01, styler_learning_rate=0.01, img_shape=(28, 28, 1), 
                     num_bases=10, K=10, T=10, I=10, lambda_club_content=10, lambda_cls_content = 1, 
                     lambda_likeli_content=1, lambda_contrast_content=1):
        # Create compressor
        fac = FactorizationCompressor(c, batch_size, train_learning_rate, img_learning_rate, styler_learning_rate,
                                      K, T, mdl, I=I, img_shape=img_shape, lambda_club_content=lambda_club_content, 
                                      lambda_cls_content = lambda_cls_content, lambda_likeli_content=lambda_likeli_content,
                                      lambda_contrast_content=lambda_contrast_content)
        # Compress data
        num_stylers = 2
        print("Compressing class {} down to {} stylers and {} base images...".format(c, num_stylers, num_bases))
        b_s, s_s, y_s = fac.compress(ds, c, img_shape, num_stylers, num_bases, self, verbose=False)

        # Add compressed data to buffer
        self.base_buffer.append(b_s)
        self.styler_buffer.append(s_s)
        self.label_buffer.append(y_s)
        
        syn_images_c = self.compose_image(self.base_buffer, self.styler_buffer, c) # synthetic images of last class
        if self.syn_images == None:
            self.syn_images = syn_images_c
            self.syn_labels = y_s
        else:
            self.syn_images = tf.concat([self.syn_images, syn_images_c], axis=0)
            self.syn_labels = tf.concat([self.syn_labels, y_s], axis=0)
        
        self.buffer_box[0] = self.base_buffer
        self.buffer_box[1] = self.styler_buffer
        self.buffer_box[2] = self.label_buffer

    @staticmethod
    def compose_image(base_buffer, styler_buffer, cl, idx_base_image=None, idx_styler=None):
        # to be modified
        if idx_base_image is None:
            syn_images = []
            for i in range(len(styler_buffer[cl])):
                syn_images.append(styler_buffer[cl][i](base_buffer[cl]))
            comp = tf.concat(syn_images, axis=0)
        else:
            comp = styler_buffer[cl][idx_styler](tf.expand_dims(base_buffer[cl][idx_base_image], axis=0))
        return comp

    def sample(self, batch_size):
        # Randomly select and return k examples with their labels from the buffer
        num_classes = len(self.base_buffer)
        if num_classes > 0:
            data = np.zeros((batch_size, tf.shape(self.base_buffer[0])[1], tf.shape(self.base_buffer[0])[2], tf.shape(self.base_buffer[0])[3]), dtype=np.single)
            labels = np.zeros((batch_size, tf.shape(self.label_buffer[0])[1]), dtype=np.single)
            
            indices = np.random.permutation(len(self.styler_buffer[0]) * tf.shape(self.base_buffer[0])[0].numpy() * 10) # Max number of self.syn_images
            indices = list(indices[:batch_size] % tf.shape(self.syn_images)[0,].numpy())
            data = tf.gather(self.syn_images, indices).numpy()
            labels = tf.gather(self.syn_labels, indices).numpy()

            return data, labels
        else:
            return None, None

    def summary(self):
        print("+======================================+")
        print("| Summary                              |")
        print("+======================================+")
        for i in range(len(self.styler_buffer)):
            print("| Class {}: {} stylers {} components".format(i, 
                                                                  len(self.styler_buffer[i]), 
                                                                  tf.shape(self.base_buffer[i])[0]
                                                                  )
                )
        print("+--------------------------------------+")


class DualClassesFactorizationCompressor(FactorizationCompressor):
    
    def __init__(self, datasets, class_labels, batch_size, train_learning_rate, img_learning_rate, styler_learning_rate, K, T, mdl, 
                 I=10, IN=1, img_shape=(28, 28, 1), lambda_club_content= 10, lambda_cls_content = 1, lambda_likeli_content=1,
                 lambda_contrast_content=1):
        super(DualClassesFactorizationCompressor, self).__init__(class_labels, batch_size, train_learning_rate, img_learning_rate, 
                                                     styler_learning_rate, K, T, mdl, I, img_shape, lambda_club_content,
                                                     lambda_cls_content, lambda_likeli_content, lambda_contrast_content)
        self.ds_iter = {}
        self.IN = IN
        for c in class_labels:
            self.ds_iter[c] = datasets[c].as_numpy_iterator()
    
    @tf.function
    def distill_step(self, x_ds, y_ds):

        with tf.GradientTape() as tape_w_s:
            with tf.GradientTape() as tape_c_s:

                dist_loss = 0
                for idx in range(len(self.class_label)):
                    l = [idx * self.num_base, (idx + 1) * self.num_base]
                    base_image_c = self.base_image[idx * self.num_base: (idx + 1) * self.num_base]
                    syn_label_c = self.syn_label[self.num_stylers * idx * self.num_base: self.num_stylers * (idx + 1) * self.num_base]
                    dist_loss += self.matching_loss(x_ds[idx], y_ds[idx], base_image_c, syn_label_c) / self.num_stylers

                # compute cosine similarity
                twin_image = self.get_twin_image()
                embed_c, _ = self.extractor(twin_image)
                club_content_loss = tf.reduce_mean(((self.cosine_similarity(embed_c[:2 * self.num_base], embed_c[2 * self.num_base:]) + 1.) / 2.))

                loss = dist_loss + self.lambda_club_content * club_content_loss

        stylers_trainable_variables = []
        for i in range(len(self.stylers)):
            stylers_trainable_variables.extend(self.stylers[i].trainable_variables)
        dist_grad_w_s = tape_c_s.gradient(loss, stylers_trainable_variables)
        dist_grad_c_s = tape_w_s.gradient(loss, [self.base_image])
        self.styler_opt.apply_gradients(zip(dist_grad_w_s, stylers_trainable_variables))
        self.dist_opt.apply_gradients(zip(dist_grad_c_s, [self.base_image]))

        return dist_loss, club_content_loss, loss

    @tf.function
    def update_extractor(self, x, y):
        # compute contrastive loss and update extractor
        sim_content_loss = 0
        with tf.GradientTape() as tape_extractor:

            # get composed images, with 0:20 of class 0, and 21:40 of class 1
            # if simply styler_0(base_image) + styler_1(base_image): 0:20 will be class 0 by styler_0 and class 1 by styler_0
            indices = np.random.permutation(self.num_stylers)
            syn_images = []
            for i in indices:
                for j in range(2):
                    syn_images.append(self.stylers[i](self.base_image[j * self.num_base:(j+1) * self.num_base]))
            # comp = tf.concat(
            #     [syn_images[indices[0]][:self.num_base], syn_images[indices[1]][:self.num_base], syn_images[indices[0]][self.num_base:], syn_images[indices[1]][self.num_base:]], axis=0)
            comp = tf.concat(syn_images, axis=0)
            label = tf.reshape(self.syn_label, (2, self.num_stylers, self.num_base, 10))
            comp_label = tf.concat(
                [label[0, indices[0], :, :], label[1, indices[0], :, :], label[0, indices[1], :, :], label[0, indices[1], :, :]], axis=0)
            seed = np.random.randint(1000)
            _, ds_logits = self.extractor(tf.random.shuffle(comp, seed=seed))
            cls_content_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(
                tf.random.shuffle(comp_label, seed=seed), ds_logits, from_logits=True
                ))

            twin_image = self.get_twin_image()
            embed_c, _ = self.extractor(twin_image)
            likeli_content_loss = tf.reduce_mean(((1. - self.cosine_similarity(embed_c[:tf.shape(self.base_image)[0]], embed_c[tf.shape(self.base_image)[0]:])) / 2.))
            embed_c_0, _ = tf.linalg.normalize(embed_c[:tf.shape(self.base_image)[0]], axis=1)
            embed_c_1, _ = tf.linalg.normalize(embed_c[tf.shape(self.base_image)[0]:], axis=1)
            feature_label = tf.concat(
                [self.syn_label[0:self.num_base], self.syn_label[tf.shape(self.base_image)[0]:tf.shape(self.base_image)[0] + self.num_base]],
                axis=0
            )
            seed = np.random.randint(1000)
            contrast_content_loss = self.contrastive_loss(
                tf.stack([tf.random.shuffle(embed_c_0, seed=seed), tf.random.shuffle(embed_c_1, seed=seed)], axis=1), 
                tf.random.shuffle(tf.argmax(feature_label, axis=1), seed=seed)
                )
            
            sim_content_loss = sim_content_loss + cls_content_loss * self.lambda_cls_content \
                                                + likeli_content_loss * self.lambda_likeli_content \
                                                + contrast_content_loss * self.lambda_contrast_content

        dist_grad_extractor = tape_extractor.gradient(sim_content_loss, self.extractor.trainable_variables)
        self.extractor_opt.apply_gradients(zip(dist_grad_extractor, self.extractor.trainable_variables))
        
        return cls_content_loss, likeli_content_loss, contrast_content_loss, sim_content_loss
    
    @tf.function
    def get_twin_image(self):
        indices = np.random.permutation(len(self.stylers))
        twin_image = []
        for i in indices[:2]:
            for j in range(2): # number of classes in one task
                twin_image.append(self.stylers[indices[i]](self.base_image[j * self.num_base : (j+1) * self.num_base]))
        twin_image = tf.concat(twin_image, axis=0)
        return twin_image
    

    def hallucination(self, by_class_arranged=False):
        syn_images = []
        indices = np.random.permutation(len(self.stylers))
        for i in indices:
            for j in range(2): # number of classes in one task
                syn_images.append(self.stylers[indices[j]](self.base_image[j * self.num_base : (j+1) * self.num_base]))
        syn_images = tf.concat(syn_images, axis=0)
        return syn_images
    
    def compress(self, c, img_shape, num_stylers, num_base, buf=None, log_histogram=False, use_image_being_condensed=False, 
                 current_data_proportion=0, verbose=False):

        # Create and initialize synthetic data
        # num_base = num_components = int(BUFFER_SIZE / len(CLASSES)) = 10
        self.num_base = num_base
        self.num_stylers = num_stylers
        self.base_image = tf.Variable(tf.random.uniform((self.num_base * 2, img_shape[0], img_shape[1], img_shape[2]), maxval=tf.constant(1.0, dtype=tf.float32)))
        classes_labels = [tf.constant(c[0], shape=(self.num_stylers * self.num_base,), dtype=tf.int32), 
                       tf.constant(c[1], shape=(self.num_stylers * self.num_base,), dtype=tf.int32)]
        self.syn_label = tf.Variable(tf.concat([tf.one_hot(classes_labels[0], 10), tf.one_hot(classes_labels[1], 10)], 0), dtype=tf.float32)
        for _ in range(num_stylers):
            styler = StyleTranslator(in_channel=img_shape[2], mid_channel=3, out_channel=img_shape[2], image_size=img_shape, kernel_size=3)
            styler.build((None, img_shape[0], img_shape[1], img_shape[2]), self.num_base)
            self.stylers.append(styler)

        # Preparation for log
        starting_step = int(self.K * self.T * self.I * c[0] * 0.5)
        distill_step = 0
        update_step = 0

        for k in tqdm(range(self.K)):
            # Reinitialize model
            utils.reinitialize_model(self.mdl)
            for t in range(self.T):
                x_ds_c0, y_ds_c0 = next(self.ds_iter[self.class_label[0]])
                x_ds_c1, y_ds_c1 = next(self.ds_iter[self.class_label[1]])
                x_ds = tf.concat([x_ds_c0, x_ds_c1], 0)
                y_ds = tf.concat([y_ds_c0, y_ds_c1], 0)
                # Perform distillation step
                for i in range(self.I): # one batch of dataset distills the components I iterations
                    dist_loss, club_content_loss, loss = self.distill_step([x_ds_c0, x_ds_c1], [y_ds_c0, y_ds_c1])
                    wandb.log({"Distill/class {}/Matching_loss".format(self.class_label): dist_loss,
                            "Distill/class {}/Club_content_loss".format(self.class_label): club_content_loss,
                            "Distill/class {}/Grand_loss".format(self.class_label): loss})
                    if log_histogram:
                        wandb.log({
                            "Distill/class {}/Synthetic_Pixels".format(self.class_label):
                            wandb.Histogram(self.hallucination(), num_bins=512),
                            "Distill/class {}/Base_Pixels".format(self.class_label): wandb.Histogram(self.base_image, num_bins=512)
                            })
                    wandb.log({"Distill/Matching Loss": dist_loss, 'Distill_step': starting_step + distill_step})
                    cls_content_loss, likeli_content_loss, contrast_content_loss, sim_content_loss = self.update_extractor(x_ds, y_ds)
                    wandb.log({"Distill/class {}/Cls_content_loss".format(self.class_label): cls_content_loss,
                               "Distill/class {}/Likeli_content_loss".format(self.class_label): likeli_content_loss,
                               "Distill/class {}/Contrast_content_loss".format(self.class_label): contrast_content_loss,
                               "Distill/class {}/Sim_content_loss".format(self.class_label): sim_content_loss})
                    
                    distill_step += 1

                # Perform innerloop training step
                #################################################################
                if c[0] == 0:
                    batch_size_previous_classes, batch_size_current_classes = 0, 256
                else:
                    batch_size_previous_classes, batch_size_current_classes = get_batch_size(self.batch_size, current_data_proportion, c)
                
                if not use_image_being_condensed:
                    x_current_class, y_current_class = sample_batch(x_ds, y_ds, batch_size_current_classes)
                else:
                    syn_image_being_condensed = []
                    for idx in range(len(c)):
                        for styler in self.stylers:
                            syn_image_being_condensed.append(styler(self.base_image[idx * self.num_base: (idx + 1) * self.num_base]))
                    syn_image_being_condensed = tf.concat(syn_image_being_condensed, axis=0)
                    x_current_class, y_current_class = sample_batch(syn_image_being_condensed, y_ds, batch_size_current_classes)

                if batch_size_previous_classes != 0:
                    x_t, y_t = buf.sample(batch_size_previous_classes)
                    x_comb = tf.concat((x_current_class, x_t), axis=0)
                    y_comb = tf.concat((y_current_class, y_t), axis=0)
                else:
                    x_comb = x_current_class # Compress at first, then train with real data or real+syn data.
                    y_comb = y_current_class # batch size of real data and synthetic data are both 256

                #################################################################

                for _ in range(self.IN):
                    train_loss = self.train_step(x_comb, y_comb, self.mdl, self.train_opt)
                loss_name = 'InnerLoop/Class ' + str(c)
                wandb.log({loss_name: train_loss, 'update_step_'+ str(c): update_step})
                update_step += 1
            if verbose:
                print("Iter: {} Dist loss: {:.3} Train loss: {:.3}".format(k, dist_loss, train_loss))
        return self.base_image, self.stylers, self.syn_label


class DualClassesFactorizationBuffer(FactorizationBalancedBuffer):
    """
    Buffer that adds a compression to the balanced buffer
    Data of two classes are compressed at a time.
    """

    def __init__(self):
        super(DualClassesFactorizationBuffer, self).__init__()
        self.image_params_count_exp = 0
        self.styler_params_count_exp = 0
        self.image_params_count = 0
        self.styler_params_count = 0

    def compress_add(self, ds, c, mdl, verbose=False, num_stylers=2, batch_size=128, train_learning_rate=0.01,
                     img_learning_rate=0.01, styler_learning_rate=0.01, img_shape=(28, 28, 1), 
                     num_bases=10, K=10, T=10, I=10, IN=1, lambda_club_content=10, lambda_cls_content = 1, 
                     lambda_likeli_content=1, lambda_contrast_content=1, log_histogram=False, current_data_proportion=0,
                     use_image_being_condensed=False):
        
        # Create compressor
        fac = DualClassesFactorizationCompressor(ds, c, batch_size, train_learning_rate, img_learning_rate, styler_learning_rate,
                                                 K, T, mdl, I=I, IN=IN, img_shape=img_shape, lambda_club_content=lambda_club_content, 
                                                 lambda_cls_content = lambda_cls_content, lambda_likeli_content=lambda_likeli_content,
                                                 lambda_contrast_content=lambda_contrast_content)
        # Compress data
        self.num_stylers = num_stylers
        print("Compressing class {} down to {} stylers and 2 * {} base images...".format(c, self.num_stylers, num_bases))
        b_s, s_s, y_s = fac.compress(c, img_shape, num_stylers, num_bases, self, log_histogram=log_histogram, 
                                     use_image_being_condensed=use_image_being_condensed, current_data_proportion=current_data_proportion, 
                                     verbose=False)

        self.get_storage(b_s, s_s)
        wandb.log({'image_params_count': self.image_params_count,
                   'styler_params_count': self.styler_params_count,
                   'Total_params_count': self.styler_params_count + self.image_params_count})

        # Add compressed data to buffer
        for i in range(len(c)):
            self.base_buffer.append(b_s[i * num_bases: (i + 1) * num_bases])
            self.styler_buffer.append(s_s)
            self.label_buffer.append(y_s[2 * i * num_bases: 2 * (i + 1) * num_bases])
        
        syn_images_c = self.compose_image(self.base_buffer, self.styler_buffer, c) # synthetic images of last class
        if self.syn_images == None:
            self.syn_images = syn_images_c
            self.syn_labels = y_s
        else:
            self.syn_images = tf.concat([self.syn_images, syn_images_c], axis=0)
            self.syn_labels = tf.concat([self.syn_labels, y_s], axis=0)
        
        self.buffer_box[0] = self.base_buffer
        self.buffer_box[1] = self.styler_buffer
        self.buffer_box[2] = self.label_buffer

    @staticmethod
    def compose_image(base_buffer, styler_buffer, cl, idx_base_image=None, idx_styler=None):
        if isinstance(cl, tuple):
            syn_images = []
            for c in list(cl):
                for i in range(len(styler_buffer[c])):
                    syn_images.append(styler_buffer[c][i](base_buffer[c]))
            comp = tf.concat(syn_images, 0)
        else:
            if idx_base_image is None:
                syn_images = []
                for i in range(len(styler_buffer[cl])):
                    syn_images.append(styler_buffer[cl][i](base_buffer[cl]))
                comp = tf.concat(syn_images, axis=0)
            else:
                comp = styler_buffer[cl][idx_styler](tf.expand_dims(base_buffer[cl][idx_base_image], axis=0))
        return comp
    
    def get_storage(self, b_s, s_s):
        self.image_params_count_exp = keras.backend.count_params(b_s)
        self.styler_params_count_exp = 0
        for styler in s_s:
            self.styler_params_count_exp += keras.utils.layer_utils.count_params(styler.trainable_weights)
        self.image_params_count += self.image_params_count_exp
        self.styler_params_count += self.styler_params_count_exp


# class StyleTranslator(tf.keras.Model):
#     """
#     Single-layer-Conv2d encoder + scaling + translation + Single-layer-ConvTranspose2d decoder
#     """

#     def __init__(self, in_channel=3, mid_channel=3, out_channel=3, image_size=(28, 28, 1), kernel_size=3):
#         super(StyleTranslator, self).__init__()
#         self.in_channel = in_channel
#         self.mid_channel = mid_channel
#         self.out_channel = out_channel
#         self.img_size = image_size
#         self.kernel_size = kernel_size
#         self.enc = None
#         self.scale = None
#         self.shift = None
#         self.dec = None
#         self.norm_0 = None
#         self.norm_1 = None

#     def build(self, input_shape):
#         self.norm_0 = tfa.layers.InstanceNormalization()
#         self.norm_1 = tfa.layers.InstanceNormalization()
#         self.enc = tf.keras.layers.Conv2D(self.mid_channel, self.kernel_size, name='Conv2D')
#         self.transform = TransformLayer(self.img_size, self.kernel_size, self.mid_channel)
#         self.dec = tf.keras.layers.Conv2DTranspose(self.out_channel, self.kernel_size, name='Conv2DTransposed')
#         super(StyleTranslator, self).build(input_shape)
        
#     def call(self, inputs, training=None):
#         output = self.norm_0(inputs)
#         output = self.enc(output)
#         output = self.transform(output)
#         output = self.dec(output)
#         output = self.norm_1(output)
#         output = tf.keras.activations.sigmoid(output)
#         return output
    
#     def model(self):
#         x = tf.keras.Input(shape=(28, 28, 1))
#         return tf.keras.Model(inputs=[x], outputs=self.call(x))


class StyleTranslator(tf.keras.Model):
    """
    Single-layer-Conv2d encoder + scaling + translation + Single-layer-ConvTranspose2d decoder
    !! Cross Hallucination: Hallucinate multiple base images to one synthetic image.
    """

    def __init__(self, in_channel=3, mid_channel=3, out_channel=3, image_size=(28, 28, 1), kernel_size=3):
        super(StyleTranslator, self).__init__()
        self.in_channel = in_channel
        self.mid_channel = mid_channel
        self.out_channel = out_channel
        self.img_size = image_size
        
        self.kernel_size_0 = kernel_size
        # self.kernel_size_1 = kernel_size
        self.enc = None
        self.scale = None
        self.shift = None
        self.dec = None
        self.norm_0 = None
        self.norm_1 = None

    def build(self, input_shape, num_base):
        self.encoder = tf.keras.Sequential()
        self.encoder.add(tfa.layers.InstanceNormalization())
        self.encoder.add(tf.keras.layers.Conv2D(self.mid_channel, self.kernel_size_0, name='Conv2D_0'))
        # self.encoder.add(tf.keras.layers.Conv2D(self.mid_channel, self.kernel_size_1, name='Conv2D_1'))
        # self.enc = tf.keras.layers.Conv2D(self.mid_channel, self.kernel_size, name='Conv2D')
        # self.transform
        self.transform = TransformLayer(self.img_size, self.kernel_size_0, self.mid_channel)
        self.decoder = tf.keras.Sequential()
        # self.decoder.add(tf.keras.layers.Conv2DTranspose(self.out_channel, self.kernel_size_1, name='Conv2DTransposed_1'))
        self.decoder.add(tf.keras.layers.Conv2DTranspose(self.out_channel, self.kernel_size_0, name='Conv2DTransposed_0'))
        self.decoder.add(tfa.layers.InstanceNormalization())
        self.weighted_sum = WeightedSum(num_base)
        # self.dec = tf.keras.layers.Conv2DTranspose(self.out_channel, self.kernel_size, name='Conv2DTransposed')
        super(StyleTranslator, self).build(input_shape)
        
    def call(self, inputs, training=None):

        output = self.encoder(inputs)
        output = self.transform(output)
        output = self.weighted_sum(output)
        output = self.decoder(output)

        output = tf.keras.activations.sigmoid(output)
        return output
    
    def model(self):
        x = tf.keras.Input(shape=(28, 28, 1))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


class Extractor(tf.keras.Model):
    """
    Model to extract feature vector for similarity/divergence measuring
    Almost the same as CNN
    """

    def __init__(self, n_classes, channel=3, image_size=32):
        super(Extractor, self).__init__()

        self.n_classes = n_classes
        self.channel = channel
        self.image_size = image_size

        self.conv0 = None
        self.norm0 = None

        self.conv1 = None
        self.norm1 = None
        
        self.conv2 = None
        self.norm2 = None

        self.relu = None
        self.pool = None
        self.flatten = None
        self.classifier = None

        self.zeropadding = None

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
        self.dense = tf.keras.layers.Dense(self.n_classes, activation="linear")
        self.zeropadding = tf.keras.layers.ZeroPadding2D(padding=(3, 3))
        super(Extractor, self).build(input_shape)
        
    def call(self, inputs, training=None):
        if self.channel == 1:
            inputs = self.zeropadding(inputs)
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
        feat = self.flatten(output)
        logits = self.dense(feat)
        return feat, logits
    
    def model(self):
        x = tf.keras.Input(shape=(28, 28, 1))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


class TransformLayer(tf.keras.layers.Layer):
    def __init__(self, img_size, kernel_size, mid_channel):
        super().__init__()
        # self.scale = tf.Variable(1.)
        self.scale = tf.Variable(
            tf.ones((1, img_size[0] - kernel_size + 1, img_size[1] - kernel_size + 1, mid_channel), 
                     dtype=tf.dtypes.float32
                     ),
            trainable=True
            )
        self.shift = tf.Variable(
            tf.zeros((1, img_size[0] - kernel_size + 1,  img_size[1] - kernel_size + 1, mid_channel), 
                     dtype=tf.dtypes.float32
                     ),
            trainable=True
            )

    def call(self, inputs):
        return inputs * self.scale + self.shift
    

class WeightedSum(keras.layers.Layer):
    def __init__(self, num_base):
        super(WeightedSum, self).__init__()
        self.num_base = num_base
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(num_base, num_base, 1, 1, 1), dtype="float32"),
            trainable=True,
        )

    def call(self, input):
        return tf.reduce_sum(tf.multiply(self.w, tf.expand_dims(input, axis=0)), axis=1)