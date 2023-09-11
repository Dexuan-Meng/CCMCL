"""
This file contains models.
"""

import tensorflow as tf
import tensorflow_addons as tfa
import utils
import abc
import numpy as np
import wandb
from tqdm import tqdm
    
    
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
                if len(g.shape) == 2:
                    g_norm = tf.math.l2_normalize(g, axis=0)
                    gs_norm = tf.math.l2_normalize(gs, axis=0)
                    inner = tf.reduce_sum(tf.multiply(g_norm, gs_norm), axis=0)
                if len(g.shape) == 4:
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

    def compress(self, ds, c, img_shape, num_synth, k, buf=None, verbose=False):
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
                x_ds, y_ds = next(ds_iter)
                # Perform distillation step
                for i in range(self.I): # one batch of dataset distills the components I iterations
                    dist_loss = self.distill_step(x_ds, y_ds, c_s, w_s, y_s)
                    wandb.log({"Distill/Matching Loss": dist_loss, 'Distill_step': starting_step + distill_step})
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


class CompositionalBalancedBuffer(object):
    """
    Buffer that adds a compression to the balanced buffer
    """

    def __init__(self):
        self.c_buffer = []
        self.w_buffer = []
        self.y_buffer = []
        self.buffer_box = []
        super(CompositionalBalancedBuffer, self).__init__()

    def compress_add(self, ds, c, batch_size, train_learning_rate, dist_learning_rate, img_shape, num_synth, K, T, I, mdl, verbose=False):
        # Create compressor
        comp = CompositionalCompressor(batch_size, train_learning_rate, dist_learning_rate, K, T, mdl, I=I)
        # Compress data
        num_weights = int(2*num_synth)
        num_components = int(num_synth)
        print("Compressing class {} down to {} weights and {} components...".format(c, num_weights, num_components))
        c_s, w_s, y_s = comp.compress(ds, c, img_shape, num_weights, num_components, self, verbose=False)
        # Add compressed data to buffer
        self.c_buffer.append(c_s)
        self.w_buffer.append(w_s)
        self.y_buffer.append(y_s)
        
        self.buffer_box[0] = self.c_buffer
        self.buffer_box[1] = self.w_buffer
        self.buffer_box[2] = self.y_buffer

    @staticmethod
    def compose_image(c_buffer, w_buffer, cl, idx):
        if idx is None:
            comp = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(w_buffer[cl], tf.expand_dims(c_buffer[cl], axis=0)), axis=1))
        else:
            comp = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(w_buffer[cl][idx], tf.expand_dims(c_buffer[cl], axis=0)), axis=1))
        return comp

    def sample(self, k):
        # Randomly select and return k examples with their labels from the buffer
        num_classes = len(self.w_buffer)
        if num_classes > 0:
            data = np.zeros((k, self.c_buffer[0].shape[1], self.c_buffer[0].shape[2], self.c_buffer[0].shape[3]), dtype=np.single)
            labels = np.zeros((k, self.y_buffer[0].shape[1]), dtype=np.single)
            for i in range(k):
                # Sample class
                cl = np.squeeze(np.random.randint(0, num_classes, 1))
                # Sample instance
                idx = np.squeeze(np.random.randint(0, self.w_buffer[cl].shape[0]))
                # Compose image
                comp = self.compose_image(self.c_buffer, self.w_buffer, cl, idx)
                # comp = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(self.w_buffer[cl][idx], tf.expand_dims(self.c_buffer[cl], axis=0)), axis=1))
                data[i] = comp
                labels[i] = self.y_buffer[cl][idx]
            return data, labels
        else:
            return None, None

    def summary(self):
        print("+======================================+")
        print("| Summary                              |")
        print("+======================================+")
        for i in range(len(self.w_buffer)):
            print("| Class {}: {} Instances {} components".format(i, self.w_buffer[i].shape[0], self.w_buffer[i].shape[1]))
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
    def distill_step(self, x, y):

        # Minimize cosine similarity between gradients
        with tf.GradientTape() as inner_tape:
            logits_x = self.mdl(x, training=False)
            loss_x = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, logits_x, from_logits=True))
        grads = inner_tape.gradient(loss_x, self.mdl.trainable_variables)

        with tf.GradientTape() as tape_w_s:
            with tf.GradientTape() as tape_c_s:

                # Make prediction using model
                with tf.GradientTape() as inner_tape:
                    comp = self.compose_image(self.base_image, self.stylers)
                    # comp = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(w_s, tf.expand_dims(self.base_image, axis=0)), axis=1))
                    logits_s = self.mdl(comp, training=False)
                    loss_s = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(self.syn_label, logits_s, from_logits=True))
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

        return dist_loss
    
    @tf.function
    def update_extractor(self, x, y):
        # compute contrastive loss and update extractor
        sim_content_loss = 0
        with tf.GradientTape() as tape_extractor:
            
            _, ds_logits = self.extractor(x)
            cls_content_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, ds_logits, from_logits=True))

            twin_image = self.get_twin_image()
            embed_c, _ = self.extractor(twin_image)
            likeli_content_loss = tf.reduce_mean(((1. - self.cosine_similarity(embed_c[:self.base_image.shape[0]], embed_c[self.base_image.shape[0]:])) / 2.))
            embed_c_0, _ = tf.linalg.normalize(embed_c[:self.base_image.shape[0]])
            embed_c_1, _ = tf.linalg.normalize(embed_c[self.base_image.shape[0]:])
            contrast_content_loss = self.contrastive_loss(
                tf.stack([embed_c_0, embed_c_1], axis=1), 
                tf.argmax(self.syn_label, axis=1)[:self.base_image.shape[0]]
                )
            
            sim_content_loss = sim_content_loss + cls_content_loss * self.lambda_cls_content \
                                                + likeli_content_loss * self.lambda_likeli_content \
                                                + contrast_content_loss * self.lambda_contrast_content

        dist_grad_extractor = tape_extractor.gradient(sim_content_loss, self.extractor.trainable_variables)
        self.extractor_opt.apply_gradients(zip(dist_grad_extractor, self.extractor.trainable_variables))

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
    
    def compress(self, ds, c, img_shape, num_stylers, k, buf=None, verbose=False):

        # Create and initialize synthetic data
        # k = num_components = int(BUFFER_SIZE / len(CLASSES)) = 10
        self.base_image = tf.Variable(tf.random.uniform((k, img_shape[0], img_shape[1], img_shape[2]), maxval=tf.constant(1.0, dtype=tf.float32)))
        self.syn_label = tf.Variable(tf.one_hot(tf.constant(c, shape=(num_stylers * k,), dtype=tf.int32), 10), dtype=tf.float32)
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
                    dist_loss = self.distill_step(x_ds, y_ds)
                    wandb.log({"Distill/Matching Loss": dist_loss, 'Distill_step': starting_step + distill_step})
                    self.update_extractor(x_ds, y_ds)
                    distill_step += 1

                # Perform innerloop training step
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
        return self.base_image, self.stylers, self.syn_label


class FactorizationBalancedBuffer(CompositionalBalancedBuffer):
    """
    Buffer that adds a compression to the balanced buffer
    """

    def __init__(self):
        self.base_buffer = []
        self.styler_buffer = []
        self.label_buffer = []
        self.buffer_box = []
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
        
        self.buffer_box.append(self.base_buffer)
        self.buffer_box.append(self.styler_buffer)
        self.buffer_box.append(self.label_buffer)

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
        # raise 'NotImplementedError'

    def sample(self, batch_size):
        # Randomly select and return k examples with their labels from the buffer
        num_classes = len(self.base_buffer)
        if num_classes > 0:
            data = np.zeros((batch_size, self.base_buffer[0].shape[1], self.base_buffer[0].shape[2], self.base_buffer[0].shape[3]), dtype=np.single)
            labels = np.zeros((batch_size, self.label_buffer[0].shape[1]), dtype=np.single)
            
            indices = np.random.permutation(len(self.styler_buffer[0]) * 100) # Max number of self.syn_images
            indices = list(indices % tf.shape(self.syn_images)[0,].numpy())[:batch_size]
            data = tf.gather(self.syn_images, indices).numpy()
            labels = tf.gather(self.syn_labels, indices).numpy()

            # for i in range(batch_size):
            #     # Sample class
            #     cl = np.squeeze(np.random.randint(0, num_classes, 1))
            #     # Sample instance
            #     idx_base_image = np.squeeze(np.random.randint(0, self.base_buffer[0].shape[0]))
            #     idx_styler = np.squeeze(np.random.randint(0, len(self.styler_buffer[cl])))
            #     # Compose image
            #     comp = self.compose_image(self.base_buffer, self.styler_buffer, cl, idx_base_image, idx_styler)
            #     data[i] = comp
            #     labels[i] = self.label_buffer[cl][0] # No need to sample labels, because they are all the same in one class

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
                                                                  self.base_buffer[i].shape[0]
                                                                  )
                )
        print("+--------------------------------------+")


class StyleTranslator(tf.keras.Model):
    """
    Single-layer-Conv2d encoder + scaling + translation + Single-layer-ConvTranspose2d decoder
    """

    def __init__(self, in_channel=3, mid_channel=3, out_channel=3, image_size=(28, 28, 1), kernel_size=3):
        super(StyleTranslator, self).__init__()
        self.in_channel = in_channel
        self.mid_channel = mid_channel
        self.out_channel = out_channel
        self.img_size = image_size
        self.kernel_size = kernel_size
        self.enc = None
        self.scale = None
        self.shift = None
        self.dec = None

    def build(self, input_shape):
        self.enc = tf.keras.layers.Conv2D(self.mid_channel, self.kernel_size, name='Conv2D')
        self.scale = tf.Variable(
            tf.ones((1, self.img_size[0]-self.kernel_size+1, self.img_size[1]-self.kernel_size+1, self.mid_channel), 
                     dtype=tf.dtypes.float32
                     ),
            trainable=True
            )
        self.shift = tf.Variable(
            tf.zeros((1, self.img_size[0]-self.kernel_size+1, self.img_size[1]-self.kernel_size+1, self.mid_channel), 
                     dtype=tf.dtypes.float32
                     ),
            trainable=True
            )
        self.dec = tf.keras.layers.Conv2DTranspose(self.out_channel, self.kernel_size, name='Conv2DTransposed')
        super(StyleTranslator, self).build(input_shape)
        
    def call(self, inputs, training=None):
        output = self.enc(inputs)
        output = self.scale * output + self.shift
        output = self.dec(output)
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
