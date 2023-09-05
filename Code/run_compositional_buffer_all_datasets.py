#!/usr/bin/python3.6
"""
Script for training
"""

import argparse
import tensorflow as tf
import models
import datasets
import utils
import numpy as np
import os
import wandb
import time

def main(args):
    

    LOG_PATH = "..\logs\{}Buffer\{}".format(args.plugin, args.dataset)
    CLASSES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    wandb.init(sync_tensorboard=False,
            name="Test: K={}, T={}, I={}".format(K, T, I),
            project="CCMCL",
            job_type="CleanRepo",
            config=config
    )

    # Create array for storing results
    res_loss = np.zeros((RUNS, len(BUFFER_SIZES)), dtype=np.float)
    res_acc = np.zeros((RUNS, len(BUFFER_SIZES)), dtype=np.float)

    start_time = time.time()

    for i, BUFFER_SIZE in enumerate(BUFFER_SIZES):
        for run in range(RUNS):
            # Instantiate model and trainer
            model = models.CNN(10)
            model.build((None, IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]))
            print(model.summary())

            buf = models.CompositionalBalancedBuffer()
            train = utils.Trainer()

            # Instantiate optimizer
            optimizer = tf.keras.optimizers.SGD(LEARNING_RATE)

            # Load data set
            ds = datasets.SplitMNIST(num_validation=VAL_BATCHES*BATCH_SIZE)
            _, val_ds, test_ds = ds.get_all()
            val_ds = val_ds.cache().repeat().batch(BATCH_SIZE).map(utils.standardize)
            test_ds = test_ds.cache().batch(BATCH_SIZE).map(utils.standardize)
            IMG_SHAPE = (28, 28, 1)
            

            # Train on sequence
            
            wandb.log({"Val_acc": 0, "Exp": 0})
            for t in range(TASKS):
                # Determine classes
                classes = CLASSES[2 * t:2 * (t + 1)]
                # Load training data set
                train_ds, _, _ = ds.get_split(classes)
                train_ds = train_ds.cache().repeat().shuffle(10000).batch(BATCH_SIZE).map(utils.standardize)
                # Distill
                print("Distilling classes {}..".format(classes))
                # Compress and store
                for c in classes:
                    # Load data set
                    train_ds, _, _ = ds.get_split(c)
                    train_ds = train_ds.cache().repeat().shuffle(10000).batch(DIST_BATCH_SIZE).map(utils.standardize)
                    buf.compress_add(train_ds, c, DIST_BATCH_SIZE, LEARNING_RATE, DIST_LEARNING_RATE, IMG_SHAPE,
                                    int(BUFFER_SIZE / len(CLASSES)), K, T, I, model)
                buf.summary()
                # Training loop
                m_train_loss = tf.keras.metrics.Mean()
                m_val_acc = tf.keras.metrics.Accuracy()
                m_val_loss = tf.keras.metrics.Mean()
                utils.reinitialize_model(model)
                for iters in range(ITERS):
                    # Sample a batch from the buffer and train
                    x_r, y_r = buf.sample(BATCH_SIZE)
                    current_loss = train.train_step(x_r, y_r, model, optimizer)
                    m_train_loss.update_state(current_loss)
                    if iters % VAL_ITERS == VAL_ITERS - 1:
                        # Validation
                        val_iters = 0
                        for x, y in val_ds:
                            val_iters += 1
                            logits = model(x, training=False)
                            current_loss = tf.keras.losses.categorical_crossentropy(y, logits, from_logits=True)
                            m_val_loss.update_state(current_loss)
                            m_val_acc.update_state(tf.argmax(y, axis=-1), tf.argmax(logits, axis=-1))
                            if val_iters == VAL_BATCHES:
                                # Get metrics
                                train_loss = m_train_loss.result()
                                m_train_loss.reset_states()
                                val_acc = m_val_acc.result()
                                m_val_acc.reset_states()
                                val_loss = m_val_loss.result()
                                m_val_loss.reset_states()
                                print("Task: {} Iter: {} Train Loss: {:.3} Val Loss: {:.3} Val Accuracy: {:.3}".format(t,
                                                                                                                    iters,
                                                                                                                    train_loss,
                                                                                                                    val_loss,
                                                                                                                    val_acc))
                                # Reset validation iterations
                                val_iters = 0
                                break
                            
                wandb.log({"Val_acc": val_acc, "Exp": t + 1})

            # Test model on complete data set
            m_test_acc = tf.keras.metrics.Accuracy()
            m_test_loss = tf.keras.metrics.Mean()
            for x, y in test_ds:
                logits = model(x, training=False)
                current_loss = tf.keras.losses.categorical_crossentropy(y, logits, from_logits=True)
                m_test_acc.update_state(tf.argmax(y, axis=-1), tf.argmax(logits, axis=-1))
                m_test_loss.update_state(current_loss)
            test_loss = m_test_loss.result()
            test_acc = m_test_acc.result()
            print("Test Loss: {:.3} Test Accuracy {:.3}".format(test_loss, test_acc))
            # Store result
            res_loss[run, i] = test_loss
            res_acc[run, i] = test_acc

            # Write results
            print("Saving results to {}...".format(LOG_PATH))
            if not(os.path.exists(LOG_PATH)):
                # create the directory you want to save to
                os.makedirs(LOG_PATH)
            np.save(LOG_PATH + "/acc.npy", res_acc)
            np.savetxt(LOG_PATH + "/acc.log", res_acc, fmt="%.4f", delimiter=";",
                    header="Buffer sizes: " + str(BUFFER_SIZES))
            np.save(LOG_PATH + "/loss.npy", res_loss)
            np.savetxt(LOG_PATH + "/loss.log", res_loss, fmt="%.4f", delimiter=";",
                    header="Buffer sizes: " + str(BUFFER_SIZES))
            np.save(LOG_PATH + "/w_" + str(BUFFER_SIZE) + ".npy", buf.w_buffer)
            np.save(LOG_PATH + "/c_" + str(BUFFER_SIZE) + ".npy", buf.c_buffer)

    final_time_cost = time.time() - start_time
    wandb.log({"Final Time Cost": final_time_cost})


if __name__ == "__main__":

    utils.enable_gpu_mem_growth()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MNIST',
                        choices=['MNIST', 'CIFAR10', 'FashionMNIST', 'SVHN'])
    parser.add_argument('--RUNS', type=int, default=1,
                        help='how many times the experiment is repeated')
    parser.add_argument('--TASKS', type=int, default=5,
                        help='number of groups in which all classes are divided')
    parser.add_argument('-BUFFER_SIZES', type=int, default=100,
                        help='total memory size')
    parser.add_argument('--LEARNING_RATE', type=float, default=0.01,
                        help='learning rate for training (updating networks)')
    parser.add_argument('--DIST_LEARNING_RATE', type=float, default=0.01,
                        help='learning rate for distillation (updating images)')
    parser.add_argument('--BATCH_SIZE', type=int, default=128,
                        help='')
    parser.add_argument('--DIST_BATCH_SIZE', type=int, default=128,
                        help='')
    parser.add_argument('--ITERS', type=int, default=500,
                        help='number of iterations for validation training')
    parser.add_argument('--VAL_ITERS', type=int, default=10,
                        help='Validation interval during test training')
    parser.add_argument('--VAL_BATCHES', type=int, default=10,
                        help='Batchsize for validation')
    
    # Hyperparameters to be heavily tuned
    parser.add_argument('--K', type=int, default=10, 
                        help='number of distillation iterations')
    parser.add_argument('--T', type=int, default=10,
                        help='number of outerloops')
    parser.add_argument('--I', type=int, default=10,
                        help='number of image update within one outerloop')
    
    parser.add_argument('--plugin', type=str, default='Compositional', 
                        choices=['Compositional', 'Compressed'],
                        help='method for condensation')

    args = parser.parse_args()

    main(args)