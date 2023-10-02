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
from utils import make_grid

def main(args):
    
    CLASSES = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    task_classes = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    # Load data set
    if args.dataset == 'MNIST':
        ds = datasets.SplitMNIST(num_validation=args.VAL_BATCHES * args.BATCH_SIZE)
        _, _, test_ds = ds.get_all()
        val_ds_splitted = []
        for classes in task_classes:
            _, val_ds, _ = ds.get_split(classes)
            val_ds_splitted.append(val_ds.cache().repeat().batch(args.BATCH_SIZE).map(utils.standardize))
        test_ds = test_ds.cache().batch(args.BATCH_SIZE).map(utils.standardize)
        IMG_SHAPE = (28, 28, 1)
    elif args.dataset == 'CIFAR10':
        ds = datasets.SplitCIFAR10(num_validation=args.VAL_BATCHES * args.BATCH_SIZE)
        _, _, test_ds = ds.get_all()
        val_ds_splitted = []
        for classes in task_classes:
            _, val_ds, _ = ds.get_split(classes)
            val_ds_splitted.append(val_ds.cache().repeat().batch(args.BATCH_SIZE).map(utils.standardize))
        # val_ds = val_ds.cache().repeat().batch(args.BATCH_SIZE).map(utils.standardize)
        test_ds = test_ds.cache().batch(args.BATCH_SIZE).map(utils.standardize)
        IMG_SHAPE = (32, 32, 3)
    elif args.dataset not in ['MNIST', 'CIFAR10']:
        raise 'NotImplementedError'
    
    if args.plugin != 'Factorization':
        args.DUAL_CLASSES = False
    
    ID = str(np.random.randint(999)).zfill(3)

    for run in range(args.RUNS):

        
        val_acc_splitted = {0:[], 1:[], 2:[], 3:[], 4:[]}
        val_forgetting_splitted = {0:[], 1:[], 2:[], 3:[], 4:[]}

        wandb.init(sync_tensorboard=False,
                name="Proportion Study: {} {}-{} ".format(args.dataset, ID, run), 
                project="CCMCL",
                job_type="CleanRepo",
                config=args
        )

        start_time = time.time()
        # Instantiate model and trainer
        model = models.CNN(10)
        model.build((None, IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]))
        # print(model.summary())
        if args.plugin == 'Compositional':
            buf = models.CompositionalBalancedBuffer()
            condensation_args = {
                'batch_size': args.BATCH_SIZE,
                'train_learning_rate': args.LEARNING_RATE,
                'dist_learning_rate': args.DIST_LEARNING_RATE,
                'img_shape': IMG_SHAPE,
                'num_synth': int(args.BUFFER_SIZE / len(CLASSES)),
                'K': args.K,
                'T': args.T,
                'I': args.I,
                'log_histogram': args.log_histogram
            }
        elif args.plugin == 'Factorization':
            if args.DUAL_CLASSES:
                buf = models.DualClassesFactorizationBuffer()
            else:
                buf = models.FactorizationBalancedBuffer()
            
            condensation_args = {
                'num_stylers': args.num_stylers,
                'batch_size': args.BATCH_SIZE,
                'train_learning_rate': args.LEARNING_RATE,
                'img_learning_rate': args.DIST_LEARNING_RATE,
                'styler_learning_rate': args.styler_lr,
                'img_shape': IMG_SHAPE,
                'num_bases': int(args.BUFFER_SIZE / len(CLASSES)),
                'K': args.K,
                'T': args.T,
                'I': args.I,
                'IN': args.IN,
                'lambda_club_content': args.lambda_club_content,
                'lambda_likeli_content': args.lambda_likeli_content,
                'lambda_cls_content': args.lambda_cls_content,
                'lambda_contrast_content': args.lambda_contrast_content,
                'log_histogram': args.log_histogram,
                'current_data_proportion': args.current_data_proportion,
                'use_image_being_condensed': args.use_image_being_condensed
            }

        train = utils.Trainer()

        # Instantiate optimizer
        optimizer = tf.keras.optimizers.SGD(args.LEARNING_RATE)

        # Train on sequence
        
        wandb.log({"Validation/Val_acc": 0, "Validation/Task": 0})


        for t in range(args.TASKS):

            # Determine classes
            classes = CLASSES[2 * t : 2 * (t + 1)]
            # Load training data set
            train_ds, _, _ = ds.get_split(classes)
            train_ds = train_ds.cache().repeat().shuffle(10000).batch(args.BATCH_SIZE).map(utils.standardize)

            # Distill
            print("Distilling classes {}..".format(classes))
            # Compress and store

            if args.DUAL_CLASSES:
                dual_train_datasets = {}
                for c in classes:
                    # Load training data set
                    train_ds, _, _ = ds.get_split(c)
                    dual_train_datasets[c] = train_ds.cache().repeat().shuffle(10000).batch(args.BATCH_SIZE).map(utils.standardize)
                buf.compress_add(dual_train_datasets, classes, model, **condensation_args)
                buf.summary()
            else:
                for c in classes:
                    # Load training data set
                    train_ds, _, _ = ds.get_split(c)
                    train_ds = train_ds.cache().repeat().shuffle(10000).batch(args.BATCH_SIZE).map(utils.standardize)
                    buf.compress_add(train_ds, c, model, **condensation_args)
                buf.summary()

            for cl in classes:
                grid = make_grid(buf.buffer_box[0][cl])
                wandb.log({'BaseImages/class_{}'.format(cl): wandb.Image(grid)})
                comp = buf.compose_image(buf.buffer_box[0], buf.buffer_box[1], cl)
                # comp = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(buf.buffer_box[1][cl], tf.expand_dims(buf.buffer_box[0][cl], axis=0)), axis=1))
                grid = make_grid(comp)
                wandb.log({'SynImages/class_{}'.format(cl): wandb.Image(grid)})

            # Training loop
            m_train_loss = tf.keras.metrics.Mean()
            m_val_acc = tf.keras.metrics.Accuracy()
            m_val_loss = tf.keras.metrics.Mean()
            utils.reinitialize_model(model)
            for iters in range(args.ITERS):
                # Sample a batch from the buffer and train
                x_r, y_r = buf.sample(args.BATCH_SIZE)
                current_loss = train.train_step(x_r, y_r, model, optimizer)
                wandb.log({"Validation/train_loss": current_loss, "Validation/train_iters": t * args.ITERS + iters})
                m_train_loss.update_state(current_loss)

                val_acc_container = []
                val_forgetting_container = []
                if iters % args.VAL_ITERS == args.VAL_ITERS - 1:
                    # Validation
                    for idx, val_ds in enumerate(val_ds_splitted):
                        val_iters = 0
                        for x, y in val_ds:
                            val_iters += 1
                            logits = model(x, training=False)
                            current_loss = tf.keras.losses.categorical_crossentropy(y, logits, from_logits=True)
                            m_val_loss.update_state(current_loss)
                            m_val_acc.update_state(tf.argmax(y, axis=-1), tf.argmax(logits, axis=-1))
                            if val_iters == args.VAL_BATCHES / 5:
                                # Get metrics
                                train_loss = m_train_loss.result()
                                m_train_loss.reset_states()
                                val_acc = m_val_acc.result()
                                m_val_acc.reset_states()
                                val_loss = m_val_loss.result()
                                m_val_loss.reset_states()
                                print("Task: {} Iter: {} Classes:{} Train Loss: {:.3} Val Loss: {:.3} Val Accuracy: {:.3}".format(t,
                                                                                                                                  iters,
                                                                                                                                  task_classes[idx],
                                                                                                                                  train_loss,
                                                                                                                                  val_loss,
                                                                                                                                  val_acc))
                                # Reset validation iterations
                                val_iters = 0
                                break
                        
                        val_acc_splitted[idx].append(val_acc)
                        val_acc_container.append(val_acc)
                        if len(val_acc_splitted[idx]) < idx + 2:
                            val_forgetting = 0
                        else:
                            val_forgetting = val_acc_splitted[idx][idx] - val_acc
                        val_forgetting_splitted[idx].append(val_forgetting)
                        val_forgetting_container.append(val_forgetting)
                        wandb.log({"Validation/Val_acc_{}".format(task_classes[idx]): val_acc,
                                   "Validation/Val_forgetting_{}".format(task_classes[idx]): val_forgetting,
                                   "Validation/Task_{}".format(task_classes[idx]): t + 1})
                    s = 0 if t == 0 else np.sum(val_forgetting_container) / t
                    wandb.log({"Validation/Val_acc": np.mean(val_acc_container),
                               "Validation/Val_forgetting": 0 if t == 0 else np.sum(val_forgetting_container) / t,
                               "Validation/Task": t + 1})
                    

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
        wandb.log({'Test/Loss': test_loss,
                'Test/Accuracy': test_acc})

        final_time_cost = time.time() - start_time
        wandb.log({"Final Time Cost": final_time_cost})

        wandb.finish()


if __name__ == "__main__":

    utils.enable_gpu_mem_growth()

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--DUAL_CLASSES', type=bool, default=True,
                        help='whether compressing data of two classes at a time')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        choices=['MNIST', 'CIFAR10', 'FashionMNIST', 'SVHN'])
    parser.add_argument('--TASKS', type=int, default=5,
                        help='number of groups in which all classes are divided')
    parser.add_argument('--BUFFER_SIZE', type=int, default=100,
                        help='total memory size')
    parser.add_argument('--LEARNING_RATE', type=float, default=0.01,
                        help='learning rate for training (updating networks)')
    parser.add_argument('--DIST_LEARNING_RATE', type=float, default=0.1,
                        help='learning rate for distillation (updating images)')
    parser.add_argument('--styler_lr', type=float, default=0.01,
                        help='learning rate for distillation (updating styler)')
    parser.add_argument('--BATCH_SIZE', type=int, default=128,
                        help='')
    parser.add_argument('--DIST_BATCH_SIZE', type=int, default=128,
                        help='')
    parser.add_argument('--ITERS', type=int, default=1000,
                        help='number of iterations for validation training')
    parser.add_argument('--VAL_ITERS', type=int, default=1000,
                        help='Validation interval during test training')
    parser.add_argument('--VAL_BATCHES', type=int, default=10,
                        help='Batchsize for validation')
    parser.add_argument('--log_histogram', type=bool, default=False,
                        help='whether to log histogram to wandb')
    parser.add_argument('--current_data_proportion', type=float, default=0.2,
                        help='proportion of data of current classes for updating model in innerloop')
    parser.add_argument('--use_image_being_condensed', type=bool, default=True,
                        help='whether to use image being condensed or real images as data of current \
                            classes while updating model in Innerloop')

    # Hyperparameters to be heavily tuned
    parser.add_argument('--RUNS', type=int, default=1,
                        help='how many times the experiment is repeated')
    parser.add_argument('--num_stylers', type=int, default=2)

    parser.add_argument('--K', type=int, default=20, 
                        help='number of distillation iterations')
    parser.add_argument('--T', type=int, default=10,
                        help='number of outerloops')
    parser.add_argument('--I', type=int, default=10,
                        help='number of image update within one outerloop')
    parser.add_argument('--IN', type=int, default=1,
                        help='number of image update within one outerloop')

    parser.add_argument('--lambda_club_content', type=float, default=10)
    parser.add_argument('--lambda_contrast_content', type=float, default=10)
    parser.add_argument('--lambda_likeli_content', type=float, default=1)
    parser.add_argument('--lambda_cls_content', type=float, default=1)

    parser.add_argument('--group', type=int, default=10)

    parser.add_argument('--plugin', type=str, default='Factorization', 
                        choices=['Compositional', 'Compressed', 'Factorization'],
                        help='method for condensation')

    args = parser.parse_args()

    main(args)