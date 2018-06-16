/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as tf from '@tensorflow/tfjs';

import {MnistData} from './data';

import * as vis from "./visual";

// Hyperparameters.
const BATCH_SIZE = 64;
const TRAIN_STEPS = 100;

// Data constants.
const IMAGE_SIZE = 28;
const LABELS_SIZE = 10;

const TRAIN_BATCHES = 150;
const TEST_BATCH_SIZE = 1000;
const TEST_ITERATION_FREQUENCY = 5;

/*--------------------------------------------------------------------------------*/
// CNN
function create_model(){
  const model = tf.sequential();
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_SIZE, IMAGE_SIZE, 1],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense(
      {units: 10, kernelInitializer: 'varianceScaling', activation: 'softmax'}
  ));

  return model;
}

// CNN with BatchNorm
function create_model_BN(){

  const model = tf.sequential();
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_SIZE, IMAGE_SIZE, 1],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));
  model.add(tf.layers.batchNormalization({}));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense(
      {units: 10, kernelInitializer: 'varianceScaling', activation: 'softmax'}));

  return model;
}

/*-----------------------------------------------------------------------------------------------*/

// Train the CNN model (No BN)
export async function train(data, log, LEARNING_RATE) {
  let model = create_model();
  var losses = ['CNN Loss (LR: ' + LEARNING_RATE + ')'];
  var accuracyValues = ['CNN Accuracy (LR: ' + LEARNING_RATE + ')'];

  const optimizer = tf.train.sgd(LEARNING_RATE);
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  // Iteratively train our model on mini-batches of data.
  for (let i = 0; i < TRAIN_STEPS; i++) {
   const [batch, validationData] = tf.tidy(() => {
     const batch = data.nextTrainBatch(BATCH_SIZE);
     batch.xs = batch.xs.reshape([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1]);

     let validationData;
     // Every few batches test the accuracy of the model.
     if (i % TEST_ITERATION_FREQUENCY === 0) {
       const testBatch = data.nextTestBatch(TEST_BATCH_SIZE);
       validationData = [
         // Reshape the training data from [64, 28x28] to [64, 28, 28, 1] so
         // that we can feed it to our convolutional neural net.
         testBatch.xs.reshape([TEST_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1]), testBatch.labels
       ];
     }
     return [batch, validationData];
   });

   const history = await model.fit(
        batch.xs, batch.labels,
        {batchSize: BATCH_SIZE, validationData, epochs: 1}
    );

    const loss = history.history.loss[0];
    const accuracy = history.history.acc[0];

    losses.push(loss);
    vis.chart.load({
        columns: [
            losses
        ]
    });

    tf.dispose([batch, validationData]);

    await tf.nextFrame();
  }
}


// Train the model with BN
export async function train_BN(data, log, LEARNING_RATE) {
  let model = create_model_BN();
  var losses = ['CNN with BatchNorm Loss (LR: ' + LEARNING_RATE + ')'];
  var accuracyValues = ['CNN with BatchNorm Accuracy (LR: ' + LEARNING_RATE + ')'];

  const optimizer = tf.train.sgd(LEARNING_RATE);
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  // Iteratively train our model on mini-batches of data.
  for (let i = 0; i < TRAIN_STEPS; i++) {
   const [batch, validationData] = tf.tidy(() => {
     const batch = data.nextTrainBatch(BATCH_SIZE);
     batch.xs = batch.xs.reshape([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1]);

     let validationData;
     // Every few batches test the accuracy of the model.
     if (i % TEST_ITERATION_FREQUENCY === 0) {
       const testBatch = data.nextTestBatch(TEST_BATCH_SIZE);
       validationData = [
         // Reshape the training data from [64, 28x28] to [64, 28, 28, 1] so
         // that we can feed it to our convolutional neural net.
         testBatch.xs.reshape([TEST_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1]), testBatch.labels
       ];
     }
     return [batch, validationData];
   });

   const history = await model.fit(
        batch.xs, batch.labels,
        {batchSize: BATCH_SIZE, validationData, epochs: 1}
    );

    const loss = history.history.loss[0];
    const accuracy = history.history.acc[0];

    losses.push(loss);
    vis.chart.load({
        columns: [
            losses
        ]
    });

    tf.dispose([batch, validationData]);

    await tf.nextFrame();
  }
}
