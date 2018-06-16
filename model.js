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
const LEARNING_RATE = .1;
const BATCH_SIZE = 64;
const TRAIN_STEPS = 100;

// Data constants.
const IMAGE_SIZE = 28;
const LABELS_SIZE = 10;
const optimizer = tf.train.sgd(LEARNING_RATE);

// Variables that we want to optimize
const conv1OutputDepth = 8;
const conv1Weights =
    tf.variable(tf.randomNormal([5, 5, 1, conv1OutputDepth], 0, 0.1));

const BN1_Weights =
    tf.variable(tf.randomNormal([5, 5, 1, conv1OutputDepth], 0, 0.1));


const conv2InputDepth = conv1OutputDepth;
const conv2OutputDepth = 16;
const conv2Weights = tf.variable(
    tf.randomNormal([5, 5, conv2InputDepth, conv2OutputDepth], 0, 0.1));

const fullyConnectedWeights = tf.variable(tf.randomNormal(
    [7 * 7 * conv2OutputDepth, LABELS_SIZE], 0,
    1 / Math.sqrt(7 * 7 * conv2OutputDepth)));
const fullyConnectedBias = tf.variable(tf.zeros([LABELS_SIZE]));

// Loss function
function loss(labels, ys) {
  return tf.losses.softmaxCrossEntropy(labels, ys).mean();
}

// CNN
function model(inputXs) {
  const xs = inputXs.as4D(-1, IMAGE_SIZE, IMAGE_SIZE, 1);

  const strides = 2;
  const pad = 0;

  // Conv 1
  const layer1 = tf.tidy(() => {
    return xs.conv2d(conv1Weights, 1, 'same')
        .relu()
        .maxPool([2, 2], strides, pad);
  });

  // Conv 2
  const layer2 = tf.tidy(() => {
    return layer1.conv2d(conv2Weights, 1, 'same')
        .relu()
        .maxPool([2, 2], strides, pad);
  });

  // Final layer
  return layer2.as2D(-1, fullyConnectedWeights.shape[0])
      .matMul(fullyConnectedWeights)
      .add(fullyConnectedBias);
}

// CNN with BatchNorm
const model_BN = tf.sequential();
model_BN.add(tf.layers.conv2d({
  inputShape: [28, 28, 1],
  kernelSize: 5,
  filters: 8,
  strides: 1,
  activation: 'relu',
  kernelInitializer: 'varianceScaling'
}));
model_BN.add(tf.layers.batchNormalization({}));
model_BN.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
model_BN.add(tf.layers.conv2d({
  kernelSize: 5,
  filters: 16,
  strides: 1,
  activation: 'relu',
  kernelInitializer: 'varianceScaling'
}));
model_BN.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
model_BN.add(tf.layers.flatten());
model_BN.add(tf.layers.dense(
    {units: 10, kernelInitializer: 'varianceScaling', activation: 'softmax'}));



// Train the model.
export async function train(data, log) {
  var losses = ['CNN Loss'];
  const returnCost = true;

  for (let i = 0; i < TRAIN_STEPS; i++) {
    const cost = optimizer.minimize(() => {
      const batch = data.nextTrainBatch(BATCH_SIZE);
      return loss(batch.labels, model(batch.xs));
    }, returnCost);

    losses.push(cost.dataSync());
    log(`loss[${i}]: ${cost.dataSync()}`);

    vis.chart.load({
        columns: [
            losses
        ]
    });

    await tf.nextFrame();
  }
}

// Train the model with BN
export async function train_BN(data, log) {
  var losses = ['CNN with BatchNorm Loss'];
  var accuracyValues = ['CNN with BatchNorm Accuracy'];

  const optimizer = tf.train.sgd(LEARNING_RATE);
  model_BN.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
  const TRAIN_BATCHES = 150;
  const TEST_BATCH_SIZE = 1000;
  const TEST_ITERATION_FREQUENCY = 5;

  // Iteratively train our model on mini-batches of data.
  for (let i = 0; i < TRAIN_STEPS; i++) {
   const [batch, validationData] = tf.tidy(() => {
     const batch = data.nextTrainBatch(BATCH_SIZE);
     batch.xs = batch.xs.reshape([BATCH_SIZE, 28, 28, 1]);

     let validationData;
     // Every few batches test the accuracy of the model.
     if (i % TEST_ITERATION_FREQUENCY === 0) {
       const testBatch = data.nextTestBatch(TEST_BATCH_SIZE);
       validationData = [
         // Reshape the training data from [64, 28x28] to [64, 28, 28, 1] so
         // that we can feed it to our convolutional neural net.
         testBatch.xs.reshape([TEST_BATCH_SIZE, 28, 28, 1]), testBatch.labels
       ];
     }
     return [batch, validationData];
   });

   const history = await model_BN.fit(
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

// Predict the digit number from a batch of input images.
export function predict(x) {
  const pred = tf.tidy(() => {
    const axis = 1;
    return model(x).argMax(axis);
  });
  return Array.from(pred.dataSync());
}

// Given a logits or label vector, return the class indices.
export function classesFromLabel(y) {
  const axis = 1;
  const pred = y.argMax(axis);

  return Array.from(pred.dataSync());
}
