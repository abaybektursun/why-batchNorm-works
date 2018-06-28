import * as tf from '@tensorflow/tfjs';
import {Scalar, serialization, Tensor, tidy, util} from '@tensorflow/tfjs-core';

import * as hparam from "./hyperParams"
var stats = require("stats-lite")

// Loss function
export function loss(labels, ys) {
  return tf.losses.softmaxCrossEntropy(labels, ys).mean();
}

// Variables that we want to optimize****************************************************
var strides = 2;
var pad = 0;

var conv1OutputDepth = 8;
var conv1Weights;

var conv2InputDepth = conv1OutputDepth;
var conv2OutputDepth = 16;
var conv2Weights;

var fullyConnectedWeights;
var fullyConnectedBias;

var scale1;
var offset1;

var scale2;
var offset2;

var moments;
var moments2;

var moments_nonTrain;
var moments2_nonTrain;

export var train_step;
//**************************************************************************************

export function freshParams(){
  conv1Weights =
      tf.variable(tf.randomNormal([5, 5, 1, conv1OutputDepth], 0, 0.1));


  scale1 = tf.variable(tf.randomNormal([conv1OutputDepth], 0, 0.1));
  offset1 = tf.variable(tf.zeros([conv1OutputDepth]));

  conv2Weights =
      tf.variable(tf.randomNormal([5, 5, conv2InputDepth, conv2OutputDepth], 0, 0.1));

  scale2 = tf.variable(tf.randomNormal([conv2OutputDepth], 0, 0.1));
  offset2 = tf.variable(tf.zeros([conv2OutputDepth]));

  fullyConnectedWeights = tf.variable(tf.randomNormal(
      [7 * 7 * conv2OutputDepth, hparam.LABELS_SIZE], 0,
      1 / Math.sqrt(7 * 7 * conv2OutputDepth)));
  fullyConnectedBias = tf.variable(tf.zeros([hparam.LABELS_SIZE]));

}

export let conv1, batchNorm1, conv2, batchNorm2;
export let layer1_data = [];
export let layer2_data = [];
export let moments_data = [];
export let moments2_data = [];


// Our actual model
export function model(inputXs, noise=false) {
  var xs = inputXs.as4D(-1, hparam.IMAGE_SIZE, hparam.IMAGE_SIZE, 1);

  // Conv 1
  conv1 = tf.tidy(() => {
    return xs.conv2d(conv1Weights, 1, 'same')
        .relu()
        .maxPool([2, 2], strides, pad);
  });

  // BatchNorm 1
  var varianceEpsilon = 1e-6
  moments = tf.tidy(() => {
    return tf.moments(conv1, [0, 1, 2]);
  });

  batchNorm1 = tf.tidy(() => {
    return conv1.batchNormalization(moments.mean, moments.variance, varianceEpsilon, scale1, offset1);
  });
  //layer1_data = layer1_data.concat(batchNorm1.dataSync());

  if (noise){
    batchNorm1 = tf.tidy(() => {
      return batchNorm1.add(tf.randomNormal(batchNorm1.shape, 0.15, 0.3));
    });
  }
  moments_nonTrain = tf.tidy(() => {
    return tf.moments(batchNorm1, [0, 1, 2]);
  });
  moments_data = {
    mean: stats.mean(moments_nonTrain.mean.dataSync()),
    variance: stats.mean(moments_nonTrain.variance.dataSync())
  };

  // Conv 2
  conv2 = tf.tidy(() => {
    return batchNorm1.conv2d(conv2Weights, 1, 'same')
        .relu()
        .maxPool([2, 2], strides, pad);
  });

  // BatchNorm 2
  moments2 = tf.tidy(() => {
    return tf.moments(conv2, [0, 1, 2]);
  });
  moments2_data = {
    mean: stats.mean(moments2.mean.dataSync()),
    variance: stats.mean(moments2.variance.dataSync())
  };
  batchNorm2 = tf.tidy(() => {
    return conv2.batchNormalization(moments2.mean, moments2.variance, varianceEpsilon, scale2, offset2);
  });
  if (noise){
    batchNorm2 = tf.tidy(() => {
      return batchNorm2.add(tf.randomNormal(batchNorm2.shape, 0.1, 0.5));
    });
  }
  //layer2_data = layer2_data.concat(batchNorm2.dataSync());

  // Final layer
  return batchNorm2.as2D(-1, fullyConnectedWeights.shape[0])
      .matMul(fullyConnectedWeights)
      .add(fullyConnectedBias);
}

/*module.exports.conv1 = conv1;
module.exports.batchNorm1 = batchNorm1;
module.exports.conv2 = conv2;
module.exports.batchNorm2 = batchNorm2;*/








// Predict the digit number from a batch of input images.
export function predict(x) {
  return tf.tidy(() => {
    const axis = 1;
    return model(x);
  });
  //return Array.from(pred.dataSync());
}

// Given a logits or label vector, return the class indices.
export function classesFromLabel(y) {
  const axis = 1;
  const pred = y.argMax(axis);

  return Array.from(pred.dataSync());
}
