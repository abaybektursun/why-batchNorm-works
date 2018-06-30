import * as tf from '@tensorflow/tfjs';
import {Scalar, serialization, Tensor, tidy, util} from '@tensorflow/tfjs-core';

import * as hparam from "./hyperParams"
var stats = require("stats-lite")

// Loss function
export function loss(labels, ys) {
  return tf.losses.softmaxCrossEntropy(labels, ys).mean();
}

// Variables that we want to optimize****************************************************
export let strides = 2;
export let pad = 0;

export let conv1OutputDepth = 8;
export let conv1Weights;

export let conv2InputDepth = conv1OutputDepth;
export let conv2OutputDepth = 16;
export let conv2Weights;

export let fullyConnectedWeights;
export let fullyConnectedBias;

export let scale1;
export let offset1;

export let scale2;
export let offset2;

export let moments;
export let moments2;

export let moments_nonTrain;
export let moments2_nonTrain;

export let train_step;
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
export let conv1g, conv1gl, beta_smoothness;
export let layer1_data;
export let layer2_data;
export let moments_data;
export let moments2_data;
export let grad, gradl;

// Our actual model
export function model(inputXs, noise=false, doGrad=false) {
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
  layer1_data = batchNorm1.flatten().dataSync();
  //grad = tf.grad(batchNorm1).dataSync();

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



  // Gradient ******************************************************************\
  if (doGrad){
  let a = 0.1; let betasl = [];
    while (a < hparam.A){
      conv1g = x => tf.tidy(() => {
        return batchNorm1.conv2d(conv2Weights, 1, 'same')
            .relu()
            .maxPool([2, 2], strides, pad)
            .batchNormalization(moments2.mean, moments2.variance, varianceEpsilon, scale2, offset2)
            .as2D(-1, fullyConnectedWeights.shape[0])
            .matMul(fullyConnectedWeights)
            .add(fullyConnectedBias);
      });
      grad = tf.grad(conv1g);
      let conv1l = batchNorm1.sub(grad(batchNorm1).mul(tf.scalar(a)));
      // Along the gradient
      conv1gl = x => tf.tidy(() => {
        return conv1l.conv2d(conv2Weights, 1, 'same')
            .relu()
            .maxPool([2, 2], strides, pad)
            .batchNormalization(moments2.mean, moments2.variance, varianceEpsilon, scale2, offset2)
            .as2D(-1, fullyConnectedWeights.shape[0])
            .matMul(fullyConnectedWeights)
            .add(fullyConnectedBias);
      });
      gradl = tf.grad(conv1gl);
      betasl.push(
        tf.norm(grad(batchNorm1).sub(gradl(conv1l)))
          .div(tf.norm(grad(batchNorm1).mul(tf.scalar(a)))).dataSync()
      );
      a += 0.05;
    }
    beta_smoothness = Math.max(...betasl);
  }
  //****************************************************************************\

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
