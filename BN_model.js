
// Loss function
export function loss(labels, ys) {
  return tf.losses.softmaxCrossEntropy(labels, ys).mean();
}

// Our actual model
export function model_BN(LABELS_SIZE, IMAGE_SIZE, inputXs, noise=false) {
  // Variables that we want to optimize****************************************************
  const conv1OutputDepth = 8;
  const conv1Weights =
      tf.variable(tf.randomNormal([5, 5, 1, conv1OutputDepth], 0, 0.1));

  const conv2InputDepth = conv1OutputDepth;
  const conv2OutputDepth = 16;
  const conv2Weights =
      tf.variable(tf.randomNormal([5, 5, conv2InputDepth, conv2OutputDepth], 0, 0.1));

  const fullyConnectedWeights = tf.variable(tf.randomNormal(
      [7 * 7 * conv2OutputDepth, LABELS_SIZE], 0,
      1 / Math.sqrt(7 * 7 * conv2OutputDepth)));
  const fullyConnectedBias = tf.variable(tf.zeros([LABELS_SIZE]));
  //**************************************************************************************

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









// Predict the digit number from a batch of input images.
export function predict(x) {
  const pred = tf.tidy(() => {
    const axis = 1;
    return model_BN(x).argMax(axis);
  });
  return Array.from(pred.dataSync());
}

// Given a logits or label vector, return the class indices.
export function classesFromLabel(y) {
  const axis = 1;
  const pred = y.argMax(axis);

  return Array.from(pred.dataSync());
}
