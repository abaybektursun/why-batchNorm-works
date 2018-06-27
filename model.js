
import * as tf from '@tensorflow/tfjs';
import {Scalar, serialization, Tensor, tidy, util} from '@tensorflow/tfjs-core';

import {MnistData} from './data';

import * as vis from "./visual";

import * as model_BN from "./BN_model"
import * as model_CNN from "./CNN_model"

import * as hparam from "./hyperParams"


const LR = hparam.LR
const BATCH_SIZE = hparam.BATCH_SIZE
const TRAIN_STEPS = hparam.TRAIN_STEPS
const IMAGE_SIZE = hparam.IMAGE_SIZE
const LABELS_SIZE = hparam.LABELS_SIZE
const TRAIN_BATCHES = hparam.TRAIN_BATCHES
const TEST_BATCH_SIZE = hparam.TEST_BATCH_SIZE
const TEST_ITERATION_FREQUENCY = hparam.TEST_ITERATION_FREQUENCY



/***************************************** Custom Noise Layer *****************************************/
/*export class Noise extends Layer {
  static className = 'Noise';

  constructor() {
    super({});
  }

  call(inputs: Tensor|Tensor[], kwargs: Kwargs): Tensor|Tensor[] {
    return tidy(() => {
      this.invokeCallHook(inputs, kwargs);
      const input = generic_utils.getExactlyOneTensor(inputs);
      const training = kwargs['training'] == null ? false : kwargs['training'];
      const noiseShape = this.getNoiseShape(input);
      const output =  K.inTrainPhase(
              () => K.dropout(input, this.rateScalar, noiseShape, this.seed),
              () => input, training) as Tensor;
      return output;
    });
  }
}
serialization.SerializationMap.register(Noise);*/
/*******************************************************************************************/

/*--------------------------------------------------------------------------------*/


/*-----------------------------------------------------------------------------------------------*/
// CNN
/*http://otoro.net/kanji-rnn/
function create_model(){
  const model = tf.sequential();
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_SIZE, IMAGE_SIZE, 1],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
  }));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
  }));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense(
      {units: 10, activation: 'softmax'}
  ));

  return model;
}

// CNN with BatchNorm
function create_model_BN(){http://otoro.net/kanji-rnn/
  const model = tf.sequential();
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_SIZE, IMAGE_SIZE, 1],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
  }));
  model.add(tf.layers.batchNormalization({}));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
  }));
  model.add(tf.layers.batchNormalization({}));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense(
      {units: 10,  activation: 'softmax'}));

  return model;
}*
/
/*-----------------------------------------------------------------------------------------------*/

export async function train(model_str, data, log, LEARNING_RATE, chart_id, noise=false, subId=undefined) {
  var model, col_accs, col_accs, col_losses;
  var activations = [];
  switch(model_str) {
    case 'CNN':
      model = model_CNN;
      col_losses = 'CNN Loss (LR: ' + LEARNING_RATE + ')'
      col_accs = 'CNN Accuracy (LR: ' + LEARNING_RATE + ')'
      if (LEARNING_RATE === undefined){
        LEARNING_RATE = LR
        col_losses =  'CNN Loss'
        col_accs = 'CNN Accuracy'
      }
      break;
    case 'CNN_BN':
      model = model_BN;
      col_losses = 'CNN + BatchNorm Loss (LR: ' + LEARNING_RATE + ')'
      col_accs = 'CNN + BatchNorm Accuracy (LR: ' + LEARNING_RATE + ')'
      if (LEARNING_RATE === undefined){
        LEARNING_RATE = LR
        col_losses =  'CNN + BatchNorm Loss'
        col_accs = 'CNN + BatchNorm Accuracy'
      }
      break;
    default:
      throw "Unknown model: " + model_str;
  }
  model.freshParams()

  if (noise){
    col_losses += ' w/ Noise';
    col_accs += ' w/ Noise';
  }

  var losses = [col_losses];
  var accuracies = [col_accs];

  const optimizer = tf.train.sgd(LEARNING_RATE);

  // Iteratively train our model on mini-batches of data.
  for (let i = 0; i < TRAIN_STEPS; i++) {
   const [batch, validationData] = tf.tidy(() => {
     const batch = data.nextTrainBatch(BATCH_SIZE);
     batch.xs = batch.xs.reshape([BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1]);

     let validationData;
     // Every few batches test the accuracy of the model.
     if (i % TEST_ITERATION_FREQUENCY === 0) {
       const testBatch = data.nextTestBatch(TEST_BATCH_SIZE);
       validationData = {
         xs: testBatch.xs.reshape([TEST_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1]),
         labels: testBatch.labels
       };
     }
     return [batch, validationData];
   });

    // Core Version Optimization
    const returnCost = true;
    const cost = optimizer.minimize(() => {
      return model.loss(batch.labels, model.model(batch.xs));
    }, returnCost);


   // First learning Curves need to be on test set
   if (validationData != null && chart_id == '0' ) {
    /* Plot and test curves*/
    const testPred = model.predict(validationData.xs)
    const loss = model.loss(validationData.labels, testPred).dataSync();
    const accuracy = tf.metrics.categoricalAccuracy(validationData.labels, testPred).sum().dataSync()/TEST_BATCH_SIZE;

    losses.push(loss);
    accuracies.push(accuracy);
    vis['chart_losses_'+chart_id].load({
        columns: [
            losses
        ]
    });
    vis['chart_accuracy_'+chart_id].load({
        columns: [
            accuracies
        ]
    });
  }
  // Second Charts on train set
  else if(chart_id == '1'){
    /* Plot and test curves*/
    const testPred = model.predict(batch.xs)
    const loss = model.loss(batch.labels, testPred).dataSync();
    const accuracy = tf.metrics.categoricalAccuracy(batch.labels, testPred).sum().dataSync()/BATCH_SIZE;

    losses.push(loss);
    accuracies.push(accuracy);
    vis['chart_losses_'+chart_id].load({
        columns: [
            losses
        ]
    });
    vis['chart_accuracy_'+chart_id].load({
        columns: [
            accuracies
        ]
      });

      // Activation Distributions
      /*if(i % 90 == 0){
        model.layer1_data.forEach(function(activs) {
          // Sample activations
          if (Math.random() < 0.01){
            activs.forEach(function(element){
              activations.push({
                date: i,
                value: element
              });
            });
          }
        });
      }*/

    }

    tf.dispose([batch, validationData]);
    await tf.nextFrame();
  }
  console.log(activations.length);
  //violin.plot(chart_id+subId, activations);
}
