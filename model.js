
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
  var model, col_accs, col_accs, col_losses, col_meanCh, col_varCh, col_meanCh2, col_varCh2, col_losseLands, col_betas;
  var activations = [];
  switch(model_str) {
    case 'CNN':
      model = model_CNN;
      col_losses = 'CNN Loss (LR: ' + LEARNING_RATE + ')';
      col_accs = 'CNN Accuracy (LR: ' + LEARNING_RATE + ')';
      col_meanCh = 'CNN Mean Change (LR: ' + LEARNING_RATE + ')';
      col_varCh  = 'CNN Variance Change (LR: ' + LEARNING_RATE + ')';
      col_meanCh2 = col_meanCh;
      col_varCh2  = col_varCh;
      col_losseLands = 'CNN Loss Change (LR: ' + LEARNING_RATE + ')';
      col_betas = 'CNN β (LR: ' + LEARNING_RATE + ')';

      if (LEARNING_RATE === undefined){
        LEARNING_RATE = LR;
        col_losses =  'CNN Loss';
        col_accs = 'CNN Accuracy';
        col_meanCh = 'CNN Mean Change';
        col_varCh  = 'CNN Variance Change';
        col_meanCh2 = col_meanCh;
        col_varCh2  = col_varCh;
        col_losseLands = 'CNN Loss Change';
        col_betas = 'CNN β'
      }
      break;
    case 'CNN_BN':
      model = model_BN;
      col_losses = 'CNN + BatchNorm Loss (LR: ' + LEARNING_RATE + ')';
      col_accs = 'CNN + BatchNorm Accuracy (LR: ' + LEARNING_RATE + ')';
      col_meanCh = 'CNN + BatchNorm Mean Change (LR: ' + LEARNING_RATE + ')';
      col_varCh  = 'CNN + BatchNorm Variance Change (LR: ' + LEARNING_RATE + ')';
      col_meanCh2 = col_meanCh;
      col_varCh2  = col_varCh;
      col_losseLands = 'CNN + BatchNorm Loss Change (LR: ' + LEARNING_RATE + ')';
      col_betas = 'CNN + BatchNorm β (LR: ' + LEARNING_RATE + ')';
      if (LEARNING_RATE === undefined){
        LEARNING_RATE = LR;
        col_losses =  'CNN + BatchNorm Loss';
        col_accs = 'CNN + BatchNorm Accuracy';
        col_meanCh = 'CNN + BatchNorm Mean Change';
        col_varCh  = 'CNN + BatchNorm Variance Change';
        col_meanCh2 = col_meanCh;
        col_varCh2  = col_varCh;
        col_losseLands = 'CNN + BatchNorm Loss Change';
        col_betas = 'CNN + BatchNorm β'
      }
      break;
    default:
      throw "Unknown model: " + model_str;
  }
  model.freshParams()

  if (noise){
    col_losses += ' w/ Noise';
    col_accs += ' w/ Noise';
    col_meanCh += ' w/ Noise';
    col_varCh += ' w/ Noise';
    col_meanCh2 = col_meanCh;
    col_varCh2  = col_varCh;
  }

  let losses = [col_losses];
  let losseLands = [col_losseLands];
  let accuracies = [col_accs];
  let meanChange = [col_meanCh];
  let varChange = [col_varCh];
  let meanChange2 = [col_meanCh2];
  let varChange2 = [col_varCh2];
  let lossChanges = [col_losseLands];
  let betas = [col_betas];

  const optimizer = tf.train.sgd(LEARNING_RATE);

  let prevMean, prevVar, prevMean2, prevVar2, prevLoss, prevlayer1_data, prevGrad;

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
    const doGrad = (chart_id == '2');
    const cost = optimizer.minimize(() => {
      return model.loss(batch.labels, model.model(batch.xs, noise, doGrad));
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
    const trainPred = model.predict(batch.xs);
    const loss = model.loss(batch.labels, trainPred).dataSync();
    const accuracy = tf.metrics.categoricalAccuracy(batch.labels, trainPred).sum().dataSync()/BATCH_SIZE;

    losses.push(loss);
    accuracies.push(accuracy);
    vis['chart_losses_'+chart_id].load({
        columns: [
            losses
        ],
        type: 'spline'
    });
    vis['chart_accuracy_'+chart_id].load({
        columns: [
            accuracies
        ],
        type: 'spline'
    });

    if (prevMean2 === undefined)
    {
      //prevMean = model.moments_data.mean;
      //prevVar = model.moments_data.variance;
      prevMean2 = model.moments_data.mean;
      prevVar2 = model.moments_data.variance;
      meanChange2.push(0);
      varChange2.push(0);
    }
    else{
      // 1st layer
      /*meanChange.push(Math.abs(prevMean - model.moments_data.mean));
      varChange.push(Math.abs(prevVar - model.moments_data.variance));
      vis['mean_change_'+chart_id].load({
          columns: [
              meanChange
          ],
          type: 'spline'
      });
      vis['var_change_'+chart_id].load({
          columns: [
              varChange
          ],
          type: 'spline'
      });*/

      // 2nd layer
      meanChange2.push(Math.abs(prevMean2 - model.moments2_data.mean));
      varChange2.push(Math.abs(prevVar2 - model.moments2_data.variance));
      vis['l2mean_change_'+chart_id].load({
          columns: [
              meanChange2
          ],
          type: 'spline'
      });
      vis['l2var_change_'+chart_id].load({
          columns: [
              varChange2
          ],
          type: 'spline'
      });

      //prevMean = model.moments_data.mean;
      //prevVar = model.moments_data.variance;
      prevMean2 = model.moments2_data.mean;
      prevVar2 = model.moments2_data.variance;

    }

  }
  // Loss landscape and beta-smoothness
  else if(chart_id == '2'){
    const trainPred = model.predict(batch.xs);
    const loss = model.loss(batch.labels, trainPred).dataSync();

    if (prevLoss === undefined){
      prevLoss = loss;
      lossChanges.push(0);


    }
    else{
      lossChanges.push(Math.abs(prevLoss - loss));
      betas.push(model.beta_smoothness);

      vis['lossLand'+chart_id].load({
          columns: [
              lossChanges
          ],
          type: 'area-spline'
      });


      vis['betaSmooth'+chart_id].load({
          columns: [
              betas
          ]
      });

      prevLoss = loss;
    }


  }

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


    tf.dispose([batch, validationData]);
    await tf.nextFrame();
  }

  //violin.plot(chart_id+subId, activations);
}
































// Assumes a valid matrix and returns its dimension array.
// Won't work for irregular matrices, but is cheap.
function dim(mat) {
    if (mat instanceof Array) {
        return [mat.length].concat(dim(mat[0]));
    } else {
        return [];
    }
}

// Makes a validator function for a given matrix structure d.
function validator(d) {
    return function (mat) {
        if (mat instanceof Array) {
            return d.length > 0
                && d[0] === mat.length
                && every(mat, validator(d.slice(1)));
        } else {
            return d.length === 0;
        }
    };
}

// Combines dim and validator to get the required function.
function getdim(mat) {
    var d = dim(mat);
    return validator(d)(mat) ? d : false;
}

// Checks whether predicate applies to every element of array arr.
// This ought to be built into JS some day!
function every(arr, predicate) {
    var i, N;
    for (i = 0, N = arr.length; i < N; ++i) {
        if (!predicate(arr[i])) {
            return false;
        }
    }

    return true;
}
