import {MnistData} from './data';
import * as model from './model';
import * as ui from './ui';

var $ = require('jquery');

let data;
async function load(chart_id) {
  ui.isLoading(chart_id);
  data = new MnistData();
  await data.load();
}

async function train(LEARNING_RATE, chart_id) {
  ui.isTraining(chart_id);
  await model.train(data, ui.trainingLog, LEARNING_RATE, chart_id);
}

async function train_BN(LEARNING_RATE, chart_id, noise=false) {
  ui.isTraining(chart_id);
  await model.train_BN(data, ui.trainingLog, LEARNING_RATE, chart_id, noise=noise);
}

async function test() {
  const testExamples = 50;
  const batch = data.nextTestBatch(testExamples);
  const predictions = model.predict(batch.xs);
  const labels = model.classesFromLabel(batch.labels);

  ui.showTestResults(batch, predictions, labels);
}

async function train0() {
  var id = '0';
  await load(id);
  await train(0.1, id);
  await train_BN(0.1, id);
  await train(0.5, id);
  await train_BN(0.5, id);
  $('#train'+'0').prop('disabled', false);
  $('#train'+'0').text('Done. Train Again?');
}

async function train1() {
  var id = '1';
  await load(id);
  await train(undefined, id);
  await train_BN(undefined, id);
  await train_BN(undefined, id, true);
  $('#train'+'1').prop('disabled', false);
  $('#train'+'1').text('Done. Train Again?');
}


document.getElementById('train0').addEventListener('click', function() {
  train0();
});

document.getElementById('train1').addEventListener('click', function() {
  train1();
});
