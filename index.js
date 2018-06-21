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

async function train_BN(LEARNING_RATE, chart_id) {
  ui.isTraining(chart_id);
  await model.train_BN(data, ui.trainingLog, LEARNING_RATE, chart_id);
}

async function test() {
  const testExamples = 50;
  const batch = data.nextTestBatch(testExamples);
  const predictions = model.predict(batch.xs);
  const labels = model.classesFromLabel(batch.labels);

  ui.showTestResults(batch, predictions, labels);
}

async function mnist() {
  await load('0');
  await train(0.1, '0');
  await train_BN(0.1, '0');
  await train(0.5, '0');
  await train_BN(0.5, '0');

  $('#train'+'0').prop('disabled', false);
  $('#train'+'0').text('Done. Train Again?');
}


document.getElementById('train0').addEventListener('click', function() {
  mnist();
});
