
import * as tf from '@tensorflow/tfjs';

var $ = require('jquery');



export function isTraining(id) {
  $('#train'+id).prop('disabled', true);
  $('#train'+id).text('Training...');
}

export function isLoading(id) {
  $('#train'+id).prop('disabled', true);
  $('#train'+id).text('Loading Data...');
}

export function done(id) {
  $('#train'+id).prop('disabled', false);
  $('#train'+id).text('Done. Train Again?');
}
