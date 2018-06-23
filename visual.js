import * as d3 from "d3";
import * as c3 from "c3";
import Chart from "chart.js";

import * as utils from "./utils";
var stats = require("stats-lite")

var $ = require('jquery');

let chart_losses_0 = c3.generate({
    bindto: '#chart_losses_0',
    data: {
      empty: {
        label: {
          text: "Test Set Losses"
        }
      },
      columns: []
    }
});
let chart_accuracy_0 = c3.generate({
    bindto: '#chart_accuracy_0',
    data: {
      empty: {
        label: {
          text: "Test Set Accuracies"
        }
      },
      columns: []
    }
});
let chart_losses_1 = c3.generate({
    bindto: '#chart_losses_1',
    data: {
      empty: {
        label: {
          text: "Train Set Losses"
        }
      },
      columns: []
    }
});
let chart_accuracy_1 = c3.generate({
    bindto: '#chart_accuracy_1',
    data: {
      empty: {
        label: {
          text: "Train Set Accuracies"
        }
      },
      columns: []
    }
});



module.exports = {
  chart_losses_0: chart_losses_0,
  chart_accuracy_0: chart_accuracy_0,
  chart_losses_1: chart_losses_1,
  chart_accuracy_1: chart_accuracy_1,
};


/*--------------------------- Scatter plot --------------------------*/
/*-------------------------------------------------------------------*/

var color = Chart.helpers.color;

var numPoints = 100
var dataPoints = []
var i;
for(i = 0; i < numPoints; i++){
  dataPoints.push({
    x: randomScalingFactorX(),
    y: randomScalingFactorY(),
  })
};


var scatterChartData = {
	datasets: [{
		label: 'Data Points',
		borderColor: window.chartColors.red,
		backgroundColor: color(window.chartColors.red).alpha(0.2).rgbString(),
		data: dataPoints
	}]
};

window.onload = function() {
	var ctx = document.getElementById('canvas').getContext('2d');
	window.myScatter = Chart.Scatter(ctx, {
		data: scatterChartData,
		options: {
      scales: {
        yAxes: [{
            ticks: {
                beginAtZero:true
            }
        }],
        xAxes: [{
            ticks: {
                beginAtZero:true
            }
        }]
      },
			title: {
				display: true,
				text: 'Random Data'
			},
		}
	});
};

// Radmonize the Data
document.getElementById('randomizeData').addEventListener('click', function() {

  // Radmonize the Data
	scatterChartData.datasets[0].data = scatterChartData.datasets[0].data.map(function() {
		return {
			x: randomScalingFactorX(),
			y: randomScalingFactorY()
		};
	});

  window.myScatter.options.title.text = 'Random Data';
	window.myScatter.update();
  $('#subMean').prop('disabled', false);

});

// Subtract Mean
document.getElementById('subMean').addEventListener('click', function() {
  var xs = [];
  var ys = [];
  scatterChartData.datasets[0].data.forEach(function(point) {
    xs.push(point.x);
    ys.push(point.y);
  });

  var xmean = stats.mean(xs);
  var ymean = stats.mean(ys);

  // Radmonize the Data
	scatterChartData.datasets[0].data = scatterChartData.datasets[0].data.map(point =>{
    return{
	    x: point.x - xmean,
	    y: point.y - ymean
		};
  });

  window.myScatter.options.title.text = 'Centered Around Zero';
	window.myScatter.update();
});

// Scale The Variance
document.getElementById('normVar').addEventListener('click', function() {
  $('#subMean').prop('disabled', true);

  var xs = [];
  var ys = [];
  scatterChartData.datasets[0].data.forEach(function(point) {
    xs.push(point.x);
    ys.push(point.y);
  });

  var xmean = stats.mean(xs);
  var ymean = stats.mean(ys);

  var xstd = stats.stdev(xs);
  var ystd = stats.stdev(ys);

  // Radmonize the Data
	scatterChartData.datasets[0].data = scatterChartData.datasets[0].data.map(point =>{
    return{
	    x: (point.x - xmean)/xstd,
	    y: (point.y - ymean)/ystd
		};
  });
  window.myScatter.options.title.text = 'Normalized';
	window.myScatter.update();
});


//
