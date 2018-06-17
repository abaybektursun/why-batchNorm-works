import * as d3 from "d3";
import * as c3 from "c3";


// Loss
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


module.exports = {
  chart_losses_0: chart_losses_0,
  chart_accuracy_0: chart_accuracy_0,
};
