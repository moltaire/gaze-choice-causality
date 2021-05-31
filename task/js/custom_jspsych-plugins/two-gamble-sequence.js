/**
 * jspsych-two-gamble-sequence
 * Felix Molter
 *
 * a jsPsych plugin for displaying parts of two all-or-nothing gamble stimuli, described by a pie chart and a bar chart in sequence
 *
 **/

jsPsych.plugins["two-gamble-sequence"] = (function () {
  var plugin = {};

  plugin.info = {
    name: "two-gamble-sequence",
    description: "",
    parameters: {
      stimulus: {
        type: jsPsych.plugins.parameterType.OBJECT,
        description: "Stimulus object, containing attributes p0, p1, m0, m1",
      },
      choices: {
        type: jsPsych.plugins.parameterType.KEYCODE,
        array: true,
        pretty_name: "Choices",
        default: jsPsych.ALL_KEYS,
        description:
          "The keys the subject is allowed to press to respond to the stimulus.",
      },
      topAttribute: {
        type: jsPsych.plugins.parameterType.STR,
        pretty_name: "Top attribute.",
        default: "probability",
        description:
          "Which attribute ('probability' or 'outcome') is displayed in the top row.",
      },
      sequence: {
        type: jsPsych.plugins.parameterType.OBJECT,
        description:
          "Sequence object, containing attributes `durations`, `alternatives`, `attributes`",
      },
      choicePrompt: {
        type: jsPsych.plugins.parameterType.STR,
        pretty_name: "Choice prompt.",
        default: "Choose!",
        description: "The text shown after sequence presentation.",
      },
      choiceTimeout: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: "Choice timeout",
        default: 1000,
        description:
          "How long to wait for a response after sequence was shown.",
      },
      feedbackDuration: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: "Feedback duration.",
        default: 500,
        description: "Duration for which the feedback frame is shown (ms).",
      },
      feedbackColor: {
        type: jsPsych.plugins.parameterType.STR,
        pretty_name: "Feedback color",
        default: "Gold",
        description: "Color of the choice feedback frame and choice prompt",
      },
      stimForegroundColor: {
        type: jsPsych.plugins.parameterType.STR,
        pretty_name: "Stimulus foreground color",
        default: "ForestGreen",
        description: "Color of the filled portions of the gamble stimuli",
      },
      stimBackgroundColor: {
        type: jsPsych.plugins.parameterType.STR,
        pretty_name: "Stimulus background color",
        default: "FireBrick",
        description: "Color of the unfilled portions of the gamble stimuli",
      },
      stimFrameColor: {
        type: jsPsych.plugins.parameterType.STR,
        pretty_name: "Stimulus Frame color",
        default: "White",
        description: "Color of the frame around each gamble stimulus",
      },
      pieChartRadius: {
        type: jsPsych.plugins.parameterType.FLOAT,
        pretty_name: "Pie chart radius, relative to window width.",
        default: 0.1,
        description:
          "The width of the pie charts (and height of the bar charts), expressed in units of window width: A value of 0.1 corresponds to a tenth of the window's width.",
      },
      boxWidth: {
        type: jsPsych.plugins.parameterType.FLOAT,
        pretty_name: "Bounding box width, relative to pieChartRadius.",
        default: 2,
      },
      boxHeight: {
        type: jsPsych.plugins.parameterType.FLOAT,
        pretty_name: "Bounding box height, relative to window height.",
        default: 0.9,
      },
      timeoutWarningColor: {
        type: jsPsych.plugins.parameterType.STR,
        pretty_name: "Timeout warning color",
        default: "red",
        description: "Color of warning after time out (no response).",
      },
      timeoutWarningText: {
        type: jsPsych.plugins.parameterType.STR,
        pretty_name: "Timeout warning text",
        default: "Too slow!",
        description: "Text of warning after time out (no response).",
      },
      showTimeoutWarning: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: "Show timeout warning",
        default: true,
        description: "Whether to show timeout warning after no response.",
      },
    },
  };

  plugin.trial = async function (display_element, trial) {
    // Randomize item positions
    var stimuli = [
      [trial.stimulus.p0, trial.stimulus.m0],
      [trial.stimulus.p1, trial.stimulus.m1],
    ];
    var leftItem = jsPsych.randomization.sampleWithoutReplacement([0, 1], 1)[0];
    trial.pL = stimuli[leftItem][0]; // left probability
    trial.mL = stimuli[leftItem][1]; // left magnitude
    trial.pR = stimuli[1 - leftItem][0]; // right probability
    trial.mR = stimuli[1 - leftItem][1]; // right magnitude

    var response = {
      rt: null,
      key: null,
    };

    var choice;
    var chosenP;
    var chosenM;

    //--------Set up Canvas start-------
    var gambleCanvas = document.createElement("canvas");
    gambleCanvas.width = 0.95 * window.innerWidth;
    gambleCanvas.height = 0.95 * window.innerHeight;
    display_element.appendChild(gambleCanvas);

    function drawPiechart(
      ctx,
      p,
      x,
      y,
      radius,
      fillColor = "green",
      backColor = "red"
    ) {
      // Filled segment
      ctx.fillStyle = fillColor;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.arc(
        x,
        y,
        radius,
        1.5 * Math.PI,
        1.5 * Math.PI + p * 2 * Math.PI,
        false
      );
      ctx.lineTo(x, y);
      ctx.closePath();
      ctx.fill();

      // Background segment
      ctx.fillStyle = backColor;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.arc(
        x,
        y,
        radius,
        1.5 * Math.PI + p * 2 * Math.PI,
        3.5 * Math.PI,
        false
      );
      ctx.lineTo(x, y);
      ctx.closePath();
      ctx.fill();
    }

    function drawBarchart(
      ctx,
      m,
      x,
      y,
      width,
      height,
      fillColor = "green",
      backColor = "red"
    ) {
      // Background segment
      ctx.fillStyle = backColor;
      ctx.beginPath();
      ctx.fillRect(x - 0.5 * width, y - 0.5 * height, width, height);

      // Filled segment
      ctx.fillStyle = fillColor;
      ctx.beginPath();
      ctx.fillRect(
        x - 0.5 * width,
        y - 0.5 * height + (1 - m) * height,
        width,
        m * height
      );
      ctx.closePath();
    }

    var drawStims = function (i) {
      // Frames
      // Left
      ctx.strokeStyle = trial.stimFrameColor;
      ctx.lineWidth = 2;
      ctx.strokeRect(
        left_xpos - trial.boxWidth * radius, // frame top left x-coordinate
        (gambleCanvas.height * (1 - trial.boxHeight)) / 2, // frame top left y-coordinate
        2 * trial.boxWidth * radius, // frame width
        gambleCanvas.height * trial.boxHeight // frame height. Use 2x + height = 1 for symmetric layout
      );
      // Right
      ctx.strokeRect(
        right_xpos - trial.boxWidth * radius, // frame top left x-coordinate
        (gambleCanvas.height * (1 - trial.boxHeight)) / 2, // frame top left y-coordinate
        2 * trial.boxWidth * radius, // frame width
        gambleCanvas.height * trial.boxHeight // frame height
      );

      // Loop over alternatives and draw everything
      var alt;
      for (alt = 0; alt < 2; alt++) {
        // -- Probability
        if (
          trial.sequence.alternatives[i] == alt ||
          trial.sequence.alternatives[i] == "all"
        ) {
          if (
            trial.sequence.attributes[i] == "p" ||
            trial.sequence.attributes[i] == "all"
          ) {
            drawPiechart(
              ctx,
              stimuli[alt][0],
              xpos[alt],
              probability_ypos,
              radius,
              trial.stimForegroundColor,
              trial.stimBackgroundColor
            );
          }
          // -- Magnitude
          if (
            trial.sequence.attributes[i] == "m" ||
            trial.sequence.attributes[i] == "all"
          ) {
            drawBarchart(
              ctx,
              stimuli[alt][1],
              xpos[alt],
              magnitude_ypos,
              width,
              height,
              trial.stimForegroundColor,
              trial.stimBackgroundColor
            );
          }
        }
      }
    };

    // function to end trial when it is time
    var end_trial = function () {
      // kill any remaining setTimeout handlers
      jsPsych.pluginAPI.clearAllTimeouts();

      // kill keyboard listeners
      if (typeof keyboardListener !== "undefined") {
        jsPsych.pluginAPI.cancelKeyboardResponse(keyboardListener);
      }

      // gather the data to store for the trial
      var trial_data = {
        rt: response.rt,
        key_press: response.key,
        pL: trial.pL,
        mL: trial.mL,
        pR: trial.pR,
        mR: trial.mR,
        p0: trial.stimulus.p0,
        p1: trial.stimulus.p1,
        m0: trial.stimulus.m0,
        m1: trial.stimulus.m1,
        phase: trial.stimulus.phase,
        condition: trial.stimulus.condition,
        sequence: trial.sequence,
        choice: choice,
        chosenP: chosenP,
        chosenM: chosenM,
        wWidth: window.innerWidth,
        wHeight: window.innerHeight,
      };

      if (response.key == null && trial.showTimeoutWarning) {
        // Clear canvas
        ctx.clearRect(0, 0, gambleCanvas.width, gambleCanvas.height);
        // Draw "Too slow!" feedback
        ctx.font = "60px sans-serif";
        ctx.fillStyle = trial.timeoutWarningColor;
        ctx.textAlign = "center";
        ctx.fillText(
          trial.timeoutWarningText,
          gambleCanvas.width / 2,
          gambleCanvas.height / 2
        );
        jsPsych.pluginAPI.setTimeout(function () {
          // move on to the next trial
          // Remove the canvas as the child of the display_element element
          display_element.innerHTML = "";
          jsPsych.finishTrial(trial_data);
        }, trial.feedbackDuration);
      } else {
        // move on to the next trial
        // Remove the canvas as the child of the display_element element
        display_element.innerHTML = "";
        jsPsych.finishTrial(trial_data);
      }
    };

    // function to handle responses by the subject
    var after_response = function (info) {
      // only record the first response
      if (response.key == null) {
        response = info;
        chosenP = null;
        chosenM = null;
        choice = null;
      }
      if (jsPsych.pluginAPI.compareKeys(response.key, trial.choices[0])) {
        var xposFeedback = left_xpos;
        choice = leftItem;
        chosenP = trial.pL;
        chosenM = trial.mL;
      } else {
        if (jsPsych.pluginAPI.compareKeys(response.key, trial.choices[1])) {
          var xposFeedback = right_xpos;
          choice = 1 - leftItem;
          chosenP = trial.pR;
          chosenM = trial.mR;
        }
      }
      // Feedback Frame
      ctx = gambleCanvas.getContext("2d");
      ctx.strokeStyle = trial.feedbackColor;
      ctx.lineWidth = 5;
      ctx.strokeRect(
        xposFeedback - trial.boxWidth * radius, // frame top left x-coordinate
        (gambleCanvas.height * (1 - trial.boxHeight)) / 2, // frame top left y-coordinate
        2 * trial.boxWidth * radius, // frame width
        gambleCanvas.height * trial.boxHeight // frame height
      );

      // Code before the pause
      // kill any remaining setTimeout handlers
      jsPsych.pluginAPI.clearAllTimeouts();
      setTimeout(function () {
        end_trial();
      }, trial.feedbackDuration);
    };

    // Trial procedure
    ctx = gambleCanvas.getContext("2d");

    // Set up positions
    var left_xpos = gambleCanvas.width * 0.25;
    var right_xpos = gambleCanvas.width * 0.75;
    if (leftItem == 0) {
      var xpos = [left_xpos, right_xpos];
    } else {
      var xpos = [right_xpos, left_xpos];
    }
    if (trial.topAttribute == "probability") {
      var probability_ypos = gambleCanvas.height * 0.75;
      var magnitude_ypos = gambleCanvas.height * 0.25;
    } else {
      var probability_ypos = gambleCanvas.height * 0.25;
      var magnitude_ypos = gambleCanvas.height * 0.75;
    }

    // Set up stimulus properties
    var radius = trial.pieChartRadius * gambleCanvas.width;
    var height = 2 * radius;
    var width = (radius * Math.PI) / 2; // This way, the area of the barplot and the pie chart are identical

    // Draw stimuli
    var i;
    for (i = 0; i < trial.sequence.durations.length; i++) {
      // Draw
      drawStims(i);
      // Wait
      await new Promise((r) => setTimeout(r, trial.sequence.durations[i]));
      // Clear canvas
      ctx.clearRect(0, 0, gambleCanvas.width, gambleCanvas.height);
    }

    for (alt = 0; alt < 2; alt++) {
      ctx.strokeRect(
        xpos[alt] - trial.boxWidth * radius,
        (gambleCanvas.height * (1 - trial.boxHeight)) / 2,
        2 * trial.boxWidth * radius,
        gambleCanvas.height * trial.boxHeight
      );
    }

    // Draw choice prompt
    ctx.font = "60px sans-serif";
    ctx.fillStyle = trial.feedbackColor;
    ctx.textAlign = "center";
    ctx.fillText(
      trial.choicePrompt,
      gambleCanvas.width / 2,
      gambleCanvas.height / 2
    );

    // start the response listener
    if (trial.choices != jsPsych.NO_KEYS) {
      var keyboardListener = jsPsych.pluginAPI.getKeyboardResponse({
        callback_function: after_response,
        valid_responses: trial.choices,
        rt_method: "performance",
        persist: false,
        allow_held_key: false,
      });
    }

    // end trial if choiceTimeout is set
    if (trial.choiceTimeout !== null) {
      jsPsych.pluginAPI.setTimeout(function () {
        end_trial();
      }, trial.choiceTimeout);
    }
  };

  return plugin;
})();
