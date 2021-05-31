/**
 * jspsych-two-gamble-choice
 * Felix Molter
 *
 * a jsPsych plugin for displaying two all-or-nothing gamble stimuli, described by a pie chart and a bar chart, and getting a keyboard response
 *
 **/

jsPsych.plugins["two-gamble-choice"] = (function () {
  var plugin = {};

  plugin.info = {
    name: "two-gamble-choice",
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
      trial_duration: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: "Trial duration",
        default: null,
        description: "How long to show trial before it ends.",
      },
      top_attribute: {
        type: jsPsych.plugins.parameterType.STR,
        pretty_name: "Top attribute.",
        default: "probability",
        description:
          "Which attribute ('probability' or 'outcome') is displayed in the top row.",
      },
      feedback_duration: {
        type: jsPsych.plugins.parameterType.INT,
        pretty_name: "Feedback duration.",
        default: 500,
        description: "Duration for which the feedback frame is shown (ms).",
      },
      doEyeTracking: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: "eye-tracking",
        default: true,
        description: "Whether to do the eye tracking during this trial.",
      },
      showPredictionPoints: {
        type: jsPsych.plugins.parameterType.BOOL,
        pretty_name: "webgazer-prediction-point",
        default: true,
        description: "Whether to show the current webgazer-prediction.",
      },
    },
  };

  plugin.trial = function (display_element, trial) {
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

    //--------Set up Canvas start-------
    var gambleCanvas = document.createElement("canvas");
    gambleCanvas.width = 0.95 * screen.width;
    gambleCanvas.height = 0.95 * screen.height;
    display_element.appendChild(gambleCanvas);

    function drawPiechart(
      ctx,
      p,
      x,
      y,
      radius,
      fillColor = "ForestGreen",
      backColor = "FireBrick"
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
      fillColor = "ForestGreen",
      backColor = "FireBrick"
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

    var drawStims = function () {
      // Left gamble
      // Frame
      ctx.strokeRect(
        left_xpos - 1.5 * radius,
        gambleCanvas.height * 0.1,
        3 * radius,
        gambleCanvas.height * 0.8
      );
      // -- Probability
      drawPiechart(
        ctx,
        trial.pL,
        left_xpos,
        probability_ypos,
        radius,
        "ForestGreen",
        "FireBrick"
      );
      // -- Magnitude
      drawBarchart(
        ctx,
        trial.mL,
        left_xpos,
        magnitude_ypos,
        width,
        height,
        "ForestGreen",
        "FireBrick"
      );
      // Right gamble
      // Frame
      ctx.strokeRect(
        right_xpos - 1.5 * radius,
        gambleCanvas.height * 0.1,
        3 * radius,
        gambleCanvas.height * 0.8
      );

      // -- Probability
      drawPiechart(
        ctx,
        trial.pR,
        right_xpos,
        probability_ypos,
        radius,
        "ForestGreen",
        "FireBrick"
      );
      // -- Magnitude
      drawBarchart(
        ctx,
        trial.mR,
        right_xpos,
        magnitude_ypos,
        width,
        height,
        "ForestGreen",
        "FireBrick"
      );
    };

    // function to end trial when it is time
    var end_trial = function () {
      if (trial.doEyeTracking) {
        webgazer.showPredictionPoints(false);
        webgazer.pause();
        clearInterval(eye_tracking_interval);
      }

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
        eyeData: JSON.stringify(eyeData),
      };

      // clear the display
      display_element.innerHTML = "";

      // move on to the next trial
      jsPsych.finishTrial(trial_data);
    };

    // function to handle responses by the subject
    var after_response = function (info) {
      // only record the first response
      if (response.key == null) {
        response = info;
      }
      if (String.fromCharCode(response.key) == trial.choices[0]) {
        var xpos = left_xpos;
      } else {
        if (String.fromCharCode(response.key) == trial.choices[1]) {
          var xpos = right_xpos;
        }
      }
      // Feedback Frame
      ctx = gambleCanvas.getContext("2d");

      ctx.strokeStyle = "Gold";
      ctx.lineWidth = 5;
      ctx.strokeRect(
        xpos - 1.5 * radius,
        gambleCanvas.height * 0.1,
        3 * radius,
        gambleCanvas.height * 0.8
      );

      // code before the pause
      setTimeout(function () {
        end_trial();
      }, trial.feedback_duration);
    };

    // Trial procedure

    ctx = gambleCanvas.getContext("2d");

    // Set up positions
    left_xpos = gambleCanvas.width * 0.25;
    var right_xpos = gambleCanvas.width * 0.75;
    if (trial.top_attribute == "probability") {
      var probability_ypos = gambleCanvas.height * 0.75;
      var magnitude_ypos = gambleCanvas.height * 0.25;
    } else {
      var probability_ypos = gambleCanvas.height * 0.25;
      var magnitude_ypos = gambleCanvas.height * 0.75;
    }

    // Set up stimulus properties
    var radius = gambleCanvas.width / 15;
    var width = (radius * Math.PI) / 2;
    var height = 2 * radius;

    // Initialize eye movement data
    var eyeData = { history: [] };

    // Draw stimuli
    drawStims();

    // Start eyetracking
    if (trial.doEyeTracking) {
      webgazer.resume();
      webgazer.showVideo(false);
      webgazer.showPredictionPoints(trial.showPredictionPoints);
      webgazer.showFaceOverlay(false);
      webgazer.showFaceFeedbackBox(false);
      var starttime = performance.now();
      var eye_tracking_interval = setInterval(function () {
        var pos = webgazer.getCurrentPrediction();
        if (pos) {
          var relativePosX = pos.x / screen.width;
          var relativePosY = pos.y / screen.height;
          eyeData.history.push({
            "relative-x": relativePosX,
            "relative-y": relativePosY,
            "elapse-time": performance.now() - starttime,
          });
        }
      }, 20);
    }

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

    // end trial if trial_duration is set
    if (trial.trial_duration !== null) {
      jsPsych.pluginAPI.setTimeout(function () {
        end_trial();
      }, trial.trial_duration);
    }
  };

  return plugin;
})();
