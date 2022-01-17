// Content from settings.js
// Experiment settings
const COMPLETION_CODE = "COMPLETIONCODE"; // This has to be changed to the real completion code of the Prolific study
const TASK_VERSION = "main";

const fixationDuration = 500; // this is in ms
const withinBlockBreakDuration = 30 * 1000; // in ms, too
const betweenBlockBreakDuration = 50 * 1000; // in ms, too
const getReadyDuration = 5 * 1000;
const countdownDuration = 3 * 1000;
var breakAfter = 35; // 140 trials / 4
const choiceTimeout = 3000;
const feedbackColor = "Gold";
const stimBackgroundColor = "DarkRed";
const stimForegroundColor = "ForestGreen";
const stimFrameColor = "#DDD";
var timeline = [];
var trialCounter = 1;
var repeatPractice = false;

// Debugging parameters
const debug_logMessages = false; // false
const debug_nTrialsPerBlock = 70; // 70 (60 exp + 10 catch)
const debug_showValidation = false; // false

// Eye tracking parameters
const calibrationPointSize = 30;
const calibrationPointDuration = 3000;
const calibrationTimeToSaccade = 1000;
const validationPointDuration = 3000;

// Quiz parameters
var nQuestions = 3; // How many questions from the pool to ask
var nCorrectRequired = 3; // How many questions have to be answered correctly
var nCorrect = 0;
var reminder_pages = [];
var questionSample = [];

// conditions are loaded from separate .js file
// practice trials are loaded from separate .js file

// Content of quiz.js
// Define the question pool
// Each question has a prompt, multiple options, a correct answer, and reminder text that is shown if it is not answered correctly.
var questions = [
  {
    prompt: "A lottery's pie chart illustrates its...",
    options: [
      "possible winning amount.",
      "winning probability.",
      "average outcome.",
    ],
    required: true,
    horizontal: false,
    correct: 1,
    reminder: `
    <div style="max-width: 800px">
    <h2><img src="icons/answer-wrong_inv.png", width="50"><br><br>A lottery's pie chart illustrates its...</h2>
    <p style="color:orangered"><strong>Your answer was incorrect!</strong></p>
    <p align="justify">For each lottery, the <strong>pie chart</strong> indicates the probability that this lottery results in a win!</p>
    <p>Press <strong><em>SPACE BAR</em></strong> to continue.</p>
    `,
  },
  {
    prompt: "A lottery's bar chart illustrates its...",
    options: [
      "possible winning amount.",
      "winning probability.",
      "average outcome.",
    ],
    required: true,
    horizontal: false,
    correct: 0,
    reminder: `
    <div style="max-width: 800px">
    <h2><img src="icons/answer-wrong_inv.png", width="50"><br><br>A lottery's bar chart illustrates...</h2>
    <p style="color:orangered"><strong>Your answer was incorrect!</strong></p>
    <p align="justify">For each lottery, the <strong>bar chart</strong> indicates the possible amount that can be won from this lottery. A fully filled bar corresponds to £10.</p>
    <p>Press <strong><em>SPACE BAR</em></strong> to continue.</p>
    `,
  },
  {
    prompt: "The bonus payment is determined...",
    options: [
      "from the number of correct responses.",
      "in a completely random fashion.",
      "by playing out one lottery chosen by the participant.",
    ],
    required: true,
    horizontal: false,
    correct: 2,
    reminder: `
    <div style="max-width: 800px">
    <h2><img src="icons/answer-wrong_inv.png", width="50"><br><br>The bonus payment is determined...</h2>
    <p style="color:orangered"><strong>Your answer was incorrect!</strong></p>
    <p align="justify">The bonus payment is determined by randomly choosing one of the lotteries chosen by the participant over the course of the study. This lottery is then played out, according to its winning probability and amount.</p>
    <p>Press <strong><em>SPACE BAR</em></strong> to continue.</p>
    `,
  },
];

// Draw random questions from the question pool
questionSample = jsPsych.randomization.sampleWithoutReplacement(
  questions,
  nQuestions
);

// Content from instructions.js
taskInstructions = [
  // Page 1: What is a lottery?
  `<div style="max-width: 800px">
<h1><img src="icons/arrows-up_inv.png", width="50"/><br><br>Decision-making task</h1><h2>Lotteries</h2>
<p align="justify">
In this study, you will make choices between two risky lotteries.
Each lottery describes the prospect to win an amount between £1 and £10 with some probability.

<p align='center'><img height="400px" src="img/single-lottery.png"></p>

The possible winning <strong>amount</strong> is illustrated as a filled rectangular bar. The more filled it is, the higher the possible winning <strong>amount</strong>. A fully filled bar represents £10.<br>
The <strong>probability</strong> of winning is illustrated with a pie chart. The more it is filled, the higher the probability of winning this lottery's amount.
<p>Press the <strong><em>SPACE BAR</em></strong> to continue or <strong><em>B</em></strong> to go back.</p>
</div>`,
  // Page 2: Two lotteries!
  `<div style="max-width: 800px">
<h1><img src="icons/arrows-up_inv.png", width="50"/><br><br>Decision-making task</h1><h2>Lotteries</h2>
<p align="justify">
In each trial, you will have to decide between <strong>two lotteries</strong>. The two lotteries can differ in their winning probability and their winning amount.

<p align='center'><img height="400px" src="img/two-lotteries.png"></p>

There is usually no right or wrong choice. Choose the lottery that you personally would prefer to play.</p>
<p>Press the <strong><em>SPACE BAR</em></strong> to continue or <strong><em>B</em></strong> to go back.</p>
</div>`,
  // Page 3: Sequential presentation alternative-wise I
  `<div style="max-width: 800px">
<h1><img src="icons/arrows-up_inv.png", width="50"/><br><br>Decision-making task</h1><h2>Sequential presentation</h2>
<p align='center'><img height="400px" src="img/seq-left.png"></p>
<p align="center">
Information over the two lotteries is shown to you in a <strong>rapid sequential presentation</strong>. For example, you might be shown the left lottery first...</p>
  <p>Press the <strong><em>SPACE BAR</em></strong> to continue or <strong><em>B</em></strong> to go back.</p>
</div>`,
  // Page 4: Sequential presentation alternative-wise II
  `<div style="max-width: 800px">
<h1><img src="icons/arrows-up_inv.png", width="50"/><br><br>Decision-making task</h1><h2>Sequential presentation</h2>
<p align='center'><img height="400px" src="img/seq-right.png"></p>
<p align="center">
... followed by the right lottery, back to the left lottery...</p>
  <p>Press the <strong><em>SPACE BAR</em></strong> to continue or <strong><em>B</em></strong> to go back.</p>
</div>`,
  // Page 5: Sequential presentation Choice prompt
  `<div style="max-width: 800px">
<h1><img src="icons/arrows-up_inv.png", width="50"/><br><br>Decision-making task</h1><h2>Sequential presentation</h2>
<p align='center'><img height="400px" src="img/choiceprompt.png"></p>
<p align="center">
... until you are prompted to make a choice. <strong>You then have ${
    choiceTimeout / 1000
  } seconds</strong> to make a choice using the keyboard, with the <strong>F</strong> and <strong>J</strong> keys.</p>
  <p>Press the <strong><em>SPACE BAR</em></strong> to continue or <strong><em>B</em></strong> to go back.</p>
</div>`,
  // Page 6: Sequential presentation attribute-wise I
  `<div style="max-width: 800px">
<h1><img src="icons/arrows-up_inv.png", width="50"/><br><br>Decision-making task</h1><h2>Sequential presentation</h2>
<p align='center'><img height="400px" src="img/seq-probabilities.png"></p>
<p align="center">
In other trials, the lotteries' attributes are presented sequentially. For example, you would see both lotteries' probability first...</p>
  <p>Press the <strong><em>SPACE BAR</em></strong> to continue or <strong><em>B</em></strong> to go back.</p>
</div>`,
  // Page 7: Sequential presentation attribute-wise II
  `<div style="max-width: 800px">
<h1><img src="icons/arrows-up_inv.png", width="50"/><br><br>Decision-making task</h1><h2>Sequential presentation</h2>
<p align='center'><img height="400px" src="img/seq-amounts.png"></p>
<p align="center">... then their winning amounts...</p>
  <p>Press the <strong><em>SPACE BAR</em></strong> to continue or <strong><em>B</em></strong> to go back.</p>
</div>`,
  // Page 8: Choice prompt again
  `<div style="max-width: 800px">
<h1><img src="icons/arrows-up_inv.png", width="50"/><br><br>Decision-making task</h1><h2>Choice</h2>
<p align='center'><img height="400px" src="img/choiceprompt.png"></p>
<p align="center">
... until you are prompted to make a choice. Again, <strong>you then have ${
    choiceTimeout / 1000
  } seconds</strong> to make a choice using the keyboard.</p>
  <p>Press the <strong><em>SPACE BAR</em></strong> to continue or <strong><em>B</em></strong> to go back.</p>
</div>`,
  // Page 9: Realization
  `<div style="max-width: 800px">
<h1><img src="icons/arrows-up_inv.png", width="50"/><br><br>Decision-making task</h1><h2>Bonus payment</h2>
<p align='center'><img height="400px" src="img/realization.png"></p>
<p align="justify">After you completed a total of 140 choice trials (with multiple breaks in between), the computer will randomly determine one of the lotteries you chose and play it out – according to its winning probability and amount – as a <strong>bonus payment</strong>. Treat every choice as if it was used to determine your bonus payment!</p>
<p>Press the <strong><em>SPACE BAR</em></strong> to practice the task or <strong><em>B</em></strong> to go back.</p>
`,
];

// capture info from Prolific
var subject_id = jsPsych.data.getURLVariable("PROLIFIC_PID");
var study_id = jsPsych.data.getURLVariable("STUDY_ID");
var session_id = jsPsych.data.getURLVariable("SESSION_ID");

jsPsych.data.addProperties({
  task_version: TASK_VERSION,
  subject_id: subject_id,
  study_id: study_id,
  session_id: session_id,
  screen_width: screen.width,
  screen_height: screen.height,
});

// Counterbalance order of blocks (CONDITION is set by cognition.run to 1 or 2). If it is not, set it manually to 1
if (typeof CONDITION === "undefined") {
  // the variable is undefined
  var CONDITION = 1;
}
if (CONDITION == 1) {
  conditions_block_1 = conditions.filter(function (trial) {
    return trial.block == 0;
  });
  conditions_block_2 = conditions.filter(function (trial) {
    return trial.block == 1;
  });
} else {
  conditions_block_1 = conditions.filter(function (trial) {
    return trial.block == 1;
  });
  conditions_block_2 = conditions.filter(function (trial) {
    return trial.block == 0;
  });
}

// Preload images
var preload = {
  type: "preload",
  auto_preload: true,
  images: [
    "img/camera-setup-tips.png",
    "img/realization.png",
    "img/seq-left.png",
    "img/seq-right.png",
    "img/two-lotteries.png",
    "img/choiceprompt.png",
    "img/seq-amounts.png",
    "img/seq-probabilities.png",
    "img/single-lottery.png",
    "icons/answer-wrong_inv.png",
    "icons/eye-scanning_inv.png",
    "icons/race_inv.png",
    "icons/tea-hot_inv.png",
    "icons/wheel-of-fortune_inv.png",
    "icons/arrows-up_inv.png",
    "icons/how-to_inv.png",
    "icons/target_inv.png",
    "icons/timer_inv.png",
  ],
};

// Define post-experiment questions
var postExpQuestions = {
  type: "survey-multi-choice",
  questions: [
    {
      prompt: "Choose your gender.",
      name: "gender",
      options: ["Female", "Male", "Other", "Prefer not to say"],
      required: true,
    },
    {
      prompt: "Do you have red-green color blindness?",
      name: "redGreenColorBlind",
      options: ["yes", "no"],
      required: true,
    },
    {
      prompt:
        "Did you have difficulties identifying the red and green portions of the lotteries?",
      name: "redGreenDifficulties",
      options: ["yes", "no"],
      required: true,
    },
    {
      prompt:
        "To ensure data quality, it would be very helpful if you could tell us whether you have taken part seriously.<br><strong>Your response will not affect your payment or result in any other penalty or loss of benefits.</strong>",
      name: "seriousness",
      options: [
        "I have taken part seriously.",
        "I have not taken part seriously.",
      ],
      required: true,
    },
  ],
  on_start: () => (document.body.style.cursor = "auto"),
};

var postExpSelfReport = {
  type: "survey-text",
  questions: [
    {
      prompt: "Enter your age (in years)",
      rows: 1,
      columns: 3,
      name: "age",
      required: true,
    },
    {
      prompt:
        "Please briefly describe how you made your choices in the task.<br>(One to three sentences or bullet points)",
      rows: 5,
      columns: 40,
      name: "selfReport",
      required: true,
    },
    {
      prompt: "If you want, you can share comments and thoughts on the study.",
      rows: 5,
      columns: 40,
      name: "comments",
      required: false,
    },
  ],
  on_start: () => (document.body.style.cursor = "auto"),
};

/* Welcome */
var welcome = {
  type: "html-keyboard-response",
  stimulus: `
        <div style="max-width: 800px">
        <h1><img src="icons/arrows-up_inv.png", width="50"/><br><br>Welcome</h1>
        <p align="justify">Thank you for choosing to participate in this study. Our aim is to better understand how people make decisions.<br>
        We are also interested in people's looking behaviour during decision making, therefore this study includes <strong>webcam-based eye tracking</strong>. Please <strong>allow this page to access your webcam, when prompted</strong>. Your video feed is processed locally and is not transmitted.
        The study will take around 30 minutes to complete. Please assure that you can be focused and undisturbed for this time.
        To make sure the study runs smoothly, please</p>
        <ul>
        <li style="text-align: left;"><strong>close any unnecessary programs</strong></li>
        <li style="text-align: left;"><strong>close any other browser tabs </strong>that could produce<strong> popups </strong>or<strong> alerts</strong></li>
        <li style="text-align: left;"><strong>mute </strong>your <strong>cell phone</strong> and make sure that it will not distract you</li>
        </ul>
        <p>Press <strong><em>SPACE BAR</em></strong> to start.</p>
        </div>`,
  choices: [" "],
};

/* Initial Eye tracking setup */
var init_camera = {
  type: "webgazer-init-camera",
  instructions: `
      <div style="max-width: 800px">
      <h1><img src="icons/eye-scanning_inv.png", width="50"/><br><br>Camera Setup</h1>
      <p align="justify">Position your head so that the webcam has a good view of your eyes. Use the video above as a guide. Center your face in the box and look directly towards the camera. The blue dot mask should be stable and flush on your mouth and eyes.
      <p align='center'><strong>It is important that you try and keep your head reasonably still throughout the experiment, so please take a moment to adjust your setup as needed. Use these tips to achieve a good camera setup:</strong></p>
      <img width="700px" src="img/camera-setup-tips.png">
      <p>When your face is centered in the box and the box turns green, you can click to continue.</p>
    `,
  on_start: () => (document.body.style.cursor = "auto"),
};

/* Go fullscreen after asking for camera permission */
var goFullScreen = {
  type: "fullscreen",
  message: `
        <div style="max-width: 800px">
        <h1><img src="icons/arrows-up_inv.png", width="50"/><br><br>Full screen</h1>
        <p align="justify">The study needs to run in full screen mode. Click the button below to switch to full screen mode. <strong>Please do not leave full screen mode.</strong>
        </div>`,
  fullscreen_mode: true,
  button_label: "Switch to fullscreen and continue",
};

var calibration_instructions = {
  type: "html-keyboard-response",
  stimulus: `
      <div style="max-width: 800px">
      <h1><img src="icons/eye-scanning_inv.png", width="50"/><br><br>Calibration</h1>
      <p>Great! Now the eye tracker will be calibrated to translate the image of your eyes from the webcam to a location on your screen.
      <p>To do this, you need to look at a series of dots.</p>
      <p>Keep your head still, and focus your gaze on each dot as it appears.</p>
      <p>Press <strong><em>SPACE BAR</em></strong> to start.</p>
    `,
  choices: [" "],
  post_trial_gap: 1000,
};

var recalibration_instructions = {
  type: "html-keyboard-response",
  stimulus: `
      <div style="max-width: 800px">
      <h1><img src="icons/eye-scanning_inv.png", width="50"/><br><br>Re-Calibration</h1>
      <p>We quickly need to re-calibrate the eye tracker.
      <p>Please keep your head still, and focus your gaze on each dot as it appears.</p>
      <p>Press <strong><em>SPACE BAR</em></strong> to start.</p>
    `,
  choices: [" "],
  post_trial_gap: 1000,
};

var calibration = {
  type: "webgazer-calibrate-fm",
  calibration_points: [
    [10, 10],
    [10, 50],
    [10, 90],
    [50, 10],
    [25, 25],
    [25, 75],
    [75, 25],
    [75, 75],
    [50, 50],
    [50, 90],
    [90, 10],
    [90, 50],
    [90, 90],
  ],
  randomize_calibration_order: true,
  calibration_mode: "view",
  point_background_color: "white",
  point_size: calibrationPointSize,
  time_per_point: calibrationPointDuration,
  time_to_saccade: calibrationTimeToSaccade,
  on_start: () => (document.body.style.cursor = "none"),
};

var validation_instruction = {
  type: "html-keyboard-response",
  stimulus: `
      <div style="max-width: 800px">
      <h1><img src="icons/eye-scanning_inv.png", width="50"/><br><br>Validation</h1>
      <p>Good job! To validate the calibration, please focus on the yellow dots now.
      <p>Please keep your head still, and focus your gaze on each dot as it appears.</p>
      <p>Press <strong><em>SPACE BAR</em></strong> to start.</p>
    `,
  choices: [" "],
  post_trial_gap: 1000,
};

var validation = {
  type: "webgazer-validate-fm",
  validation_points: [
    [25, 25],
    [25, 75],
    [75, 25],
    [75, 75],
    [50, 50],
  ],
  validation_point_coordinates: "percent",
  randomize_validation_order: true,
  show_validation_data: debug_showValidation,
  point_background_color: "gold",
  point_size: calibrationPointSize,
  validation_duration: validationPointDuration,
  time_to_saccade: calibrationTimeToSaccade,
  // on_start: () => (document.body.style.cursor = "none"),
};

/* Sequential presentation gamble choice task */
var seqGambleInstruction = {
  type: "instructions",
  pages: taskInstructions,
  post_trial_gap: 1000,
  key_forward: " ",
  key_backward: "b",
  on_start: () => (document.body.style.cursor = "none"),
};

/* Comprehension quiz */
var quiz = {
  type: "survey-multi-choice",
  preamble: `
        <div style="max-width: 800px">
        <h1><img src="icons/how-to_inv.png", width="50"><br><br>Comprehension check</h1>
        <p>Please answer these questions to make sure that you understood the instructions.</p>
        </div>`,
  questions: function () {
    return questionSample;
  },
  randomize_question_order: false,
  button_label: "Submit answers",
  on_start: function () {
    document.body.style.cursor = "auto";
    nCorrect = 0;
  },
  on_finish: function (data) {
    data.questions = questionSample; // Save questions that were asked

    // Count correct responses
    var responses = data.response;
    var i;
    for (i = 0; i < questionSample.length; i++) {
      var response = responses["Q" + i];
      var correct =
        response == questionSample[i]["options"][questionSample[i]["correct"]];
      if (correct) {
        nCorrect++;
      } else {
        // Add reminders for wrong answers
        reminder_pages.push(questionSample[i]["reminder"]);
      }
    }
    // Draw a new sample of questions
    questionSample = jsPsych.randomization.sampleWithoutReplacement(
      questions,
      nQuestions
    );
  },
};

// Show reminders for questions with wrong answers.
reminders = {
  type: "instructions",
  show_page_number: true,
  page_label: "Reminder",
  key_forward: " ",
  pages: function () {
    return reminder_pages;
  },
  on_finish: function () {
    reminder_pages = [];
  },
  on_start: () => (document.body.style.cursor = "none"),
};

// Put the quiz procedure together
seqGambleQuiz = {
  timeline: [
    quiz,
    {
      timeline: [reminders],
      conditional_function: function () {
        return nCorrect < nCorrectRequired;
      },
    },
  ],
  loop_function: function () {
    return nCorrect < nCorrectRequired;
  },
};

var fixation = {
  type: "html-keyboard-response",
  stimulus: '<p style="font-size:40px;">+</p>',
  choices: jsPsych.NO_KEYS,
  trial_duration: fixationDuration,
  extensions: [{ type: "webgazer", params: { targets: [] } }],
  on_start: () => (document.body.style.cursor = "none"),
};

/* Sequential gamble presentation task */
var seqGambleChoice = {
  type: "two-gamble-sequence",
  stimulus: function () {
    stim = {
      p0: jsPsych.timelineVariable("p0", true),
      p1: jsPsych.timelineVariable("p1", true),
      m0: jsPsych.timelineVariable("m0", true) / 10,
      m1: jsPsych.timelineVariable("m1", true) / 10,
      phase: jsPsych.timelineVariable("phase", true),
      condition: jsPsych.timelineVariable("condition", true),
    };
    if (debug_logMessages) {
      console.log("Trial counter:", trialCounter);
      console.log("Stimulus:", stim);
    }
    return stim;
  },
  sequence: function () {
    if (debug_logMessages) {
      console.log("Sequence:", jsPsych.timelineVariable("sequence", true));
    }
    sequence = JSON.parse(jsPsych.timelineVariable("sequence", true));
    return sequence;
  },
  choices: ["f", "j"],
  choiceTimeout: choiceTimeout,
  stimBackgroundColor: stimBackgroundColor,
  stimForegroundColor: stimForegroundColor,
  stimFrameColor: stimFrameColor,
  pieChartWidth: 0.075,
  boxWidth: 1.5,
  boxHeight: 0.9,
  feedbackColor: feedbackColor,
  on_start: () => (document.body.style.cursor = "none"),
  on_finish: function (data) {
    if (debug_logMessages) {
      console.log("Trial data:", data);
    }
    trialCounter++;
  },
  extensions: [{ type: "webgazer", params: { targets: [] } }],
};

// Break screen procedure (break + get-ready countdown)
var withinBlockBreak = {
  timeline: [
    {
      type: "html-keyboard-response",
      on_start: () => (document.body.style.cursor = "none"),
      stimulus: `
            <div style="max-width: 800px">
            <h1><img src="icons/tea-hot_inv.png", width="50"/><br><br>Short break</h1>
            <p>Great job! Take a short break to make sure that you are focused for the next set of trials, but try to keep your head in position.</p>
            <p>Press the <strong><em>SPACE BAR</em></strong> when you are ready to continue.</p>
            </div>`,
      choices: [" "],
      trial_duration: withinBlockBreakDuration,
    },
    {
      type: "call-function",
      func: function () {
        start_time = Date.now();
        timer_ticks = setInterval(function () {
          var time_elapsed = Math.floor(Date.now() - start_time);
          var time_remaining = getReadyDuration - time_elapsed;
          if (time_remaining <= countdownDuration) {
            document.querySelector("#timer").innerHTML = `${Math.floor(
              time_remaining / 1000 + 1
            )}`;
            document.querySelector("#timer").style.color = "orangered";
          }
        }, 1000);
      },
    },
    {
      type: "html-keyboard-response",
      on_start: () => (document.body.style.cursor = "none"),
      stimulus: `
            <div style="max-width: 800px">
            <h1><img src="icons/timer_inv.png", width="50"/><br><br><span id="timer">Get ready!</span></h1>
            </div>`,
      choices: null,
      trial_duration: getReadyDuration,
      on_finish: function () {
        clearInterval(timer_ticks);
      },
      post_trial_gap: 2000,
    },
  ],
  conditional_function: function () {
    if ((trialCounter > 1) & ((trialCounter - 1) % breakAfter == 0)) {
      return true;
    } else {
      return false;
    }
  },
};

seqGamblePractice = {
  timeline: [
    {
      type: "html-keyboard-response",
      stimulus: `
            <div style="max-width: 800px">
            <h1><img src="icons/target_inv.png", width="50"/><br><br>Practice</h1>
            <p align="justify">Now you will perform a couple of practice trials that look just like the ones in the main task, but don't count for the possible bonus winning. Practice trials will start with a slower presentation, and gradually become as fast as in the main task.</p>
            Press the <strong>F</strong> key to choose the left lottery and the <strong>J</strong> key to choose the right lottery.<br>
            Place your fingers on these keys now.</p>
            <p>Please re-center your gaze on the "<strong>+</strong>" between trials.</p>
            <h3 style='color:gold'>Note that this task is designed to be very fast-paced and difficult. Try not to be discouraged by this, but focus and attempt to do your best!<\h3>
            <p>Press <strong><em>SPACE BAR</em></strong> to start practice.</p>
            </div>`,
      choices: [" "],
    },
    {
      timeline: [fixation, seqGambleChoice],
      timeline_variables: practice,
      randomize_order: false,
    },
    {
      type: "html-keyboard-response",
      stimulus: `
            <div style="max-width: 800px">
            <h1><img src="icons/target_inv.png", width="50"/><br><br>Practice finished!</h1>
            <p>Good job. You finished the practice trials!</p>
            <p>If everything is clear, press <strong><em>SPACE BAR</em></strong> to start the main task.</p>
            <p>To repeat practice, press <strong><em>R</em></strong>.</p>
            </div>`,
      choices: [" ", "r"],
      on_finish: function (data) {
        if (jsPsych.pluginAPI.compareKeys(data.response, "r")) {
          repeatPractice = true;
        } else {
          repeatPractice = false;
        }
      },
    },
  ],
  loop_function: function () {
    return repeatPractice;
  },
};

seqGambleMainTaskBlock1 = {
  timeline: [
    {
      type: "html-keyboard-response",
      stimulus: `
            <div style="max-width: 800px">
            <h1><img src="icons/arrows-up_inv.png", width="50"/><br><br>Main task</h1>
            <p align="justify">Get ready for the main task. You will perform two blocks of this task with breaks in between. Please remember to keep your head still, and focus on the "<strong>+</strong>" between trials. Good luck!</p>
            <p>Press <strong><em>SPACE BAR</em></strong> to start the main task.</p>
            </div>`,
      choices: [" "],
      post_trial_gap: 2000,
      on_start: function () {
        trialCounter = 1;
      },
    },
    {
      timeline: [withinBlockBreak, fixation, seqGambleChoice],
      timeline_variables: conditions.filter(function (trial) {
        return trial.block == 0;
      }),
      randomize_order: true,
      sample: {
        type: "without-replacement",
        size: debug_nTrialsPerBlock,
      },
    },
  ],
};

var betweenBlocksBreak = {
  timeline: [
    {
      type: "html-keyboard-response",
      on_start: () => (document.body.style.cursor = "none"),
      stimulus: `
            <div style="max-width: 800px">
            <h1><img src="icons/tea-hot_inv.png", width="50"/><br><br>Short break</h1>
            <p>Great job! You have completed the first half of the task. Take a short break to make sure that you are focused for the second half of trials. You can move your head during this break, but try to return it into position before continuing.</p>
            <p>Press the <strong><em>SPACE BAR</em></strong> when you are ready to continue.</p>
            </div>`,
      choices: [" "],
      trial_duration: betweenBlockBreakDuration,
    },
    {
      type: "call-function",
      func: function () {
        start_time = Date.now();
        timer_ticks = setInterval(function () {
          var time_elapsed = Math.floor(Date.now() - start_time);
          var time_remaining = getReadyDuration - time_elapsed;
          if (time_remaining <= countdownDuration) {
            document.querySelector("#timer").innerHTML = `${Math.floor(
              time_remaining / 1000 + 1
            )}`;
            document.querySelector("#timer").style.color = "orangered";
          }
        }, 1000);
      },
    },
    {
      type: "html-keyboard-response",
      on_start: () => (document.body.style.cursor = "none"),
      stimulus: `
            <div style="max-width: 800px">
            <h1><img src="icons/timer_inv.png", width="50"/><br><br><span id="timer">Get ready!</span></h1>
            </div>`,
      choices: null,
      trial_duration: getReadyDuration,
      on_finish: function () {
        clearInterval(timer_ticks);
      },
      post_trial_gap: 2000,
    },
  ],
};

seqGambleMainTaskBlock2 = {
  timeline: [
    {
      type: "html-keyboard-response",
      stimulus: `
            <div style="max-width: 800px">
            <h1><img src="icons/arrows-up_inv.png", width="50"/><br><br>Main task</h1>
            <p align="justify">Get ready for the the second half of the task! Please remember to keep your head still, and focus on the "<strong>+</strong>" between trials. Good luck!</p>
            <p>Press <strong><em>SPACE BAR</em></strong> to start the last block.</p>
            </div>`,
      choices: [" "],
      post_trial_gap: 2000,
      on_start: function () {
        trialCounter = 1;
      },
    },
    {
      timeline: [withinBlockBreak, fixation, seqGambleChoice],
      timeline_variables: conditions.filter(function (trial) {
        return trial.block == 1;
      }),
      randomize_order: true,
      sample: {
        type: "without-replacement",
        size: debug_nTrialsPerBlock,
      },
    },
  ],
};

/* Realization */
var chosenTrial;
var wonAmt;
var winString;
var realization = {
  timeline: [
    {
      type: "call-function",
      func: function () {
        var experimentalTrials = jsPsych.data
          .get()
          .filter({ phase: "experimental" })
          .values();
        chosenTrial = experimentalTrials[0];
        if (debug_logMessages) {
          console.log("Randomly chosen trial:", chosenTrial);
        }
        if (chosenTrial.chosenM != undefined) {
          chosenGambleString = `
                <p align="justify">The computer chose trial number ${
                  Number(chosenTrial.condition.substring(4)) + 1
                } where you chose the lottery with a winning probability of ${Math.round(
            chosenTrial.chosenP * 100
          )}% and possible winning amount of £${(
            chosenTrial.chosenM * 10
          ).toFixed(
            2
          )}. The computer then played this lottery out and determined the result:</p>`;
          var luckyNumber = Math.random();
          if (luckyNumber < chosenTrial.chosenP) {
            wonAmt = chosenTrial.chosenM * 10;
            winString = `<h2 style="color:ForestGreen">Congratulations! You won an additional £${(
              chosenTrial.chosenM * 10
            ).toFixed(2)}!</h2>`;
          } else {
            wonAmt = 0;
            winString = `<h2 style="color:OrangeRed">Unfortunately you did not win a bonus.</h2>`;
          }
        } else {
          chosenGambleString = `<p align="justify">The computer randomly chose a trial where you did not respond in time.</p>`;
          winString = `<h2 style="color:OrangeRed">Unfortunately you did not win a bonus.</h2>`;
        }
        jsPsych.data.addProperties({
          chosenTrial: chosenTrial.condition,
          luckyNumber: luckyNumber,
          wonAmt: wonAmt,
        });
      },
    },
    {
      type: "html-keyboard-response",
      choices: [" "],
      stimulus: function () {
        return (
          `<div style="max-width: 800px">
        <h1><img src="icons/wheel-of-fortune_inv.png", width="250"/><br><br>Bonus payment</h1>` +
          chosenGambleString +
          winString +
          "<p>Press <strong><em>SPACE BAR</em></strong> to continue.</p>"
        );
      },
    },
  ],
};

var debriefing = {
  type: "html-button-response",
  choices: ["CSV", "JSON", "No thanks."],
  stimulus: `
        <div style="max-width: 800px">
        <h1><img src="icons/race_inv.png", width="50"/><br><br>Study completed!</h1>
        <p align="justify">Thank you for your participation! In this study, we were specifically interested in how the order and duration of information presentation influences choices.<br>
        If you'd like to download a copy of the data to explore, click the format you prefer below.<br>
        You will then be redirected back to Prolific, where this study will be marked as completed.</p>`,
  on_start: () => (document.body.style.cursor = "auto"),
  on_finish: function (data) {
    if (data.response == 0) {
      console.log("Data download request as .csv ");
      jsPsych.data.get().localSave("csv", "decision-experiment.csv");
    }
    if (data.response == 1) {
      console.log("Data download request as .json");
      jsPsych.data.get().localSave("json", "decision-experiment.json");
    }
  },
};

jsPsych.init({
  timeline: [
    preload,
    welcome,
    init_camera,
    goFullScreen,
    seqGambleInstruction,
    seqGambleQuiz,
    seqGamblePractice,
    calibration_instructions,
    calibration,
    validation_instruction,
    validation,
    seqGambleMainTaskBlock1,
    betweenBlocksBreak,
    recalibration_instructions,
    calibration,
    validation_instruction,
    validation,
    seqGambleMainTaskBlock2,
    realization,
    postExpSelfReport,
    postExpQuestions,
    debriefing,
  ],
  on_finish: function () {
    window.location =
      "https://app.prolific.co/submissions/complete?cc=" + COMPLETION_CODE;
  },
  extensions: [{ type: "webgazer" }],
});
