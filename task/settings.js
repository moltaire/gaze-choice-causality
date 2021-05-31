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
