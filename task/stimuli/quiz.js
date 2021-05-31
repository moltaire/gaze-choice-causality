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
