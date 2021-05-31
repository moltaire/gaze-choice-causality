taskInstructions = [
  // Page 1: What is a lottery?
  `<div style="max-width: 800px">
<h1><img src="icons/arrows-up_inv.png", width="50"/><br><br>Decision-making task</h1><h2>Lotteries</h2>
<p align="justify">
In this study, you will make choices between two risky lotteries.
Each lottery describes the prospect to win an amount between £0.1 and £10 with some probability.

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
  } seconds</strong> to make a choice using the keyboard.</p>
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
<p align="justify">After you completed a total of 164 choice trials (with multiple breaks in between), the computer will randomly determine one of the lotteries you chose and play it out – according to its winning probability and amount – as a <strong>bonus payment</strong>. Treat every choice as if it was used to determine your bonus payment!</p>
<p>Press the <strong><em>SPACE BAR</em></strong> to practice the task or <strong><em>B</em></strong> to go back.</p>
`,
];
