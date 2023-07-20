# Trading bot using Reinforcement Learning:

## Overview:

This project implements a Stock/Currency Trading Bot, trained using Deep Reinforcement Learning, specifically Deep Q-Network. Implementation is kept simple for learning purposes.

## Introduction:

Generally, Reinforcement Learning is a family of machine learning techniques that allow us to create intelligent agents that learn from the environment by interacting with it, as they learn an optimal policy by trial and error. This is especially useful in many real world tasks where supervised learning might not be the best approach due to various reasons like nature of task itself, lack of appropriate labelled data, etc.

The important idea here is that this technique can be applied to any real world task that can be described loosely as a Markovian process (MDP).

## Approaches:

This work uses a Model-free Reinforcement Learning technique called Deep Q-Learning (neural variant of Q-Learning).
At any given time (episode), an agent abserves it's current state (n-day window stock price representation), selects and performs an action (buy/sell/hold), observes a subsequent state, receives some reward signal (difference in portfolio position) and lastly adjusts it's parameters based on the gradient of the loss computed.

There have been several improvements to the Q-learning algorithm over the years, and a few have been implemented in this project:

- [x] Naive DQN
- [x] Enhanced DQN (DQN with changed target distribution)

## Results:

Trained on `GOOG` 2020-2022 stock data, tested on 2022-2023 with a profit of +$109.63 (validated on every last 100 days with profit more than $24):

![Google Stock Trading episode](./extra/GOOGLE_Stock_+$109.63_(DQN_40_ep).png)

Trained on `APPLE` 2020-2022 stock data, tested on 2022-2023 with a profit of +$442.50 (validated on every last 100 days with profit more than $20):

![APPLE Stock Trading episode](./extra/APPLE_Stock_+$442.50_(DQN_30_ep).png)

Trained on `BIT` 2020-2022 crypto-currency data, tested on 2022-2023 with a profit of -$63157.24 (validated on every last 100 days with profit more than $8000):

![BITCOIN Crypto-Currency Trading episode](./extra/BITCOIN_Currency_-$63157.24_(T-DQN_30_ep_&_1000_iter_to_update_T-Net).png)

Trained on `ETH` 2020-2022 crypto-currency data, tested on 2022-2023 with a profit of +$1267.51 (validated on every last 100 days with profit more than $2000):

![ETHERIUM Crypto-Currency Trading episode](./extra/ETHERIUM_Currency_+$1267.51_(T-DQN_20_ep_&_100_iter_to_update_T-Net).png)

## Some Caveats:

- At any given state, the agent can only decide to buy/sell one stock at a time. This is done to keep things as simple as possible as the problem of deciding how much stock to buy/sell is one of portfolio redistribution.
- The n-day window feature representation is a vector of subsequent differences in Adjusted Closing price of the stock we're trading followed by a sigmoid operation, done in order to normalize the values to the range [0, 1].
- Training is prefferably done on CPU due to it's sequential manner, after each episode of trading we replay the experience (1 epoch over a small minibatch) and update model parameters.

## Data:

You can download Historical Financial data from [Yahoo! Finance](https://ca.finance.yahoo.com/) for training, or even use some sample datasets already present under `data/`.

## Getting Started:

In order to use this project, you'll need to install the required python packages: [requirements](requirements_1.txt)

## Demo:

You can check the web app of that project in [Trading Bot App](https://trading-bot.streamlit.app/)
