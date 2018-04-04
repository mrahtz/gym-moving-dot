# gym-moving-dot

A simple environment for OpenAI gym consisting of a white dot moving around in
a black square, designed as a simple test environment for reinforcement
learning experiments.

![](screenshot.gif)

Observations are given as 210 x 160 pixel image with 3 channels for red, green
and blue; the same size as Atari environments. The white dot has pixel values
(255, 255, 255), while the black square has pixel values (0, 0, 0).

Possible actions are:
* 0: do nothing
* 1: move down
* 2: move right
* 3: move up
* 4: move left

Rewards are given based on how far the dot is from the centre.
* If the dot moves closer to the centre, it receives reward +1.
* If the dot moves further away from the centre, it receives reward -1.
* If the dot sames the same distance from the centre, it receives reward 0.

The episode terminates after a given number of steps have been taken (by
default 1,000). If `env.random_start` is set to True (the default), the dot
starts in a different position at the start of each episode. Otherwise, the dot
starts at the top left corner.

Training with actor-critic (A2C from OpenAI's baselines with one worker) takes
about five minutes.  Expect your graphs to look something like:

![](training.png)

## Installation

`pip install --user git+https://github.com/mrahtz/gym-moving-dot`

## Usage

```
import gym_moving_dot

env = gym.make("MovingDot-v0")
# A synonym if you need to use with a wrapper that checks for NoFrameskip
# (e.g. wrap_deepmind)
env = gym.make("MovingDotNoFrameskip-v0")

# Adjust number of steps before termination
env.max_steps = 2000
# Adjust random start
env.random_start = False
```
