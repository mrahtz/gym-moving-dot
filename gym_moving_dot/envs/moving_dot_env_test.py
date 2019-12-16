"""
Test all envs implemented over small number of steps
"""

import gym
import gym_moving_dot

ENVS = ["MovingDotDiscrete-v0",
        "MovingDotDiscreteNoFrameskip-v0",
        "MovingDotContinuous-v0",
        "MovingDotContinuousNoFrameskip-v0"]

for env_name in ENVS:
    print("=== Test: {} ===".format(env_name))

    env = gym.make(env_name)
    env.random_start = False

    env.reset()

    for i in range(3):
        a = env.action_space.sample()
        o, r, d, info = env.step(a)
        print("Obs shape: {}, Action: {}, Reward: {}, Done flag: {}, Info: {}".format(o.shape, a, r, d, info))

    env.close()
    del env
