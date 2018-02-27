from gym.envs.registration import register

register(
    id='MovingDotNoFrameskip-v0',
    entry_point='gym_moving_dot.envs:MovingDotEnv'
)

register(
    id='MovingDot-v0',
    entry_point='gym_moving_dot.envs:MovingDotEnv'
)
