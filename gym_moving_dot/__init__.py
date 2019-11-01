from gym.envs.registration import register

register(
    id='MovingDotNoFrameskip-v0',
    entry_point='gym_moving_dot.envs.moving_dot_env:MovingDotEnv'
)

register(
    id='MovingDot-v0',
    entry_point='gym_moving_dot.envs.moving_dot_env:MovingDotEnv'
)

register(
    id='MovingDotContinuousNoFrameskip-v0',
    entry_point='gym_moving_dot.envs.moving_dot_env:MovingDotContinuousEnv'
)

register(
    id='MovingDotContinuous-v0',
    entry_point='gym_moving_dot.envs.moving_dot_env:MovingDotContinuousEnv'
)
