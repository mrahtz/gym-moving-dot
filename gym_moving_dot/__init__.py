from gym.envs.registration import register

register(
    id='MovingDotDiscreteNoFrameskip-v0',
    entry_point='gym_moving_dot.envs.moving_dot_env:MovingDotDiscreteEnv'
)

register(
    id='MovingDotDiscrete-v0',
    entry_point='gym_moving_dot.envs.moving_dot_env:MovingDotDiscreteEnv'
)

register(
    id='MovingDotContinuousNoFrameskip-v0',
    entry_point='gym_moving_dot.envs.moving_dot_env:MovingDotContinuousEnv'
)

register(
    id='MovingDotContinuous-v0',
    entry_point='gym_moving_dot.envs.moving_dot_env:MovingDotContinuousEnv'
)
