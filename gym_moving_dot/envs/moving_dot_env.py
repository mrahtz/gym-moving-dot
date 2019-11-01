"""
A simple OpenAI gym environment consisting of a white dot moving in a black
square.
"""

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding


class ALE:
    def __init__(self):
        self.lives = lambda: 0


class MovingDotEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # Environment parameters
        self.dot_size = [2, 2]
        self.random_start = True
        self.max_steps = 1000

        # environment setup
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(210, 160, 3))
        self.centre = np.array([80, 105])
        self.action_space = spaces.Discrete(5)
        self.viewer = None

        self.seed()

        # Needed by atari_wrappers in OpenAI baselines
        self.ale = ALE()
        seed = None
        self.np_random, _ = seeding.np_random(seed)

        self.reset()

    def reset(self):
        if self.random_start:
            x = self.np_random.randint(low=0, high=160)
            y = self.np_random.randint(low=0, high=210)
            self.pos = [x, y]
        else:
            self.pos = [0, 0]
        self.steps = 0
        ob = self._get_ob()
        return ob

    # This is important because for e.g. A3C each worker should be exploring
    # the environment differently, therefore seeds the random number generator
    # of each environment differently. (This influences the random start
    # location.)
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_ob(self):
        ob = np.zeros((210, 160, 3), dtype=np.uint8)
        x = self.pos[0]
        y = self.pos[1]
        w = self.dot_size[0]
        h = self.dot_size[1]
        ob[y-h:y+h, x-w:x+w, :] = 255
        return ob

    def get_action_meanings(self):
        return ['NOOP', 'DOWN', 'RIGHT', 'UP', 'LEFT']

    def step(self, action):
        assert action >= 0 and action <= 4

        prev_pos = self.pos[:]

        if action == 0:
            # NOOP
            pass
        elif action == 1:
            self.pos[1] += 1
        elif action == 2:
            self.pos[0] += 1
        elif action == 3:
            self.pos[1] -= 1
        elif action == 4:
            self.pos[0] -= 1
        self.pos[0] = np.clip(self.pos[0],
                              self.dot_size[0], 159 - self.dot_size[0])
        self.pos[1] = np.clip(self.pos[1],
                              self.dot_size[1], 209 - self.dot_size[1])

        ob = self._get_ob()

        self.steps += 1
        if self.steps < self.max_steps:
            episode_over = False
        else:
            episode_over = True

        dist1 = np.linalg.norm(prev_pos - self.centre)
        dist2 = np.linalg.norm(self.pos - self.centre)
        if dist2 < dist1:
            reward = 1
        elif dist2 == dist1:
            reward = 0
        else:
            reward = -1

        return ob, reward, episode_over, {}

    # Based on gym's atari_env.py
    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        # We only import this here in case we're running on a headless server
        from gym.envs.classic_control import rendering
        assert mode == 'human', "MovingDot only supports human render mode"
        img = self._get_ob()
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        self.viewer.imshow(img)


class MovingDotContinuousEnv(MovingDotEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, low=-1, high=1, moving_thd=0.1):  # moving_thd is empirically determined
        super(MovingDotEnv, self).__init__()
        # Environment parameters
        self.dot_size = [2, 2]
        self.random_start = True
        self.max_steps = 1000

        # environment setup
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(210, 160, 3),
                                            dtype=np.uint8)
        self.centre = np.array([80, 105])
        # specify the amount of movement along with x/y axes
        self._high = high
        self._low = low
        self._moving_thd = moving_thd  # used to decide if the object has to move, see step func below.
        self.action_space = spaces.Box(low=low, high=high, shape=(2,), dtype=np.float32)
        self.viewer = None

        self.seed()

        # Needed by atari_wrappers in OpenAI baselines
        self.ale = ALE()
        seed = None
        self.np_random, _ = seeding.np_random(seed)

        self.reset()

    def step(self, action):
        _x, _y = action
        assert self._low <= _x <= self._high, "movement along x-axis has to fall in between -1 to 1"
        assert self._low <= _y <= self._high, "movement along y-axis has to fall in between -1 to 1"

        prev_pos = self.pos[:]

        """
        [Note]
        Since the action values are continuous for each x/y pos,
        we round the position of the object after executing the action on the 2D space.
        TODO: Is this okay?
        """
        new_x = self.pos[0] + 1 if _x >= self._moving_thd else self.pos[0] - 1
        new_y = self.pos[1] + 1 if _y >= self._moving_thd else self.pos[1] - 1

        self.pos[0] = np.clip(new_x,
                              self.dot_size[0], 159 - self.dot_size[0])
        self.pos[1] = np.clip(new_y,
                              self.dot_size[1], 209 - self.dot_size[1])

        ob = self._get_ob()

        self.steps += 1
        if self.steps < self.max_steps:
            episode_over = False
        else:
            episode_over = True

        dist1 = np.linalg.norm(prev_pos - self.centre)
        dist2 = np.linalg.norm(self.pos - self.centre)
        if dist2 < dist1:
            reward = 1
        elif dist2 == dist1:
            reward = 0
        else:
            reward = -1

        return ob, reward, episode_over, {}
