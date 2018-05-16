import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

class MoveToBeaconVEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.size = 10
        self.action_space = spaces.Discrete(self.size)
        self.observation_space = spaces.Discrete(self.size)

        self._seed()
        self.viewer = None
        self.reset()
        print('realgoal', self.realgoal)

    def randomizeCorrect(self):
        self.realgoal = np.random.randint(0, self.size)
        print("new goal is " + str(self.realgoal))

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        print("seeded")
        return [seed]

    def step(self, action):
        # print('chose ', action)
        if action == self.realgoal:
            return self.obs(), 1, False, {}
        else:
            return self.obs(), 0, False, {}

    def obs(self):
        x = []
        for i in range(self.size):
            if i == self.realgoal:
                x.append(1)
            else:
                x.append(0)
        return np.array(x)

    def reset(self):
        self.randomizeCorrect()
        return self.obs()

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
