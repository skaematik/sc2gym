"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

class MovementBanditsMoveToBeaconResetA2C(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        # new action space = [left, right]
        self.numgoals = 1
        self.screen_size = 400
        self.action_size = 10
        self.action_space = spaces.Discrete(self.action_size*self.action_size)
        self.observation_space = spaces.Box(-10000000, 10000000, shape=(2,2))

        self._seed()
        self.viewer = None
        self.reset()

        self.x_clicked = self.screen_size/2
        self.y_clicked = self.screen_size/2

        self.realgoal = 0
        self.randomizeCorrect()

        self.steps_beyond_done = None

        # Just need to initialize the relevant attributes
        self._configure()

    def randomizeCorrect(self):
        self.realgoal = self.np_random.randint(0,self.numgoals)
        # print("new goal is " + str(self.realgoal))

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        print("seeded")
        return [seed]


    def _translate_action_1d_to_2d(self, action):
        max_action_number = self.action_size*self.action_size
        assert(action >= 0 or action < max_action_number)
        y = math.floor(action / self.action_size)
        x = action - (y * self.action_size)
        size_of_each_sq = self.screen_size/self.action_size
        half_length = size_of_each_sq / 2
        ret = [x * size_of_each_sq + half_length,
                y * size_of_each_sq + half_length]
        return ret

    def _step(self, action):
        # print('action ' , action)

        [x,y] = self._translate_action_1d_to_2d(action)
        self.x_clicked = x
        self.y_clicked = y

        # print('xy ' , x,y)
        # print('self.state ' , self.state[0], self.state[1])
        dist_to_move_each_timestep = 30
        dist_to_command_location = math.sqrt((self.state[1] - y)**2 + (self.state[0] - x)**2)
        if dist_to_command_location < dist_to_move_each_timestep:
            self.state = [x,y]
        else:
            x_offset = dist_to_move_each_timestep*math.cos(math.atan2(abs(y-self.state[1]),abs(x-self.state[0])))
            y_offset = dist_to_move_each_timestep*math.sin(math.atan2(abs(y-self.state[1]),abs(x-self.state[0])))
            # x_offset = (self.state[0] / dist_to_command_location) * dist_to_move_each_timestep
            # y_offset = (self.state[1] / dist_to_command_location) * dist_to_move_each_timestep
            # print(math.sqrt(x_offset**2 + y_offset**2))
            if x < self.state[0]:
                self.state[0] -= x_offset
            else:
                self.state[0] += x_offset
            if y < self.state[1]:
                self.state[1] -= y_offset
            else:
                self.state[1] += y_offset

        # print('self.offset ' , x_offset, y_offset)
        # print('self.state ' , self.state[0], self.state[1])
        distance = np.mean(
            abs(self.state[0] - self.goals[self.realgoal][0])**2 +
            abs(self.state[1] - self.goals[self.realgoal][1])**2
        )

        # reward = -distance / 5000
        # print(distance)

        if distance < 2000:
            reward = 1
            self._reset_goal()
        else:
            reward = 0

        return self.obs(), reward, False, {}

    def obs(self):
        return np.reshape(np.array([self.state] + self.goals), (2,2,-1)) / 400

    def _reset_state(self):
        self.state = self.np_random.uniform(0, 400, size=(2,))  # [200.0, 200.0]

    def _reset_goal(self):
        self.goals = []
        for x in range(self.numgoals):
            self.goals.append(self.np_random.uniform(0, 400, size=(2,)))

    def _reset(self):
        # self.randomizeCorrect()
        self._reset_goal()
        self._reset_state()
            # self.goals.append(np.array([300, 200]))

        # self.goals.append(np.array([300, 300]))
        # self.goals.append(np.array([100, 100]))
        return self.obs()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 400
        screen_height = 400


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height, display=self.display)
            self.man_trans = rendering.Transform()
            self.man = rendering.make_circle(10)
            self.man.add_attr(self.man_trans)
            self.man.set_color(1,0,0)
            self.viewer.add_geom(self.man)

            self.click_trans = rendering.Transform()
            self.click = rendering.make_circle(10)
            self.click.add_attr(self.click_trans)
            self.click.set_color(.5,.8,.5)
            self.viewer.add_geom(self.click)

            self.goal_trans = []
            for g in range(len(self.goals)):
                self.goal_trans.append(rendering.Transform())
                self.goal = rendering.make_circle(20)
                self.goal.add_attr(self.goal_trans[g])
                self.viewer.add_geom(self.goal)
                self.goal.set_color(.5,.5,g*0.8)


        self.click_trans.set_translation(self.x_clicked, self.y_clicked)
        self.man_trans.set_translation(self.state[0], self.state[1])
        for g in range(len(self.goals)):
            self.goal_trans[g].set_translation(self.goals[g][0], self.goals[g][1])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
