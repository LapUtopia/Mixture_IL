import gym
import matplotlib.pyplot as plt
import numpy as np
import copy
from gym.spaces import Box


class reach_goal(gym.core.Env):
    def __init__(self,
                 size=np.array([5, 5]),
                 target_size=0.5,
                 dt=0.1,
                 mass=1,
                 n_steps=10000,
                 acceleration_lim=np.array([1.0, 1.0]),
                 velocity_lim=np.array([1.0, 1.0]),
                 obstacle_num=3,
                 ):
        self.size = size
        self.n_steps = n_steps
        self.target_size = target_size
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.pos = np.zeros(2)
        self.dt = dt
        self.mass = mass
        self.acceleration_lim = acceleration_lim
        self.velocity_lim = velocity_lim
        self.obstacle_num = obstacle_num
        self.trajectory = []
        self.current_steps = 0

        self._observation_space = Box(
            low=-np.ones(8 + obstacle_num * 3),
            high=np.ones(8 + obstacle_num * 3)
        )
        self._action_space = Box(low=-acceleration_lim, high=acceleration_lim)
        self.fig = None
        self.obstacle_size = np.random.randint(15, 30, self.obstacle_num)

    def step(self, acceleration):
        self.current_steps += 1
        self.acceleration = np.clip(acceleration, -self.acceleration_lim, self.acceleration_lim)
        last_velocity = copy.deepcopy(self.velocity)
        self.velocity = np.clip(self.velocity + self.acceleration * self.dt, -self.velocity_lim, self.velocity_lim)

        self.pos += self.velocity * self.dt + 0.5 * (self.velocity - last_velocity) * self.dt

        reward, done, reached = self._calc_reward_done()
        self.trajectory.append(self.pos)

        return self._get_obs(), reward, done, {"reached": reached}

    def _calc_reward_done(self):
        dist_to_obstacles = np.linalg.norm(self.pos - self.obstacle_pos, axis=1)
        done = True
        reached = False
        if ((dist_to_obstacles - self.obstacle_size/50) < 0).sum():
            reward = -10
        elif (np.abs(self.pos) >= self.size).sum():
            reward = -10
        elif np.linalg.norm(self.pos - self.target_pos) < self.target_size:
            reward = 100
            reached = True
        else:
            done = False
            reward = np.linalg.norm(self.velocity) * 0.1
        return reward, done, reached

    def render(self, mode=None):
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(6, 6))
            self.ax = self.fig.add_subplot(111)
            self.hl_target = self.ax.plot([], [], markersize=int(self.target_size * 50), marker="o", color='r')[0]
            self.hl_obstacle = [self.ax.plot([], [], markersize=size, marker="o", color='gray')[0] for size in
                                self.obstacle_size]
            self.hl_agent = self.ax.plot([], [], markersize=10, marker="o", color='b')[0]
            self.hl, = self.ax.plot([], [])
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_title("Agent Trajectory")
        self.hl_target.set_xdata(self.target_pos[0])
        self.hl_target.set_ydata(self.target_pos[1])
        self.hl_agent.set_xdata(self.pos[0])
        self.hl_agent.set_ydata(self.pos[1])
        for i in range(self.obstacle_num):
            self.hl_obstacle[i].set_xdata(self.obstacle_pos[i, 0])
            self.hl_obstacle[i].set_ydata(self.obstacle_pos[i, 1])
        self.hl.set_xdata(np.array(self.trajectory)[:, 0])
        self.hl.set_ydata(np.array(self.trajectory)[:, 1])
        self.ax.set_ylim([-self.size[0], self.size[0]])
        self.ax.set_xlim([-self.size[1], self.size[1]])
        # time.sleep(0.02)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.fig.show()

    def reset(self):
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.pos = np.zeros(2) - 4.5
        self.current_steps = 0
        self.trajectory = []
        self.target_pos = np.clip(np.random.rand(2) * 3, -self.size[1], self.size[0])

        self.obstacle_pos = np.clip(np.random.randn(self.obstacle_num, 2) * 2.35, -self.size, self.size)
        self.obstacle_size = np.random.randint(15, 30, self.obstacle_num)
        if self.fig is not None:
            for obstacle in self.hl_obstacle:
                obstacle.remove()
            self.hl_obstacle = [self.ax.plot([], [], markersize=size, marker="o", color='gray')[0] for size in
                                self.obstacle_size]

        return self._get_obs()

    def _get_obs(self):
        obs = np.hstack([self.pos / self.size,
                         self.velocity,
                         self.acceleration,
                         (self.target_pos - self.pos) / self.size,
                         np.reshape((self.obstacle_pos - self.pos) / self.size, [-1]),
                         self.obstacle_size/50])
        return obs

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space


class reach_goal_fixed_obstacle(gym.core.Env):
    def __init__(self,
                 size=np.array([5, 5]),
                 target_size=0.5,
                 dt=0.1,
                 mass=1,
                 n_steps=10000,
                 acceleration_lim=np.array([1.0, 1.0]),
                 velocity_lim=np.array([1.0, 1.0]),
                 obstacle_num=3,
                 ):
        self.size = size
        self.n_steps = n_steps
        self.target_size = target_size
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.pos = np.zeros(2)
        self.dt = dt
        self.mass = mass
        self.acceleration_lim = acceleration_lim
        self.velocity_lim = velocity_lim
        self.obstacle_num = obstacle_num
        self.trajectory = []
        self.current_steps = 0

        self._observation_space = Box(
            low=-np.ones(8 + obstacle_num * 3),
            high=np.ones(8 + obstacle_num * 3)
        )
        self._action_space = Box(low=-acceleration_lim, high=acceleration_lim)
        self.fig = None
        self.obstacle_size = np.array([30, 30, 30])

    def step(self, acceleration):
        self.current_steps += 1
        # self.obstacle_pos = self.obstacle_pos + np.array([[0.05, 0.05], [0.05, -0.05], [-0.05, 0.05]])
        # self.obstacle_pos -= (self.obstacle_pos - self.pos) * 0.02
        # self.obstacle_pos[0] = self.pos + np.array([1.0, 1.0])
        self.acceleration = np.clip(acceleration, -self.acceleration_lim, self.acceleration_lim)
        last_velocity = copy.deepcopy(self.velocity)
        self.velocity = np.clip(self.velocity + self.acceleration * self.dt, -self.velocity_lim, self.velocity_lim)

        self.pos += self.velocity * self.dt + 0.5 * (self.velocity - last_velocity) * self.dt

        reward, done = self._calc_reward_done()
        self.trajectory[-1].append(copy.deepcopy(self.pos))

        return self._get_obs(), reward, done, {}

    def _calc_reward_done(self):
        dist_to_obstacles = np.linalg.norm(self.pos - self.obstacle_pos, axis=1)
        if ((dist_to_obstacles - self.obstacle_size/50) < 0).sum():
            done = True
            reward = -100
        elif (np.abs(self.pos) >= self.size).sum():
            done = True
            reward = -10
        elif np.linalg.norm(self.pos - self.target_pos) < self.target_size:
            done = True
            reward = 100
        else:
            done = False
            reward = np.linalg.norm(self.velocity) * 0.1
        return reward, done

    def render(self, mode=None):
        if self.fig is None:
            # plt.ion()
            self.fig = plt.figure(figsize=(6, 6))
            self.ax = self.fig.add_subplot(111)
            self.hl_target = self.ax.plot([], [], markersize=int(self.target_size * 50), marker="o", color='r')[0]
            self.hl_obstacle = [self.ax.plot([], [], markersize=size, marker="o", color='gray')[0] for size in
                                self.obstacle_size]
            # self.hl_agent = self.ax.plot([], [], markersize=10, marker="o", color='b')[0]
            self.hl = [self.ax.plot([], [], color='b')[0]]
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_title("Our approach: back-propagation-max", fontsize=20)
        self.hl_target.set_xdata(self.target_pos[0])
        self.hl_target.set_ydata(self.target_pos[1])
        # self.hl_agent.set_xdata(self.pos[0])
        # self.hl_agent.set_ydata(self.pos[1])
        for i in range(self.obstacle_num):
            self.hl_obstacle[i].set_xdata(self.obstacle_pos[i, 0])
            self.hl_obstacle[i].set_ydata(self.obstacle_pos[i, 1])
        for i, data in enumerate(self.trajectory):
            self.hl[i].set_xdata(np.array(data)[:, 0])
            self.hl[i].set_ydata(np.array(data)[:, 1])
        self.ax.set_ylim([-self.size[0], self.size[0]])
        self.ax.set_xlim([-self.size[1], self.size[1]])
        # time.sleep(0.02)
        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()
        # self.fig.show()

    def reset(self):
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.pos = np.zeros(2) - 4.5
        self.current_steps = 0
        self.trajectory.append([])
        self.target_pos = np.zeros(2) + 3.  # np.clip(np.random.rand(2) * 3, -self.size[1], self.size[0])

        self.obstacle_pos = np.array([[-2.5, -2.5], [-3., -0.5], [-0.5, -3.]])
        # self.obstacle_pos = np.array([[-4., -2.], [-1.5, -1.5], [-2., -4.]])
        if self.fig is not None:
            for obstacle in self.hl_obstacle:
                obstacle.remove()
            self.hl_obstacle = [self.ax.plot([], [], markersize=size, marker="o", color='gray')[0] for size in
                                self.obstacle_size]
            self.hl.append(self.ax.plot([], [], color='b')[0])

        return self._get_obs()

    def _get_obs(self):
        obs = np.hstack([self.pos / self.size,
                         self.velocity,
                         self.acceleration,
                         (self.target_pos - self.pos) / self.size,
                         np.reshape((self.obstacle_pos - self.pos) / self.size, [-1]),
                         self.obstacle_size/50])
        return obs

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space


if __name__ == "__main__":
    env = reach_goal()
    env.reset()
    counter = 0
    while True:
        action = env.target_pos - env.pos
        _, _, done, _ = env.step(action)
        env.render()
        if done:
            env.reset()
            counter += 1
        if counter == 10:
            env.fig.show()
