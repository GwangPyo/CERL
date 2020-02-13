import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener, distance)


import gym
from gym import spaces
from gym.utils import seeding, EzPickle
from collections import deque

# Routing Optimization Avoiding Obstacle.

FPS = 25
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

# Drone's shape
DRONE_POLY = [
    (-14, +17), (-17, 0), (-17, -10),
    (+17, -10), (+17, 0), (+14, +17)
]

OBSTACLE_INIT_VEL = [(1, 0), (-1, 0), (0, 1), (0, -1),
                (1/np.sqrt(2), 1/np.sqrt(2)), (1/np.sqrt(2), -1/np.sqrt(2)), (-1/np.sqrt(2), 1/np.sqrt(2)),
                     (-1/np.sqrt(2), -1/np.sqrt(2))]

VIEWPORT_W = 600
VIEWPORT_H = 400

W = VIEWPORT_W / SCALE
H = VIEWPORT_H / SCALE

# Shape of Walls
WALL_POLY = [
    (-50, +20), (50, 20),
    (-50, -20), (50, -20)
]


HORIZON_LONG = [(W, -1), (W, 1),
                  (-W, -1), (-W, 1)  ]
VERTICAL_LONG = [ (-1, H), (1, H),
                  (-1, -H), (1, -H)]

HORIZON_SHORT = [(W/3, -0.5), (W/3, 0.5),
                  (-W/3, -0.5), (-W/3, 0.5)  ]
                    # up         # right     # down    # left , left_one_third, right_one_third
WALL_INFOS = {"pos": [(W /2, H), (W, H/2), (W / 2, 0), (0, H/2), (0, H/3), (W, 2/3 * H)],
              "vertices": [HORIZON_LONG, VERTICAL_LONG, HORIZON_LONG, VERTICAL_LONG, HORIZON_SHORT, HORIZON_SHORT]
}

# Initial Position of Drone and Goal which of each chosen randomly among vertical ends.
DRONE_INIT_POS = [((i + 1) * W / 8, H / 8) for i in range(8)]
GOAL_POS = [((i + 1) * W / 8, 7 *H / 8) for i in range(8)]


class LidarCallback(Box2D.b2.rayCastCallback):
    def ReportFixture(self, fixture, point, normal, fraction):
        if (fixture.filterData.categoryBits & 1) == 0:
            return 1
        self.p2 = point
        self.fraction = fraction
        return 0


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.drone == contact.fixtureA.body or self.env.drone == contact.fixtureB.body:
            # if the drone is collide to something, set game over true
            self.env.game_over = True
            # if the drone collide with the goal, success
            if self.env.goal == contact.fixtureA.body or self.env.goal == contact.fixtureB.body:
                self.env.achieve_goal = True
            # if the drone collide with distrubing object, it do not end game
            for obj in self.env.disturbs:
                if obj == contact.fixtureA.body or obj == contact.fixtureB.body:
                    self.env.game_over = False


class NavigationEnvDefault(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    continuous = True

    def __init__(self, max_obs_range=3, num_disturb=4, num_obstacle=4, initial_speed=2, tail_latency=5,
                 latency_accuracy = 0.95, obs_delay=3,
                 *args, **kwargs):
        self.seed()
        self.viewer = None
        self.world = Box2D.b2World(gravity=(0, 0))
        self.moon = None
        self.drone = None
        self.obstacle = None
        self.disturbs = []
        self.walls = []
        self.obstacles = []
        self.goal = None
        self.obs_tracker = None
        self.obs_range_plt = None
        self.max_obs_range = max_obs_range
        self.prev_reward = None
        self.num_beams = 16
        self.lidar = None
        self.achieve_goal = False
        self.strike_by_obstacle = False
        self.tail_latency= tail_latency
        self.num_disturbs = num_disturb
        self.num_obstacles = num_obstacle
        self.dynamics = initial_speed
        self.energy = 1
        self.latency_error = (1 - latency_accuracy)
        self.max_delay = obs_delay
        self.reset()

    @property
    def observation_space(self):
        # lidar + current position + goal position + energy
        return spaces.Box(-np.inf, np.inf, shape=(self.num_beams + 5, ), dtype=np.float32)

    @property
    def action_space(self):
        if self.continuous:
            # Action is two floats [vertical speed, horizontal speed].
            return spaces.Box(-10, +10, shape=(2,), dtype=np.float32)
        else:
            # Nop, fire left engine, main engine, right engine
            return spaces.Discrete(5)

    def seed(self, seed=None):

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.moon: return
        self.world.contactListener = None
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.drone)
        self.drone = None
        self._clean_walls(True)
        self.world.DestroyBody(self.goal)
        self.goal = None
        self.world.DestroyBody(self.obs_range_plt)
        self.obs_range_plt = None
        self._clean_obstacles(True)
        self._clean_distrubs(True)

    def _observe_lidar(self, pos):
        for i in range(self.num_beams):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(i * 2 * np.pi / self.num_beams) * self.max_obs_range,
                pos[1] + math.cos(i * 2 * np.pi / self.num_beams) * self.max_obs_range)
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

    def _build_wall(self):
        wall_pos =WALL_INFOS["pos"]
        wall_ver = WALL_INFOS["vertices"]
        for p, v in zip(wall_pos, wall_ver):
            wall = self.world.CreateStaticBody(
                position=p,
                angle=0.0,
                fixtures=fixtureDef(
                    shape=polygonShape(vertices=v),
                    density=100.0,
                    friction=0.0,
                    categoryBits=0x001,
                    restitution=1.0,)  # 0.99 bouncy
            )
            wall.color1 = (1.0, 1.0, 1.0)
            wall.color2 = (1.0, 1.0, 1.0)
            self.walls.append(wall)

    def _build_obstacles(self):
        for _ in range(self.num_obstacles):
            pos = np.random.uniform(low=0.3, high=0.65, size=2)
            obstacle = self.world.CreateDynamicBody(
                position=(W * pos[0], H * pos[1]),
                angle=0.0,
                fixtures=fixtureDef(
                    shape=circleShape(radius=0.3, pos=(0, 0)),
                    density=5.0,
                    friction=0,
                    categoryBits=0x001,
                    restitution=1.0,
                )  # 0.99 bouncy
            )
            obstacle.color1 = (0.7, 0.2, 0.2)
            obstacle.color2 = (0.7, 0.2, 0.2)
            vel = OBSTACLE_INIT_VEL[np.random.randint(len(OBSTACLE_INIT_VEL))]
            obstacle.linearVelocity.Set(self.dynamics * vel[0], self.dynamics * vel[1])
            self.obstacles.append(obstacle)

    def _build_disturbs(self):
        for _ in range(self.num_obstacles):
            pos = np.random.uniform(low=0.3, high=0.65, size=2)
            disturbs = self.world.CreateDynamicBody(
                position=(W * pos[0], H * pos[1]),
                angle=0.0,
                fixtures=fixtureDef(
                    shape=circleShape(radius=0.3, pos=(0, 0)),
                    density=5.0,
                    friction=0,
                    categoryBits=0x001,
                    restitution=1.0,
                )  # 0.99 bouncy
            )
            disturbs.color1 = (0.2, 0.2, 0.7)
            disturbs.color2 = (0.2, 0.2, 0.7)
            vel = OBSTACLE_INIT_VEL[np.random.randint(len(OBSTACLE_INIT_VEL))]
            disturbs.linearVelocity.Set(self.dynamics * vel[0], self.dynamics* vel[1])
            self.disturbs.append(disturbs)

    def _clean_walls(self, all):
        while self.walls:
            self.world.DestroyBody(self.walls.pop(0))

    def _clean_obstacles(self, all):
        while self.obstacles:
            self.world.DestroyBody(self.obstacles.pop(0))

    def _clean_distrubs(self, all):
        while self.disturbs:
            self.world.DestroyBody(self.disturbs.pop(0))

    def _get_observation(self, position):
        delta_angle = 2* np.pi/self.num_beams
        ranges = [self.world.raytrace(position,
                                      i * delta_angle,
                                      self.max_obs_range) for i in range(self.num_beams)]

        ranges = np.array(ranges)
        return ranges

    def reset(self):
        self._destroy()
        self.lidar = [LidarCallback() for _ in range(self.num_beams)]
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None
        self.moon = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        self.sky_polys = []
        self.achieve_goal = False
        self.strike_by_obstacle = False
        self.energy = 1
        p1 = (1, 1)
        p2 = (W - 1, 1)
        self.moon.CreateEdgeFixture(
            vertices=[p1, p2],
            density=100,
            friction=0,
            restitution=1.0,
        )

        self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])
        self._build_wall()
        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        self._build_obstacles()
        self._build_disturbs()

        drone_pos = DRONE_INIT_POS[np.random.randint(7)]
        self.drone = self.world.CreateDynamicBody(
            position=drone_pos,
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in DRONE_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x003,  # collide all but obs range object
                restitution=0.0)  # 0.99 bouncy
        )

        self.drone.color1 = (0.5, 0.4, 0.9)
        self.drone.color2 = (0.3, 0.3, 0.5)
        goal_pos = GOAL_POS[np.random.randint(7)]
        self.goal = self.world.CreateStaticBody(
            position=goal_pos,
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in DRONE_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x002,
                maskBits=0x0010,  # collide only with ground
                restitution=0.0)  # 0.99 bouncy
        )
        self.goal.color1 = (0., 0.5, 0)
        self.goal.color2 = (0., 0.5, 0)

        self.obs_range_plt = self.world.CreateKinematicBody(
            position=(self.drone.position[0], self.drone.position[1]),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=np.float64(self.max_obs_range), pos=(0, 0)),
                density=0,
                friction=0,
                categoryBits=0x0100,
                maskBits=0x000,  # collide with nothing
                restitution=0.3)
        )
        self.obs_range_plt.color1 = (0.2, 0.2, 0.4)
        self.obs_range_plt.color2 = (0.6, 0.6, 0.6)

        self.drawlist = [self.obs_range_plt, self.drone, self.goal] + self.walls + self.obstacles + self.disturbs
        pos = np.array(self.drone.position)
        goal_pos = np.array(self.goal.position)
        self._observe_lidar(pos)
        state = [self.energy, pos[0], pos[1], goal_pos[0], goal_pos[1]] + [5 * l.fraction for l in self.lidar]
        return np.array(state)

    def step(self, action):
        pos = np.array(self.drone.position)
        goal_pos = np.array(self.goal.position)
        action = np.array(action, dtype=np.float64)
        self.drone.linearVelocity.Set(action[0], action[1])

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.energy -= 1e-3

        self._observe_lidar(pos)
        state = self.local_observation
        reward = 1.0 - np.linalg.norm(goal_pos - pos) * 0.25

        done = self.game_over
        if done:
            if self.achieve_goal:
                reward = 100
            else:
                reward = -100
        if self.energy <= 0:
            done = True
            reward = -100

        return np.array(state, dtype=np.float32), reward, done, {}

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W / SCALE, 0, VIEWPORT_H / SCALE)

        for p in self.sky_polys:
            self.viewer.draw_polygon(p, color=(0, 0, 0))

        self.obs_range_plt.position.Set(self.drone.position[0], self.drone.position[1])

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)
        self.obs_range_plt.position.Set(self.drone.position[0], self.drone.position[1])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    @property
    def local_observation(self):
        pos = self.drone.position
        goal_pos = self.goal.position
        return  [self.energy, pos[0], pos[1], goal_pos[0], goal_pos[1] ] + [5 * l.fraction for l in self.lidar]


class NavigationEnvAggressive(NavigationEnvDefault):
    def step(self, action):
        if not self.continuous:
            map = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
            action = map[action]

        pos = np.array(self.drone.position)
        goal_pos = np.array(self.goal.position)
        vel = np.random.uniform(low=-5, high=5, size=2)
        action = np.array(action, dtype=np.float64)
        self.drone.linearVelocity.Set(action[0], action[1])

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        self._observe_lidar(pos)

        state = self.local_observation
        reward = 1.0 - np.linalg.norm(goal_pos - pos) * 0.25

        done = self.game_over
        if done:
            if self.achieve_goal:
                reward = 1000
            else:
                reward = -100
        self.energy -= 1e-3
        if self.energy <= 0:
            done = True
            reward = -100

        return np.array(state, dtype=np.float32), reward, done, {}


class NavigationEnvDefensive(NavigationEnvDefault):
    def step(self, action):
        if not self.continuous:
            map = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
            action = map[action]

        pos = np.array(self.drone.position)
        goal_pos = np.array(self.goal.position)
        vel = np.random.uniform(low=-5, high=5, size=2)
        action = np.array(action, dtype=np.float64)
        self.drone.linearVelocity.Set(action[0], action[1])

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        self._observe_lidar(pos)
        state = self.local_observation
        reward = 1.0 - np.linalg.norm(goal_pos - pos) * 0.25

        done = self.game_over
        if done:
            if self.achieve_goal:
                reward = 100
            else:
                if self.strike_by_obstacle:
                    reward = -1000
                else:
                    reward = -1000
        self.energy -= 1e-3
        if self.energy <= 0:
            done = True
            reward = -1000

        return np.array(state, dtype=np.float32), reward, done, {}


class NavigationEnvMaster(NavigationEnvDefault):
    continuous = False
    subpolicies = None

    def __init__(self, tail_latency=5, max_obs_range=3, num_disturb=4, num_obstacle=4, initial_speed=2, latency_accuracy=0.95,
                 *args, **kwargs):
        self.latency_error = 1 - latency_accuracy
        self.tail_latency = tail_latency
        self.network_state = self.get_network_state()
        self.subpolicies = self.load_subpolicies()
        super().__init__(max_obs_range=max_obs_range,
                         num_obstacle=num_obstacle,
                         num_disturb=num_disturb,
                         initial_speed=initial_speed,
                         tail_latency=tail_latency,
                         latency_accuracy=latency_accuracy,
                         *args, **kwargs)

        self.cur_local_obs = None

    def get_network_state(self):
        x = np.random.exponential(1)
        if x > self.tail_latency:
            x = self.tail_latency
        elif x < 0.10001:
            x = 0.10001
        return x

    @classmethod
    def load_subpolicies(cls):
        from stable_baselines import PPO2
        return [PPO2.load(path) for path in cls.subpolicies]

    @classmethod
    def set_subpolicies(cls, subpolicies):
        cls.subpolicies = subpolicies

    def time_schedule(self, network_state):
        err = np.random.normal(loc=0, scale=self.latency_error)
        network_state += err
        if network_state > self.tail_latency:
            network_state = self.tail_latency
        elif network_state < 0.10001:
            network_state = 0.10001
        x = int(np.round(10 * network_state))
        return x

    def get_global_obs(self, local_obs):
        global_obs = []
        for obj in self.obstacles:
            global_obs.append(obj.position[0])
            global_obs.append(obj.position[1])
            global_obs.append(obj.linearVelocity[0])
            global_obs.append(obj.linearVelocity[1])
        self.network_state = self.get_network_state()
        global_obs.append(self.network_state)
        obs = np.concatenate((local_obs, np.array(global_obs)))
        return obs

    @property
    def tail_latency_step(self):
        return self.time_schedule(self.tail_latency)

    @property
    def observation_space(self):
        # lidar, obstacle position, goal position, current position, network info
        return gym.spaces.Box(-np.inf, np.inf, shape=(self.num_beams + self.num_obstacles * 4 + 6, ))

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self.subpolicies))

    def reset(self):
        obs = super().reset()
        self.cur_local_obs = obs
        return self.get_global_obs(obs)

    def step(self, action):
        """
        compute disconnected time steps, delay
        must return observation, which contains k-step before data where k is delay

        """
        time_step = self.time_schedule(self.network_state)
        subpolicy = self.subpolicies[action]
        local_obs = self.cur_local_obs
        reward = 0
        done = False
        info = {}
        obs_state = deque(maxlen=self.max_delay)
        if time_step >= self.max_delay:
            # when do you want collect global observation??
            obs_state_step = time_step - self.max_delay
        else:
            obs_state_step = 0
        for i in range(time_step):
            subpolicy_action, _ = subpolicy.predict(local_obs)
            local_obs, reward, done, info = super().step(subpolicy_action)
            obs_state.append(local_obs)
            reward = 0
            if i == obs_state_step:
                obs = self.get_global_obs(local_obs)
            if done:
                if self.achieve_goal:
                    reward = 10
                else:
                    reward = -10

        self.energy -= 1e-3 * time_step
        if self.energy <= 0:
            done = True
            reward = -10

        self.cur_local_obs = local_obs
        self.network_state = self.get_network_state()

        return obs, reward, done, info


class NavigationEnvMasterNoNetworkInfo(NavigationEnvMaster):
    @property
    def observation_space(self):
        # lidar, obstacle position, goal position, current position
        return gym.spaces.Box(-np.inf, np.inf, shape=(self.num_beams + self.num_obstacles * 2 + 4, ))

    def get_global_obs(self, local_obs):
        global_obs = []
        for obj in self.obstacles:
            global_obs.append(obj.position[0])
            global_obs.append(obj.position[1])
        self.network_state = self.get_network_state()

        obs = np.concatenate((local_obs, np.array(global_obs)))
        return obs


class NavigationEnvPlanning(NavigationEnvMaster):
    @property
    def action_space(self):
        max_time = self.tail_latency_step
        return gym.spaces.Box(low=-5, high=5, shape=(2 * max_time, ))

    @staticmethod
    def load_subpolicies():
        return []

    def step(self, action: np.ndarray):
        time_step = self.time_schedule(self.network_state)
        local_obs = self.cur_local_obs
        reward = 0
        done = False
        info = {}
        for i in range(time_step):
            a = action[2 * i: 2 * i + 2]
            local_obs, reward, done, info = NavigationEnvDefault.step(self, a)
            pos = self.drone.position
            goal_pos = self.goal.position
            reward = 1  - np.linalg.norm(pos - goal_pos)* 0.25

            if done:
                if self.achieve_goal:
                    reward = 100
                else:
                    reward = -100

        self.energy -= 1e-3 * time_step
        if self.energy <= 0:
            done = True
            reward = -100

        self.cur_local_obs = local_obs
        self.network_state = self.get_network_state()
        obs = self.get_global_obs(local_obs)
        return obs, reward, done, info