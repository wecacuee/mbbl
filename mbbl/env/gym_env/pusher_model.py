#!/usr/bin/env python3

# Standard library
from abc import ABC, abstractmethod
import time
from functools import partial
from tempfile import NamedTemporaryFile
import os
import os.path as osp

# 3rd party
import numpy as np
import em
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.mujoco_env import MujocoEnv
from gym import utils
from pkg_resources import resource_filename
from keyword2cmdline import command, command_config
from PIL import Image
from functools import wraps

# local imports
from .mujoco_utils import (mjc_qpos_indices_from_jnt_names,
                           mjc_dof_indices_from_jnt_names)
from .cacheutils import cached_property

# Global variables
PKG = 'mbbl'
resource_filename = partial(resource_filename, PKG)


def genmodel(outfile, template, **kw):
    outfile.write(em.expand(open(template).read(), kw).encode('utf-8'))


class MBBLGymEnv(ABC):
    @abstractmethod
    def step(self, action):
        pass


class Encoder(ABC):
    """ @brief: Encode decode state representation
    """
    @abstractmethod
    def __init__(self, env):
        """ @brief: Initialize encoder from Env of type
        """

    @abstractmethod
    def encode(self, *args):
        """ @brief: Encode state vector from semantic understandable arguments
        """
        return

    @abstractmethod
    def decode(self, state):
        """ @brief: Decode state vector to semantic understandable arguments
        """
        return ()


class StateEncoder(Encoder):
    def __init__(self, env):
        self.env = env
        self.part_lengths = [len(env.qpos_indices),
                             len(env.qpos_indices),
                             len(env.dof_indices),
                             3,
                             3]

    def _fromparts(self, parts):
        assert self.part_lengths == list(map(len, parts))
        return np.concatenate(parts)

    def _toparts(self, state):
        breakpoints = np.cumsum([0] + self.part_lengths)
        parts = [state[s:e]
                 for s, e in zip(breakpoints[:-1], breakpoints[1:])]
        return parts

    def encode(self, theta, qvel, finger_vec, goal_vec):
        return self._fromparts([
            np.cos(theta),
            np.sin(theta),
            qvel,
            finger_vec,
            goal_vec
        ])

    def decode(self, state):
        cos_thetas, sin_thetas, qvel, finger_vec, goal_vec = self._toparts(state)
        thetas = np.arctan2(cos_thetas, sin_thetas)
        return thetas, qvel, finger_vec, goal_vec


# Copied from gym.envs.mujoco.reacher.py
@command_config
class PusherEnv(utils.EzPickle, MujocoEnv):
    def __init__(self,
                 xml_loc='env/gym_env/assets/mjxml',
                 emxml='flat_pusher.xml.em',
                 frame_skip=2,
                 stl_loc='env/gym_env/assets/mjxml/stls',
                 forearm_stl='Flat-T.stl',
                 joints=["joint%d" % i for i in range(4)],
                 no_environment=False,
                 obs_encoder_gen=StateEncoder):
        self.obs_encoder_gen = obs_encoder_gen
        self.joints = joints
        utils.EzPickle.__init__(self)
        emxmlfile = resource_filename('/'.join((xml_loc, emxml)))
        emxml_dir = osp.dirname(emxmlfile)
        with NamedTemporaryFile(mode='w+b', suffix='.xml',
                                dir=emxml_dir) as outfile:
            genmodel(outfile,
                     emxmlfile,
                     forearm_stl_path=resource_filename('/'.join(
                         (stl_loc, forearm_stl))),
                     no_environment=no_environment)
            outfile.flush()
            MujocoEnv.__init__(self, outfile.name, frame_skip)
            outfile.close()

    @cached_property
    def obs_encoder(self):
        return self.obs_encoder_gen(self)

    def jnt_type_for_qpos(self, qposadr):
        m = self.model
        jntidx = list(m.jnt_qposadr.flat).index(qposadr)
        return m.jnt_type.reshape(-1)[jnt_type]

    def jnt_type_for_qvel(self, qveladr):
        m = self.model
        jntidx = list(m.jnt_qveladr.flat).index(qveladr)
        return m.jnt_type.reshape(-1)[jnt_type]

    @cached_property
    def qpos_indices(self):
        return mjc_qpos_indices_from_jnt_names(self.model, self.joints)

    @cached_property
    def dof_indices(self):
        return mjc_dof_indices_from_jnt_names(self.model, self.joints)

    def _compute_reward(self, a, reward_wts=[1, 0.1, 0.5]):
        vec_1 = self.get_body_com("object") - self.get_body_com("fingertip")
        vec_2 = self.get_body_com("object") - self.get_body_com("target")

        reward_near = -np.linalg.norm(vec_1)
        reward_dist = -np.linalg.norm(vec_2)
        reward_ctrl = -np.square(a).sum()
        reward_components = [reward_dist, reward_ctrl, reward_near]
        reward = np.array(reward_wts).dot(reward_components)
        return reward, reward_components

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        reward, reward_components = self._compute_reward(a)
        return (ob, reward, done, dict(reward_components=reward_components))

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.data.qpos.reshape(-1).copy()
        qpos[self.qpos_indices] = self.np_random.uniform(
            low=-0.1, high=0.1,
            size=len(self.qpos_indices)) + self.init_qpos[self.qpos_indices]
        while True:
            self.target = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.target) < 0.3:
                break

        while True:
            self.object_pos = self.np_random.uniform(low=-0.2, high=.2, size=2)
            if (np.linalg.norm(self.object_pos - self.target) > 0.1
                    and np.linalg.norm(self.object_pos) < 0.3):
                break

        qvel = self.data.qvel.reshape(-1).copy()
        qvel[self.dof_indices] = (
            self.init_qvel[self.dof_indices]
            + self.np_random.uniform(low=-.005, high=.005,
                                     size=len(self.dof_indices)))
        super().set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.data.qpos.reshape(-1)[self.qpos_indices]
        qvel = self.data.qvel.reshape(-1)[self.dof_indices]
        finger_vec = self.get_body_com("fingertip") - self.get_body_com("object")
        goal_vec = self.get_body_com("object") - self.get_body_com("target")
        return self.obs_encoder.encode(theta, qvel, finger_vec, goal_vec)

    def obs2state(self, obs):
        theta, qvel_decoded, finger_vec, goal_vec = self.obs_encoder.decode(obs)
        qpos = np.zeros_like(self.data.qpos).ravel()
        qvel = np.zeros_like(self.data.qvel).ravel()
        qpos[self.qpos_indices] = theta
        qvel[self.dof_indices] = qvel_decoded
        return qpos, qvel

    def set_state(self, state):
        super().set_state(*self.obs2state(state))

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 0.2
        self.viewer.cam.lookat[2] = 0.8925
        self.viewer.cam.elevation = -90

    @cached_property
    def dynamics(self):
        from .dynamics import MujocoDynamics
        return MujocoDynamics(self.sim, self.joints)

    def togoal(self, x):
        qpos = self.data.qpos.reshape(-1)
        qpos[self.qpos_indices] = x[:len(self.qpos_indices)]
        qvel = self.data.qpos.reshape(-1)
        qvel[self.dof_indices] = x[len(self.qpos_indices):]
        super().set_state(qpos, qvel)
        return self.get_body_com("fingertip")


class RenderWrapper:
    @command_config
    def __init__(self,
                 envfactory,
                 fname_fmt,
                 every=10,
                 mode='rgb_array',
                 shape=(640, 800)):
        self.env         = envfactory()
        self.fname_fmt   = fname_fmt
        self.every       = every
        self.mode        = mode
        self.shape       = shape

    def render(self, i):
        env         = self.env
        every       = self.every
        fname_fmt   = self.fname_fmt
        mode        = self.mode
        shape       = self.shape
        if i % every == 0:
            img = env.render(mode=mode,
                            width=shape[1],
                            height=shape[0])
            im = Image.fromarray(img)
            fname = fname_fmt(i=i)
            dir_fname = os.path.dirname(fname)
            if not os.path.exists(dir_fname):
                os.makedirs(dir_fname)

            print('Saving image %s' % fname)
            im.save(fname)

    def __getattr__(self, attr):
        return getattr(self.env, attr)


class RandomAgent:
    def __init__(self, env, T, cost):
        self.env = env

    def control(self, obs, rew, i):
        return self.env.action_space.sample()


class ConstAgent:
    def __init__(self, env, T, cost, const=0):
        self.env = env
        self.const = const

    def control(self, obs, rew, i):
        return np.ones_like(self.env.action_space.sample()) * self.const


def make_target_reach_cost(env):
    target = env.get_body_com("target")
    from ..cost import ReachCost
    return ReachCost(target, env)


@command_config
def make_rendering_wrapper_pusher_env(
        parent=PusherEnv,
        forearm_stl='flat-T.stl',
        fname_fmt='data/model-imgs/{forearm_stl}-{i}.png'):
    return RenderWrapper(partial(parent,
                                 forearm_stl=forearm_stl),
                         partial(fname_fmt.format,
                                 forearm_stl=forearm_stl))


def play(envfactory=make_rendering_wrapper_pusher_env,
         costfactory=make_target_reach_cost,
         agentfactory=RandomAgent,
         seed=None,
         T=100):
    if seed is not None:
        print("seed=%d" % seed)
        np.random.seed(seed=seed)
    env = envfactory()
    cost = costfactory(env)
    agent = agentfactory(env, T, cost)
    obs = env.reset()
    r = 0
    for i in range(T):
        u = agent.control(obs, r, i)
        print(env.state_vector(), u)
        obs, r, done, info = env.step(u)
        env.render(i)
        time.sleep(0.300)


if __name__ == '__main__':
    command(play)()
