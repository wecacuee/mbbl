# -----------------------------------------------------------------------------
#   @author:
#       Tingwu Wang
#   @brief:
#       Several basic gym environments that
#       1. Provide ground-truth reward function.
#       2. Reward is a function of the observation.
#       3. an episode has fixed length.
#       4. provide ground-truth dynamics
# -----------------------------------------------------------------------------
from itertools import combinations

import autograd.numpy as np
from autograd import elementwise_grad as egrad, grad, jacobian

from mbbl.config import init_path
from mbbl.env import base_env_wrapper
from mbbl.env import env_register
from mbbl.env import env_util

from mbbl.env.gym_env.pusher_model import PusherEnv
import tensorflow as tf


class env(base_env_wrapper.base_env):

    # reacher have observations from qpos (applied sin / cos)
    ARM_2D = ['gym_pusher']

    def __init__(self, env_name, rand_seed, misc_info, envgen=PusherEnv):
        # _build_env dependencies should be initialized before calling super().__init__
        self.envgen = envgen
        super(env, self).__init__(env_name, rand_seed, misc_info)
        self._base_path = init_path.get_abs_base_dir()

        self._len_qpos, self._len_qvel = \
            env_util.get_gym_q_info(self._env, self._current_version)

        # return the reset as the gym?
        if 'reset_type' in misc_info and misc_info['reset_type'] == 'gym':
            # TODO: Separate reset_type for observation_space, action_space
            # sources.
            # TODO: rename reset_type -> reset_return_type
            self._reset_return_obs_only = True
            self.observation_space, self.action_space = \
                self._env.observation_space, self._env.action_space
            # it's possible some environments have different obs
            self.observation_space = \
                env_util.box(self._env_info['ob_size'], -1, 1)
        else:
            self._reset_return_obs_only = False

    def step(self, action):
        # get the observation
        ob, _, _, info = self._env.step(action)

        # get the reward
        reward = self.reward(
            {'end_state': ob, 'start_state': self._old_ob, 'action': action}
        )

        # get the end signal
        self._current_step += 1
        if self._current_step >= self._env_info['max_length']:
            done = True
        else:
            done = False
        # TODO: Why is ob not an numpy array while _old_ob is?
        self._old_ob = np.array(ob)
        return ob, reward, done, info

    def reset(self, control_info={}):
        self._current_step = 0
        self._old_ob = self._env.reset()

        # TODO: Isn't _old_ob already a nupy array?
        if self._reset_return_obs_only:
            return np.array(self._old_ob)
        else:
            return np.array(self._old_ob), 0.0, False, {}

    def _build_env(self):
        import gym
        self._current_version = gym.__version__
        if self._current_version not in ['0.7.4', '0.9.4']:
            raise ValueError("Invalid gym-{}".format(self._current_version))

        # make the environments
        self._env = self.envgen()
        self._env_info = env_register.get_env_info(self._env_name)

    def _set_groundtruth_api(self):
        """ @brief:
                In this function, we could provide the ground-truth dynamics
                and rewards APIs for the agent to call.
                For the new environments, if we don't set their ground-truth
                apis, then we cannot test the algorithm using ground-truth
                dynamics or reward
        """
        self._set_reward_api()
        self._set_dynamics_api()

    def _set_dynamics_api(self):
        '''
        def fdynamics(self, data_dict):
            raise NotImplementedError

        self.fdynamics = fdynamics
        '''
        def set_state(data_dict):
            """ @brief: Set the simulator state from data dict

            @input:
                data_dict: A dictionary with keys
                    start_state: State vector in cos-sin for controller
            """
            start_state = data_dict['start_state']
            self._env.set_state(start_state)
            # reset the state
            sim = (self._env
                   if self._current_version in ['0.7.4', '0.9.4']
                   else self._env.env.sim)
            sim.model._compute_subtree()  # pylint: disable=W0212
            sim.model.forward()
        self.set_state = set_state

        def fdynamics(data_dict):
            """ @brief: Get dynamics output at given state

            @input:
                data_dict: A dictionary with keys
                    start_state: State vector in cos-sin for controller
                    action     : Action
            """
            self.set_state(data_dict)
            return self.step(data_dict['action'])[0]
        self.fdynamics = fdynamics

    def _set_reward_api(self):
        # step 1, set the zero-order reward function
        assert self._env_name in self.ARM_2D

        state = tf.placeholder(tf.float32, shape=(self._env_info['ob_size'],))
        action = tf.placeholder(tf.float32, shape=(self._env_info['action_size'],))
        def _tf_reward_g(state, action):
            dist_vec = state[-6:]
            reward_dist = - tf.linalg.norm(dist_vec)
            reward_ctrl = - tf.reduce_sum(tf.square(action))
            return reward_dist + reward_ctrl
        tf_reward_g = _tf_reward_g(state, action)

        def tf_reward(data_dict):
            return session.run(tf_reward_g,
                               feed_dict={
                                   state: data_dict['start_state'],
                                   action: data_dict['action']})

        def reward_np(dist_vec, action):
            """ @brief: Return the reward for given state

            @input:
                data_dict: A dictionary with keys
                    start_state: State vector in cos-sin for controller
            """
            reward_dist = - np.linalg.norm(dist_vec, axis=-1)
            reward_ctrl = - np.square(action).sum(axis=-1)

            return reward_dist + reward_ctrl

        def reward(data_dict):
            return reward_np(data_dict['start_state'][-6:], data_dict['action'])

        self.reward = reward
        # self.reward = tf_reward

        def tf_reward_derivative(data_dict, target):
            num_data = len(data_dict['start_state'])
            rew_x_graph = tf.gradients([tf_reward_g], [state])
            rew_u_graph = tf.gradients([tf_reward_g], [action])
            feed_dict = {state: data_dict['start_state'],
                         action: data_dict['action']}
            if target == 'state':
                derivative_data = session.run(rew_grad_graph,
                                              feed_dict=feed_dict)
            elif target == 'action':
                derivative_data = session.run(rew_u_graph,
                                              feed_dict=feed_dict)
            elif target == 'state-state':
                rew_xx_graph = tf.gradients([rew_x_graph],
                                            [state])
                derivative_data = session.run(rew_xx_graph,
                                              feed_dict=feed_dict)
            elif target == 'action-action':
                rew_uu_graph = tf.gradients([rew_u_graph],
                                            [action])
                derivative_data = session.run(rew_uu_graph,
                                              feed_dict=feed_dict)
            elif target == 'action-state':
                rew_ux_graph = tf.gradients([rew_u_graph],
                                            [state])
                derivative_data = session.run(rew_ux_graph,
                                              feed_dict=feed_dict)
            elif target == 'state-action':
                rew_xu_graph = tf.gradients([rew_x_graph],
                                            [action])
                derivative_data = session.run(rew_xu_graph,
                                              feed_dict=feed_dict)
            else:
                raise NotImplementedError("Invalid target %s" % target)
            return derivative_data


        def reward_derivative(data_dict, target):
            """ @brief: Return the reward derivative

            @input:
                data_dict: A dictionary with keys
                    start_state: State vector in cos-sin for controller

                target   : One of the following strings:
                      state, action, state-state, state-action, action-action,
                      action-state
            """
            num_data = len(data_dict['start_state'])
            state = data_dict['start_state']
            dist_vec = state[:, -6:]
            ssize = dist_vec.shape[1]
            action = data_dict['action']
            if target == 'state':
                # reward - \sqrt(x^2 + y^2 + z^2)
                derivative_data = np.zeros(
                    [num_data, self._env_info['ob_size']], dtype=np.float
                )
                norm = np.linalg.norm(data_dict['start_state'][:, -6:],
                                      axis=1, keepdims=True)
                derivative_data[:, -6:] = \
                    - data_dict['start_state'][:, -6:] / norm

                derivative_data_ag = egrad(reward_np, argnum=0)(dist_vec, action)
                assert np.allclose(derivative_data[:, -6:], derivative_data_ag)
            elif target == 'action':
                derivative_data = np.zeros(
                    [num_data, self._env_info['action_size']], dtype=np.float
                )
                derivative_data[:, :] = - 2.0 * 1.0 * data_dict['action'][:, :]
                derivative_data_ag = egrad(reward_np, argnum=1)(dist_vec, action)
                assert np.allclose(derivative_data, derivative_data_ag)
            elif target == 'state-state':
                derivative_data = np.zeros(
                    [num_data,
                     self._env_info['ob_size'], self._env_info['ob_size']],
                    dtype=np.float
                )
                norm = np.linalg.norm(data_dict['start_state'][:, -6:],
                                      axis=1, keepdims=True)
                norm3 = np.reshape(np.power(norm, 3), [-1])

                # the diagonal term
                for i in range(-6, 0):
                    derivative_data[:, i, i] = np.reshape(
                        - 1.0 / norm.reshape([-1]) +
                        np.square(data_dict['start_state'][:, i]) / norm3,
                        [-1]
                    )
                # the off diagonal term
                for x, y in combinations(range(-6, 0), 2):
                    derivative_data[:, x, y] = \
                        data_dict['start_state'][:, x] * \
                        data_dict['start_state'][:, y] / \
                        norm3
                    derivative_data[:, y, x] = derivative_data[:, x, y]

            elif target == 'action-action':
                derivative_data = np.zeros(
                    [num_data, self._env_info['action_size'],
                     self._env_info['action_size']],
                    dtype=np.float
                )

                for diagonal_id in range(self._env_info['action_size']):
                    derivative_data[:, diagonal_id, diagonal_id] += -2.0
            elif target == 'state-action':
                derivative_data = np.zeros(
                    [num_data, self._env_info['ob_size'],
                     self._env_info['action_size']],
                    dtype=np.float
                )
            elif target == 'action-state':
                derivative_data = np.zeros(
                    [num_data, self._env_info['action_size'],
                     self._env_info['ob_size']],
                    dtype=np.float
                )

            else:
                raise NotImplementedError

            return derivative_data

        self.reward_derivative = reward_derivative
        # self.reward_derivative = tf_reward_derivative


if __name__ == '__main__':

    test_env_name = ['gym_pusher']
    for env_name in test_env_name:
        test_env = env(env_name, 1234, {})
        api_env = env(env_name, 1234, {})
        api_env.reset()
        ob, reward, _, _ = test_env.reset()
        for _ in range(100):
            action = np.random.uniform(-1, 1, test_env._env.action_space.shape)
            new_ob, reward, _, _ = test_env.step(action)

            # test the reward api
            reward_from_api = \
                api_env.reward({'start_state': ob, 'action': action})
            reward_error = np.sum(np.abs(reward_from_api - reward))

            # test the dynamics api
            newob_from_api = \
                api_env.fdynamics({'start_state': ob, 'action': action})
            ob_error = np.sum(np.abs(newob_from_api - new_ob))

            ob = new_ob

            print('reward error: {}, dynamics error: {}'.format(
                reward_error, ob_error)
            )
