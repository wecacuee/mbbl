from functools import partial

import numpy as np
import mujoco_py as mj

def mjc_addr_to_indices(addr):
    indices = (np.arange(*addr)
               if isinstance(addr, tuple)
               else np.arange(addr, addr+1))
    return indices

def mjc_get_joint_qpos_addr(model, jn, isdof=False):
    jntidx = model.joint_names.index(jn.encode('utf-8'))
    return (model.jnt_dofadr[jntidx]
            if isdof
            else model.jnt_qposadr[jntidx])

mjc_get_joint_qvel_addr = partial(mjc_get_joint_qpos_addr, isdof=True)


def mjc_qpos_indices_from_jnt_names(model, joints):
    return np.hstack([
        mjc_addr_to_indices(mjc_get_joint_qpos_addr(model, j))
        for j in joints])


def mjc_dof_indices_from_jnt_names(model, joints):
    return np.hstack([
        mjc_addr_to_indices(mjc_get_joint_qvel_addr(model, j))
        for j in joints])
