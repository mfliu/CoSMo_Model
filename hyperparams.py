from __future__ import division

from datetime import datetime
import os.path
import numpy as np

from gps.algorithm.algorithm_traj_opt_pilqr import AlgorithmTrajOptPILQR
from gps import __file__ as gps_filepath
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import RAMP_FINAL_ONLY
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.policy_opt.tf_model_example import tf_network
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION
from gps.gui.config import generate_experiment_info
from gps.algorithm.cost.cost_utils import RAMP_CONSTANT

ALGORITHM_NN_LIBRARY = "tf"

SENSOR_DIMS = {
        JOINT_ANGLES: 7,
        JOINT_VELOCITIES: 7,
        END_EFFECTOR_POINTS: 3,
        END_EFFECTOR_POINT_VELOCITIES: 3,
        ACTION:7,
        }

GAINS = np.array([500, 500, 500, 1, 0.01, 0.0025, 0.0025]) 

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/CoSMo/'


common = {
    'experiment_name': 'cosmo_reach' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentMuJoCo,
    'filename': './mjc_models/cosmo_model.xml',
    'x0': np.concatenate((np.array([0., 0., 0., 0, 0., 0., 0]), np.zeros(7))),
    'dt': 0.05,
    'substeps': 5,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([4]),
    'pos_body_offset': [[np.array([0, 0, 0])]],
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                      END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                    END_EFFECTOR_POINT_VELOCITIES],
    'camera_pos': np.array([2, -2, 3.5, 0, 0, 0]),
}

algorithm = {
    'type': AlgorithmBADMM,
    'conditions': common['conditions'],
    'iterations': 12,
    'lg_step_schedule': np.array([1e-2]), 
    'policy_dual_rate': 0.1,
    'init_pol_wt': 0.002,
    'ent_reg_schedule': np.array([1e-2]),
    'fixed_lg_step': 3,
    'kl_step': 1.0,
    'min_step_mult': 0.1,
    'max_step_mult': 10.0,
    'sample_decrease_var': 0.05,
    'sample_increase_var': 0.5,
    'policy_sample_mode': 'replace'
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains':  1.0 / GAINS,
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 1.0,
    'stiffness': 1.0,
    'stiffness_vel': 0.5,
    'final_weight': 50.0,
    'dt': agent['dt'],
    'T': agent['T'],
}

torque_cost = {
    'type': CostAction,
    'wu': 1/GAINS * 1e-6,
}

fk_cost = [{
    'type': CostFK,
    'target_end_effector': np.array([0.2, -0.2, 0.25]),
    'wp': np.array([1, 1, 1]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-2,
    'ramp_option': RAMP_CONSTANT,
} for i in xrange(common['conditions'])]

algorithm['cost'] = [{
    'type': CostSum,
    'costs': [fk_cost[i]],
    'weights': [6.0],
} for i in xrange(common['conditions'])]

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

if ALGORITHM_NN_LIBRARY == "tf":
    algorithm['policy_opt'] = {
        'type': PolicyOptTf,
        'network_params': {
            'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
            'sensor_dims': SENSOR_DIMS,
        },
        'weights_file_prefix': EXP_DIR + 'policy',
        'iterations': 3000,
        'network_model': tf_network
    }
elif ALGORITHM_NN_LIBRARY == "caffe":
    algorithm['policy_opt'] = {
        'type': PolicyOptCaffe,
        'weights_file_prefix': EXP_DIR + 'policy',
        'iterations': 5000,
    }

algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 20,
}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 5,
    'verbose_trials': 1,
    'verbose_policy_trials': 1,
    'common': common,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
}

common['info'] = generate_experiment_info(config)

