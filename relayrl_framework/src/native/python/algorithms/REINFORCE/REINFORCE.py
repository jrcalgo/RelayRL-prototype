import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from _common._algorithms.BaseAlgorithm import AlgorithmAbstract

import numpy as np
import torch
from torch.optim import Adam

from .kernel import PolicyWithoutBaseline, PolicyWithBaseline
from .replay_buffer import ReplayBuffer

from utils.logger import EpochLogger, setup_logger_kwargs
from relayrl_framework import RelayRLTrajectory, ConfigLoader


class REINFORCE(AlgorithmAbstract):
    def __init__(self, env_dir: str, config_path: str, obs_dim: int, act_dim: int, buf_size: int):
        super().__init__()

        """Import and load RelayRL/config.json REINFORCE algorithm configurations and applies them to
        the current instance.
        
        Loads defaults if config.json is unavailable or key error thrown.
        """
        config_loader = ConfigLoader(algorithm_name='REINFORCE', config_path=config_path)
        hyperparams = config_loader.get_algorithm_params()['REINFORCE']
        self.save_model_path = config_loader.get_server_model_path()

        # Hyperparameters
        self._discrete = hyperparams['discrete']
        self._with_vf_baseline = hyperparams['with_vf_baseline']
        self._seed = hyperparams['seed']
        self._traj_per_epoch = hyperparams['traj_per_epoch']
        self._gamma = hyperparams['gamma']
        self._lam = hyperparams['lam']
        self._pi_lr = hyperparams['pi_lr']
        self._vf_lr = hyperparams['vf_lr']
        self._train_vf_iters = hyperparams['train_vf_iters']

        seed = self._seed + 10000 * os.getpid()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self._replay_buffer = ReplayBuffer(obs_dim, act_dim, buf_size, self._gamma, self._lam, self._with_vf_baseline)
        if self._with_vf_baseline:
            self._model = PolicyWithBaseline(obs_dim, act_dim, self._discrete, [128, 128], torch.nn.ReLU)
            self._vf_optimizer = Adam(self._model.baseline.parameters(), lr=self._vf_lr)
        else:
            self._model = PolicyWithoutBaseline(obs_dim, act_dim, self._discrete, [128, 128])
        self._pi_optimizer = Adam(self._model.parameters(), lr=self._pi_lr)

        # set up logger
        log_data_dir = os.path.join(env_dir, './logs/')
        logger_kwargs = setup_logger_kwargs(
            "relayrl-reinforce-vf-info" if self._with_vf_baseline else "relayrl-reinforce-info", seed=seed, data_dir=log_data_dir)
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        self.traj = 0
        self.epoch = 0
        
        print("[REINFORCE Algorithm] Initialized")

    def save(self) -> None:
        self._model.eval()
        model_script = torch.jit.script(self._model)
        torch.jit.save(model_script, self.save_model_path)
        self._model.train()

    def receive_trajectory(self, trajectory: RelayRLTrajectory) -> bool:
        self.traj += 1
        ep_ret, ep_len = 0, 0

        for r4a in trajectory.get_actions():
            ep_ret += r4a.get_rew()
            ep_len += 1
            
            data = r4a.get_data()
            if not r4a.get_done():
                if self._with_vf_baseline:
                    self._replay_buffer.store(r4a.get_obs(), r4a.get_act(), r4a.get_mask(), r4a.get_rew(), data['v'], data['logp_a'])
                    self.logger.store(VVals=data['v'])
                else:
                    self._replay_buffer.store(r4a.get_obs(), r4a.get_act(), r4a.get_mask(), r4a.get_rew(), None, data['logp_a'])
            else:
                self._replay_buffer.finish_path(r4a.get_rew())
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)

        if self.traj > 0 and self.traj % self._traj_per_epoch == 0:
            self.epoch += 1
            self.train_model()
            self.log_epoch()
            return True

        return False

    def train_model(self) -> None:
        data = self._replay_buffer.get()

        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        if self._with_vf_baseline:
            v_l_old = self.compute_loss_vf(data).item()

        self._pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self._pi_optimizer.step()

        if self._with_vf_baseline:
            for i in range(self._train_vf_iters):
                self._vf_optimizer.zero_grad()
                loss_v = self.compute_loss_vf(data)
                loss_v.backward()
                self._vf_optimizer.step()

        kl, ent = pi_info['kl'], pi_info['ent']
        delta_loss_pi = loss_pi.item() - pi_l_old
        delta_loss_v = 0 if not self._with_vf_baseline else loss_v.item() - v_l_old

        self.logger.store(LossPi=loss_pi.item(), DeltaLossPi=delta_loss_pi,
                          KL=kl, Entropy=ent)

        if self._with_vf_baseline:
            self.logger.store(LossV=loss_v.item(), DeltaLossV=delta_loss_v)

    def log_epoch(self) -> None:
        self.logger.log_tabular('Epoch', self.epoch)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('DeltaLossPi', average_only=True)
        if self._with_vf_baseline:
            self.logger.log_tabular('VVals', with_min_and_max=True)
            self.logger.log_tabular('LossV', average_only=True)
            self.logger.log_tabular('DeltaLossV', average_only=True)
        self.logger.log_tabular('KL', average_only=True)
        self.logger.log_tabular('Entropy', average_only=True)
        self.logger.dump_tabular()

    def compute_loss_pi(self, data):
        obs, mask, act, adv, logp_old = data['obs'], data['mask'], data['act'], data['adv'], data['logp']
        pi_probs, pi_logits, logp = self._model.policy.forward(obs, mask, act)
        loss_pi = -(logp * adv).mean()

        approx_kl = (logp_old - logp).mean().item()
        
        # Entropy calculation
        min_real = torch.finfo(pi_logits.dtype).min
        logits = torch.clamp(pi_logits, min=min_real)
        p_log_p = logits * pi_probs
        entropy = -p_log_p.sum(-1).mean().item()

        pi_info = dict(kl=approx_kl, ent=entropy)

        return loss_pi, pi_info

    def compute_loss_vf(self, data):
        obs, mask, ret = data['obs'], data['mask'], data['ret']
        return ((self._model.baseline.forward(obs, mask) - ret) ** 2).mean()
