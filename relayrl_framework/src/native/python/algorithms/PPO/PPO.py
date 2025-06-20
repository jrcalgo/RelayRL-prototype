import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from _common._algorithms.BaseAlgorithm import AlgorithmAbstract

import numpy as np
import torch
from torch.optim import Adam

from .kernel import RLActorCritic
from .replay_buffer import ReplayBuffer

from utils.logger import EpochLogger, setup_logger_kwargs
from relayrl_framework import RelayRLTrajectory, ConfigLoader

"""Import and load RelayRL/config.json PPO algorithm configurations and applies them to
the current instance.

Loads defaults if config.json is unavailable or key error thrown.
"""
CONFIG_PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "config.json")
config_loader = ConfigLoader(algorithm_name='PPO', config_path=CONFIG_PATH)
hyperparams = config_loader.get_algorithm_params()['PPO']
save_model_path = config_loader.get_save_model_path()


class PPO(AlgorithmAbstract):
    """Algorithm class for PPO.

    See OpenAI Spinning Up PPO implementation:
    https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ppo

    Attributes:
        traj (int): number of trajectories that have occured.
        epoch (int): number of epochs that have occured.

    Hyperparameters without defaults:

        kernel_size (int): number of observations. e.g. MAX_QUEUE_SIZE
        kernel_dim (int): number of features. e.g. JOB_FEATURES
        buf_size (int): size of replay buffer.
        
    Hyperparameters with defaults:
    
        seed (int): seed for torch and numpy random number generators.

        traj_per_epoch (int): number of trajectories to receive before training the model.

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        gamma (float): Discount factor. (Always between 0 and 1.)
        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        pi_lr (float): Learning rate for policy optimizer.
        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)
        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.  
        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

    """
    def __init__(self, env_dir: str, kernel_size: int, kernel_dim: int, act_dim:int, buf_size: int, seed: int = hyperparams['seed'],
                 traj_per_epoch: int = hyperparams['traj_per_epoch'], clip_ratio: float = hyperparams['clip_ratio'],
                 gamma: float = hyperparams['gamma'], lam: float = hyperparams['lam'],
                 pi_lr: float = hyperparams['pi_lr'], vf_lr: float = hyperparams['vf_lr'],
                 train_pi_iters: int = hyperparams['train_pi_iters'], train_v_iters: int = hyperparams['train_v_iters'],
                 target_kl: float = hyperparams['target_kl']):

        super().__init__()
        seed += 10000 * os.getpid()
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Hyperparameters
        self._traj_per_epoch = traj_per_epoch
        self._clip_ratio = clip_ratio
        self._train_pi_iters = train_pi_iters
        self._train_v_iters = train_v_iters
        self._target_kl = target_kl
        
        self._replay_buffer = ReplayBuffer(kernel_size * kernel_dim, act_dim, buf_size, gamma=gamma, lam=lam)
        self._model = RLActorCritic(kernel_size, kernel_dim, act_dim)
        self._pi_optimizer = Adam(self._model.pi.parameters(), lr=pi_lr)
        self._vf_optimizer = Adam(self._model.v.parameters(), lr=vf_lr)

        # set up logger
        log_data_dir = os.path.join(env_dir, './logs/')
        logger_kwargs = setup_logger_kwargs(
            "relayrl-ppo-info", seed=seed, data_dir=log_data_dir)
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())
        self.logger.setup_pytorch_saver(self._model)

        self.traj = 0
        self.epoch = 0
        
        print("[PPO Algorithm] Initialized")

    def save(self) -> None:
        """Save model as file.

        Uses .pt file extension.

        Args:
            filename: name to save file as

        """
        self._model.eval()
        model_script = torch.jit.script(self._model)
        torch.jit.save(model_script, save_model_path)
        self._model.train()

    def receive_trajectory(self, trajectory: RelayRLTrajectory) -> bool:
        """Process a trajectory received by training_server.

        If an epoch is triggered, calls train_model().

        Args:
            trajectory: holds agent experiences since last trajectory
        Returns:
            True if an epoch was triggered and an updated model should be sent.

        """
        self.traj += 1
        ep_ret, ep_len = 0, 0
        
        for r4a in trajectory.get_actions():
            # Process each RelayRLAction in the trajectory
            ep_ret += r4a.get_rew()
            ep_len += 1
            
            data = r4a.get_data()
            if not r4a.get_done():
                self._replay_buffer.store(r4a.get_obs(), r4a.get_act(), r4a.get_mask(), r4a.get_rew(), data['v'], data['logp_a'])
                self.logger.store(VVals=data['v'])
            else:
                self._replay_buffer.finish_path(r4a.get_rew())
                self.logger.store(EpRet=ep_ret, EpLen=ep_len)

        # get enough trajectories for training the model
        if self.traj > 0 and self.traj % self._traj_per_epoch == 0:
            self.epoch += 1
            self.train_model()
            self.log_epoch()
            return True
        
        return False

    def train_model(self) -> None:
        """Train model on data from replay_buffer.
        """
        # data holds all timesteps since last epoch
        data = self._replay_buffer.get()

        # calculate loss
        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = self.compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self._train_pi_iters):
            self._pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = (pi_info['kl'])
            if kl > 1.5 * self._target_kl:
                self.logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break
            loss_pi.backward()
            self._pi_optimizer.step()
        
        self.logger.store(StopIter=i)

        # Value function learning
        for i in range(self._train_v_iters):
            self._vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            self._vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        self.logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    def log_epoch(self) -> None:
        """Log the information collected in logger over the course of the last epoch.
        """
        # Log info about epoch
        self.logger.log_tabular('Epoch', self.epoch)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', with_min_and_max=True)
        self.logger.log_tabular('VVals', with_min_and_max=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossV', average_only=True)
        self.logger.log_tabular('DeltaLossPi', average_only=True)
        self.logger.log_tabular('DeltaLossV', average_only=True)
        self.logger.log_tabular('Entropy', average_only=True)
        self.logger.log_tabular('KL', average_only=True)
        self.logger.log_tabular('ClipFrac', average_only=True)
        self.logger.log_tabular('StopIter', average_only=True)
        self.logger.dump_tabular()

    def compute_loss_pi(self, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute loss for PPO policy.

        Args:
            data: properties from each timestep in replay buffer
        Returns:
            loss, statistics for logging
        """
        obs, act, adv, logp_old, mask = data['obs'], data['act'], data['adv'], data['logp'], data['mask']
        
        print("obs: ", obs.shape, "act: ", act.shape, "adv: ", adv.shape, "logp_old: ", logp_old.shape, "mask: ", mask.shape)
        # Policy loss
        pi, logp = self._model.pi.forward(obs, mask, act)
        print("pi: ", pi.shape, "logp: ", logp.shape)
        ratio = torch.exp(logp - logp_old)
        print("ratio: ", ratio.shape)
        clip_adv = torch.clamp(ratio, 1-self._clip_ratio, 1+self._clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self._clip_ratio) | ratio.lt(1-self._clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_v(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute value loss.

        Args:
            data: properties from each timestep in replay buffer
        Returns:
            loss
        """
        obs, ret, mask = data['obs'], data['ret'], data['mask']
        return ((self._model.v(obs, mask) - ret)**2).mean()