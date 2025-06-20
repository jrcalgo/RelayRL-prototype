//! This module provides a Python wrapper around the RelayRL `ConfigLoader` struct using PyO3.
//! It allows Python code to instantiate and query configuration parameters such as algorithm
//! settings, server addresses, tensorboard parameters, and model paths.

use pyo3::Bound;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::{pyclass, pymethods};
use std::path::PathBuf;

use crate::sys_utils::config_loader::{
    ConfigLoader, LoadedAlgorithmParams, ServerParams, TensorboardParams,
};
use crate::sys_utils::misc_utils::round_to_8_decimals;
use std::sync::{Arc, Mutex, MutexGuard};

/// A Python-exposed struct that wraps the RelayRL `ConfigLoader`.
///
/// This class holds an `Arc<Mutex<conf_loader::ConfigLoader>>`, ensuring that the Rust-side
/// configuration loader is thread-safe and can be accessed concurrently from Python.
#[pyclass(name = "ConfigLoader")]
pub struct PyConfigLoader {
    /// An `Arc<Mutex<conf_loader::ConfigLoader>>` that provides synchronized, shared ownership
    /// of the underlying RelayRL ConfigLoader.
    inner: Arc<Mutex<ConfigLoader>>,
}

#[pymethods]
impl PyConfigLoader {
    /// Creates a new `ConfigLoader` Python object.
    ///
    /// # Arguments
    ///
    /// * `algorithm_name` - An optional string specifying which algorithm's parameters to load.
    /// * `config_path` - An optional path to the JSON configuration file. If none is provided,
    ///   a default path is used.
    ///
    /// # Returns
    ///
    /// A Python-wrapped `ConfigLoader` instance.
    #[new]
    #[pyo3(signature = (algorithm_name = None, config_path = None))]
    fn new(algorithm_name: Option<String>, config_path: Option<String>) -> Self {
        let config_path: Option<PathBuf> = match config_path {
            Some(path) => Some(PathBuf::from(path)),
            None => None,
        };

        PyConfigLoader {
            inner: Arc::new(Mutex::new(ConfigLoader::new(algorithm_name, config_path))),
        }
    }

    /// Retrieves algorithm-specific parameters from the loaded configuration, if available.
    ///
    /// If the algorithm is set to a recognized variant (e.g., `DQN`, `PPO`, `REINFORCE`, `SAC`),
    /// this method returns a Python dictionary containing the algorithm fields.
    /// Otherwise, it returns `None`.
    ///
    /// # Returns
    /// * `Option<PyObject>` - A dictionary of algorithm parameters or `None` if not set/found.
    #[pyo3(signature = ())]
    fn get_algorithm_params(&self, py: Python) -> PyResult<Option<PyObject>> {
        let config: MutexGuard<ConfigLoader> = self
            .inner
            .lock()
            .expect("Failed to lock `inner` configloader");
        let algorithm_params: &Option<LoadedAlgorithmParams> = config.get_algorithm_params();

        if let Some(algorithm_config) = algorithm_params {
            let dict: Bound<PyDict> = PyDict::new(py);

            // C51
            if let LoadedAlgorithmParams::C51(c51_params) = algorithm_config {
                let c51_dict: Bound<PyDict> = PyDict::new(py);
                c51_dict.set_item("batch_size", c51_params.batch_size)?;
                c51_dict.set_item("seed", c51_params.seed)?;
                c51_dict.set_item("traj_per_epoch", c51_params.traj_per_epoch)?;
                c51_dict.set_item("n_atoms", c51_params.n_atoms)?;
                c51_dict.set_item("v_min", round_to_8_decimals(c51_params.v_min))?;
                c51_dict.set_item("v_max", round_to_8_decimals(c51_params.v_max))?;
                c51_dict.set_item("gamma", round_to_8_decimals(c51_params.gamma))?;
                c51_dict.set_item("epsilon", round_to_8_decimals(c51_params.epsilon))?;
                c51_dict.set_item("epsilon_min", round_to_8_decimals(c51_params.epsilon_min))?;
                c51_dict.set_item(
                    "epsilon_decay",
                    round_to_8_decimals(c51_params.epsilon_decay),
                )?;
                c51_dict.set_item("train_update_freq", c51_params.train_update_freq)?;
                c51_dict.set_item("target_update_freq", c51_params.target_update_freq)?;
                c51_dict.set_item("q_lr", round_to_8_decimals(c51_params.q_lr))?;
                c51_dict.set_item("train_q_iters", c51_params.train_q_iters)?;
                dict.set_item("C51", c51_dict)?;
            }

            // DQN
            if let LoadedAlgorithmParams::DQN(dqn_params) = algorithm_config {
                let dqn_dict: Bound<PyDict> = PyDict::new(py);
                dqn_dict.set_item("batch_size", dqn_params.batch_size)?;
                dqn_dict.set_item("seed", dqn_params.seed)?;
                dqn_dict.set_item("traj_per_epoch", dqn_params.traj_per_epoch)?;
                dqn_dict.set_item("gamma", dqn_params.gamma)?;
                dqn_dict.set_item("epsilon", dqn_params.epsilon)?;
                dqn_dict.set_item("epsilon_min", dqn_params.epsilon_min)?;
                dqn_dict.set_item("epsilon_decay", dqn_params.epsilon_decay)?;
                dqn_dict.set_item("train_update_freq", dqn_params.train_update_freq)?;
                dqn_dict.set_item("q_lr", dqn_params.q_lr)?;
                dqn_dict.set_item("train_q_iters", dqn_params.train_q_iters)?;
                dict.set_item("DQN", dqn_dict)?;
            }

            // PPO
            if let LoadedAlgorithmParams::PPO(ppo_params) = algorithm_config {
                let ppo_dict: Bound<PyDict> = PyDict::new(py);
                ppo_dict.set_item("seed", ppo_params.seed)?;
                ppo_dict.set_item("traj_per_epoch", ppo_params.traj_per_epoch)?;
                ppo_dict.set_item("clip_ratio", ppo_params.clip_ratio)?;
                ppo_dict.set_item("gamma", ppo_params.gamma)?;
                ppo_dict.set_item("lam", ppo_params.lam)?;
                ppo_dict.set_item("pi_lr", ppo_params.pi_lr)?;
                ppo_dict.set_item("vf_lr", ppo_params.vf_lr)?;
                ppo_dict.set_item("train_pi_iters", ppo_params.train_pi_iters)?;
                ppo_dict.set_item("train_v_iters", ppo_params.train_v_iters)?;
                ppo_dict.set_item("target_kl", ppo_params.target_kl)?;
                dict.set_item("PPO", ppo_dict)?;
            }

            // REINFORCE
            if let LoadedAlgorithmParams::REINFORCE(reinforce_params) = algorithm_config {
                let reinforce_dict: Bound<PyDict> = PyDict::new(py);
                reinforce_dict.set_item("discrete", reinforce_params.discrete)?;
                reinforce_dict.set_item("with_vf_baseline", reinforce_params.with_vf_baseline)?;
                reinforce_dict.set_item("seed", reinforce_params.seed)?;
                reinforce_dict.set_item("traj_per_epoch", reinforce_params.traj_per_epoch)?;
                reinforce_dict.set_item("gamma", reinforce_params.gamma)?;
                reinforce_dict.set_item("lam", reinforce_params.lam)?;
                reinforce_dict.set_item("pi_lr", reinforce_params.pi_lr)?;
                reinforce_dict.set_item("vf_lr", reinforce_params.vf_lr)?;
                reinforce_dict.set_item("train_vf_iters", reinforce_params.train_vf_iters)?;
                dict.set_item("REINFORCE", reinforce_dict)?;
            }

            // SAC
            if let LoadedAlgorithmParams::SAC(sac_params) = algorithm_config {
                let sac_dict: Bound<PyDict> = PyDict::new(py);
                sac_dict.set_item("discrete", sac_params.discrete)?;
                sac_dict.set_item("adaptive_alpha", sac_params.adaptive_alpha)?;
                sac_dict.set_item("act_dim", sac_params.act_dim)?;
                sac_dict.set_item("batch_size", sac_params.batch_size)?;
                sac_dict.set_item("seed", sac_params.seed)?;
                sac_dict.set_item("traj_per_epoch", sac_params.traj_per_epoch)?;
                sac_dict.set_item("log_std_min", sac_params.log_std_min)?;
                sac_dict.set_item("log_std_max", sac_params.log_std_max)?;
                sac_dict.set_item("gamma", sac_params.gamma)?;
                sac_dict.set_item("polyak", sac_params.polyak)?;
                sac_dict.set_item("alpha", sac_params.alpha)?;
                sac_dict.set_item("lr", sac_params.lr)?;
                sac_dict.set_item("clip_grad_norm", sac_params.clip_grad_norm)?;
                sac_dict.set_item("train_update_freq", sac_params.train_update_freq)?;
                sac_dict.set_item("train_iters", sac_params.train_iters)?;
                dict.set_item("SAC", sac_dict)?;
            }

            // Return the constructed dictionary as a PyObject
            Ok(Some(dict.into()))
        } else {
            // No algorithm params loaded
            Ok(None)
        }
    }

    /// Retrieves the training server parameters as a Python dictionary.
    ///
    /// # Returns
    /// A `PyDict` containing the "prefix", "host", and "port" fields.
    #[pyo3(signature = ())]
    fn get_train_server<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let config: MutexGuard<ConfigLoader> = self
            .inner
            .lock()
            .expect("Failed to lock `inner` configloader");
        let server_params: &ServerParams = &config.train_server;
        let (prefix, host, port): (&str, &str, &str) = (
            server_params.prefix.as_str(),
            server_params.host.as_str(),
            server_params.port.as_str(),
        );

        let dict: Bound<'py, PyDict> = PyDict::new(py);
        dict.set_item("prefix", prefix)?;
        dict.set_item("host", host)?;
        dict.set_item("port", port)?;

        Ok(dict)
    }

    /// Retrieves the trajectory server parameters as a Python dictionary.
    ///
    /// # Returns
    /// A `PyDict` containing the "prefix", "host", and "port" fields.
    #[pyo3(signature = ())]
    fn get_traj_server<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let config = self
            .inner
            .lock()
            .expect("Failed to lock `inner` configloader");
        let server_params: &ServerParams = &config.traj_server;
        let (prefix, host, port): (&str, &str, &str) = (
            server_params.prefix.as_str(),
            server_params.host.as_str(),
            server_params.port.as_str(),
        );

        let dict: Bound<'py, PyDict> = PyDict::new(py);
        dict.set_item("prefix", prefix)?;
        dict.set_item("host", host)?;
        dict.set_item("port", port)?;

        Ok(dict)
    }

    /// Retrieves the tensorboard configuration parameters as a Python dictionary.
    ///
    /// # Returns
    /// A `PyDict` containing the "scalar_tags", "max_count_per_scalar", and "global_step_tag" fields.
    #[pyo3(signature = ())]
    fn get_tb_params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let config: MutexGuard<ConfigLoader> = self
            .inner
            .lock()
            .expect("Failed to lock `inner` configloader");
        let tb_params: &TensorboardParams = &config.tb_params;
        let (launch_tb_on_startup, scalar_tags, global_step_tag): (bool, Vec<String>, &str) = (
            tb_params.launch_tb_on_startup,
            tb_params.scalar_tags.clone(),
            tb_params.global_step_tag.as_str(),
        );

        let dict: Bound<'py, PyDict> = PyDict::new(py);
        dict.set_item("launch_tb_on_startup", launch_tb_on_startup)?;
        dict.set_item("scalar_tags", scalar_tags)?;
        dict.set_item("global_step_tag", global_step_tag)?;

        Ok(dict)
    }

    /// Retrieves the client model path as a string.
    ///
    /// # Returns
    /// A `String` representing the full file path where models should be loaded from.
    #[pyo3(signature = ())]
    fn get_client_model_path(&self) -> PyResult<String> {
        let config: MutexGuard<ConfigLoader> = self
            .inner
            .lock()
            .expect("Failed to lock `inner` configloader");
        Ok(config
            .client_model_path
            .to_str()
            .expect("Failed to get client model path")
            .to_string())
    }

    /// Retrieves the server model path as a string.
    ///
    /// # Returns
    /// A `String` representing the full file path where models should be saved.
    #[pyo3(signature = ())]
    fn get_server_model_path(&self) -> PyResult<String> {
        let config: MutexGuard<ConfigLoader> = self
            .inner
            .lock()
            .expect("Failed to lock `inner` configloader");
        Ok(config
            .server_model_path
            .to_str()
            .expect("Failed to get server model path")
            .to_string())
    }

    /// Retrieves the maximum trajectory length from the configuration.
    ///
    /// # Returns
    /// A `u32` specifying the maximum allowed trajectory length.
    #[pyo3(signature = ())]
    fn get_max_traj_length(&self) -> PyResult<u32> {
        let config: MutexGuard<ConfigLoader> = self
            .inner
            .lock()
            .expect("Failed to lock `inner` configloader");
        Ok(config.max_traj_length)
    }
}
