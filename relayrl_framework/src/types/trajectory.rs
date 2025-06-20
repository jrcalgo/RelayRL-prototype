//! This module provides utilities for serializing and sending trajectories as well as defining
//! the RelayRLTrajectory type and trait. It uses serde_pickle for serialization and ZMQ for sending
//! the serialized data to a trajectory server.

use crate::sys_utils::config_loader::ConfigLoader;
use crate::types::action::RelayRLAction;
use crate::types::action::RelayRLActionTrait;
use serde::{Deserialize, Serialize};
use serde_pickle as pickle;
use std::io::Cursor;

#[cfg(feature = "zmq_network")]
use zmq;
#[cfg(feature = "zmq_network")]
use zmq::{Context, Socket};

#[cfg(any(
    feature = "networks",
    feature = "grpc_network",
    feature = "zmq_network",
    feature = "python_bindings"
))]
use crate::bindings::python::o3_trajectory::PyRelayRLTrajectory;

/// Trait that defines the interface for trajectory handling.
///
/// Any trajectory struct must implement this trait, which requires a method to add an action
/// to the trajectory. The method may also send the trajectory if a terminal action is encountered.
pub trait RelayRLTrajectoryTrait {
    /// The associated action type that this trajectory holds.
    type Action: RelayRLActionTrait;
    /// Adds an action to the trajectory.
    ///
    /// # Arguments
    ///
    /// * `action` - A reference to the action to be added.
    /// * `send_if_done` - A boolean flag that, if true and the action is terminal, will trigger sending the trajectory.
    fn add_action(&mut self, action: &Self::Action, send_if_done: bool);
}

/// Serializes a given trajectory into a vector of bytes using serde_pickle.
///
/// # Arguments
///
/// * `trajectory` - A reference to the trajectory object that implements Serialize.
///
/// # Returns
///
/// A vector of bytes representing the serialized trajectory.
pub fn serialize_trajectory<T: Serialize>(trajectory: &T) -> Vec<u8> {
    let mut buf = Cursor::new(Vec::new());
    pickle::to_writer(&mut buf, trajectory, Default::default())
        .expect("Failed to serialize trajectory");
    buf.into_inner()
}

/// Sends a serialized trajectory to the specified trajectory server via a ZMQ PUSH socket.
///
/// # Arguments
///
/// * `trajectory` - A reference to the trajectory object that implements Serialize.
/// * `trajectory_server` - A string slice representing the address of the trajectory server.
///
/// # Returns
///
/// * `Result<(), zmq::Error>` - Ok(()) if the trajectory was sent successfully, or an error otherwise.
///
#[cfg(feature = "zmq_network")]
pub fn send_trajectory<T: Serialize>(
    trajectory: &T,
    trajectory_server: &str,
) -> Result<(), zmq::Error> {
    let serialized_trajectory: Vec<u8> = serialize_trajectory(trajectory);

    let context: Context = zmq::Context::new();
    let socket: Socket = context.socket(zmq::PUSH)?;
    socket
        .set_sndtimeo(0)
        .expect("[RelayRLTrajectory - send] Failed to set send timeout");

    socket
        .connect(trajectory_server)
        .expect("[RelayRLTrajectory - send] Failed to connect to trajectory server");

    socket
        .send(serialized_trajectory, 0)
        .expect("[RelayRLTrajectory - send] Failed to send trajectory");

    Ok(())
}

/// The RelayRLTrajectory struct represents a trajectory composed of a sequence of actions.
///
/// It stores an optional trajectory server address, a maximum trajectory length, and a vector of actions.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RelayRLTrajectory {
    /// The optional address of the trajectory server.
    trajectory_server: Option<String>,
    /// The maximum number of actions allowed in the trajectory.
    max_length: u32,
    /// A vector storing the actions in the trajectory.
    pub actions: Vec<RelayRLAction>,
}

impl RelayRLTrajectory {
    /// Creates a new RelayRLTrajectory.
    ///
    /// # Arguments
    ///
    /// * `max_length` - An optional maximum trajectory length. Must be provided.
    /// * `trajectory_server` - An optional trajectory server address.
    ///
    /// # Returns
    ///
    /// A new instance of RelayRLTrajectory.
    pub fn new(max_length: Option<u32>, trajectory_server: Option<String>) -> Self {
        let traj_server = match trajectory_server {
            Some(_) => trajectory_server,
            None => None,
        };

        let max_length: u32 = max_length
            .unwrap_or_else(|| *ConfigLoader::get_max_traj_length(&ConfigLoader::new(None, None)));
        println!(
            "[RelayRLTrajectory] New {:?} length trajectory created",
            max_length
        );

        RelayRLTrajectory {
            trajectory_server: traj_server,
            max_length,
            actions: Vec::new(),
        }
    }

    /// Converts the RelayRLTrajectory into its Python wrapper representation.
    ///
    /// # Returns
    ///
    /// A PyRelayRLTrajectory that wraps the current trajectory.
    #[cfg(any(
        feature = "networks",
        feature = "grpc_network",
        feature = "zmq_network",
        feature = "python_bindings"
    ))]
    pub fn into_py(self) -> PyRelayRLTrajectory {
        PyRelayRLTrajectory {
            inner: RelayRLTrajectory {
                trajectory_server: self.trajectory_server,
                max_length: self.max_length,
                actions: self.actions,
            },
        }
    }
}

/// Implementation of the RelayRLTrajectoryTrait for RelayRLTrajectory.
///
/// This implementation defines how an action is added to the trajectory. If the trajectory reaches
/// its maximum length and the send_if_done flag is set along with the action being terminal,
/// the trajectory is sent to the training server and then cleared.
impl RelayRLTrajectoryTrait for RelayRLTrajectory {
    type Action = RelayRLAction;

    /// Adds an action to the trajectory and conditionally sends it if the trajectory is full and the action is terminal.
    ///
    /// # Arguments
    ///
    /// * `action` - A reference to the RelayRLAction to be added.
    /// * `send_if_done` - A flag indicating whether to send the trajectory if the action's done flag is true.
    fn add_action(&mut self, action: &RelayRLAction, zmq_send_if_done: bool) {
        let action_done: bool = action.done;
        let traj_server: &str = match &self.trajectory_server {
            Some(server) => server.as_str(),
            None => "",
        };

        self.actions.push(action.to_owned());

        #[cfg(feature = "zmq_network")]
        {
            if action_done && zmq_send_if_done {
                println!("[RelayRLTrajectory - action_done] Sending to TrainingServer");

                // Send the trajectory to the training server
                if let Err(e) = send_trajectory(&self.actions, traj_server) {
                    eprintln!(
                        "[RelayRLTrajectory - action_done] Failed to send trajectory: {}",
                        e
                    );
                }
            }
        }

        if action_done
            && self.actions.len()
                >= <u32 as TryInto<usize>>::try_into(self.max_length)
                    .expect("Failed to convert max_length to usize")
        {
            self.actions.clear();
        }
    }
}
