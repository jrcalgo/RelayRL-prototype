# RelayRL Example Notebooks

This directory contains example Jupyter notebooks demonstrating how to use the RelayRL framework with different reinforcement learning algorithms, environments, and communication backends (gRPC and ZMQ).

## Directory Structure

```
examples/
  gRPC/
    PPO/
      classic_control/
        cartpole/
          cartpole.ipynb
        mountain_car/
          mountain_car.ipynb
      box2d/
        lunar_lander/
          lunar_lander.ipynb
    REINFORCE/
      classic_control/
        cartpole/
          cartpole.ipynb
        mountain_car/
          mountain_car.ipynb
      box2d/
        lunar_lander/
          lunar_lander.ipynb
  ZMQ/
    PPO/
      classic_control/
        cartpole/
          cartpole.ipynb
        mountain_car/
          mountain_car.ipynb
      box2d/
        lunar_lander/
          lunar_lander.ipynb
    REINFORCE/
      classic_control/
        cartpole/
          cartpole.ipynb
        mountain_car/
          mountain_car.ipynb
      box2d/
        lunar_lander/
          lunar_lander.ipynb
  README.md
```

## How to Use

1. **Install dependencies** as described in the main project README.
2. **Open any notebook** (e.g., `gRPC/PPO/classic_control/cartpole/cartpole.ipynb`) in Jupyter or VSCode.
3. **Run the cells** to launch the training server and start the RL training loop using the RelayRL Python bindings.

## Adding New Environments

To add a new environment (e.g., BipedalWalker or CartPole):
- Copy one of the existing notebooks (e.g., `lunar_lander.ipynb` or `cartpole.ipynb`).
- Update the `env = gym.make(...)` line to your desired environment.
- Set the correct observation and action dimensions in the `TrainingServer` and `RelayRLAgent` initialization.
- Place the notebook in the appropriate directory reflecting the backend (gRPC or ZMQ), algorithm (PPO or REINFORCE), and environment type (classic_control, box2d, etc).

## Notes

- All examples use the RelayRL Python API, which interfaces with the underlying Rust implementation.
- If you encounter missing dependencies, ensure your Python environment matches the requirements in `relayrl_framework/src/native/python/requirements`.
- The structure supports easy extension to new environments and algorithms.

---

For more details, see the main project [README](../README.md).
