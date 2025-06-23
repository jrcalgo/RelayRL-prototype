# RelayRL Prototype

**Prototype Distributed Reinforcement Learning Framework**

> **Platform Support:**
> RelayRL runs on **MacOS, Linux, and Windows** (x86_64). Some dependencies may require additional setup on Windows.

> **Warning:**  
> This project is a **prototype** and is **unstable during training**. Use at your own risk and expect breaking changes.

RelayRL is a high-performance, distributed reinforcement learning (RL) framework designed for research and experimentation. By using proven transport-layer communication protocols, RelayRL integrates a client-server architecture. It provides a robust Rust backend with Python bindings, enabling seamless integration with modern ML workflows and high-speed system-level orchestration. 

## Features

- **Distributed RL Orchestration:**  
  Run agents and training servers as separate processes, communicating via gRPC or ZeroMQ.
- **Python & Rust Interoperability:**  
  Use Python's PyTorch for RL algorithms and Rust for orchestration and performance-critical components.
- **Extensible Algorithm Support:**  
  Plug in your own RL algorithms in Python (see `examples/` for Jupyter notebooks).
- **Benchmarking & Profiling:**  
  Tools and scripts for measuring network and runtime performance.
- **Configurable Experiments:**  
  JSON-based configuration for easy experiment management.

## Quick Start

**Supported Platforms:** MacOS, Linux, Windows (x86_64)

1. **Clone the repository:**
   ```sh
   git clone https://github.com/jrcalgo/RelayRL-prototype.git
   cd RelayRL-prototype
   ```

2. **Build the Rust framework:**
   ```sh
   cd relayrl_framework
   cargo build --release
   ```

3. **Install Python bindings (from the `relayrl_framework` directory):**
   ```sh
   pip install torch==2.5.1
   pip install maturin
   maturin develop --release
   ```

4. **Run an example (see `examples/`):**
   - Jupyter notebooks demonstrate how to use the Python API for training and evaluation.

## Single-Agent Focus

RelayRL is **primarily designed for single-agent RL**.  
While it is theoretically possible to launch multiple agent and training server processes for multi-agent scenarios, the framework does **not** natively support scalable multi-agent orchestration at this time.

## Directory Structure

- `relayrl_framework/` — Core Rust framework and Python bindings
- `examples/` — Example scripts and Jupyter notebooks (Python API usage)
- `README.md` — This file (project overview)
- `relayrl_framework/README.md` — Framework-level and development instructions

## Future Work (Currently in-progress in a separate repository)

- Improved multi-agent support and orchestration by enhancing modularity
- Enhanced stability, scalability and error handling during training through comprehensive network error handling, backpressure, etc.
- Exposing scalable components for containerized automation
- Easing execution and external application integration through compilation into CLI executables
- More built-in RL algorithms available
- Better documentation and API reference

## Disclaimer

This project is a **prototype** and is **unstable during training**.  
APIs and features are subject to change.  
Contributions and feedback are welcome!

## License

[MIT](LICENSE)
