# RelayRL — Prototype Distributed Reinforcement Learning Framework

> **Prototype Warning:**  
> This repository contains an early-stage prototype. The framework is unstable during training, and intended primarily for research and experimentation.
> 
> **Platform Support:**
> RelayRL runs on **MacOS, Linux, and Windows** (x86_64). Some dependencies may require additional setup on Windows.

RelayRL aims to be a high-performance, distributed reinforcement learning (RL) framework designed to explore single-agent RL with robust system-level orchestration. It provides a Rust backend for performance-critical components, Python bindings for integration with ML workflows, and a flexible client-server architecture using gRPC or ZeroMQ.

## Prototype Scope

RelayRL is currently **focused on single-agent RL**. Multi-agent training is not fully supported—while you can launch multiple agents manually, the framework does not yet provide scalable multi-agent orchestration.

This repository reflects early-stage development: APIs, performance optimizations, and features are experimental. The codebase is a **research prototype** rather than a production-ready framework.

## Prototype Features

- **Distributed RL Orchestration:**
  Run agents and training servers as separate processes, communicating via gRPC or ZeroMQ.
- **Python & Rust Interoperability:**
  Natively supports built-in or custom Python PyTorch for RL algorithms and Rust for orchestration and performance-critical components.
- **Extensible Algorithm/Environment Support:**
  Plug in your own RL algorithms/environments in Python (see `examples/` for Jupyter notebooks).
- **Benchmarking & Profiling:**
  Tools and scripts for measuring network and runtime performance.
- **Configurable Experiments:**
  JSON-based configuration for easy experiment management.

## Quick Start

**Install RelayRL Framework:**
  ```sh
  pip install relayrl_framework
  ```

## Build From Source

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
  
## Tutorial

[See the examples' README!](https://github.com/jrcalgo/RelayRL-proto/blob/main/examples/README.md#how-to-use-in-novel-environments)

## Directory Structure

- `relayrl_framework/` — Core Rust framework and Python bindings
- `examples/` — Example scripts and Jupyter notebooks (Python API usage)
- `README.md` — This file (project overview)
- `relayrl_framework/README.md` — Framework-level and development instructions
- `examples/README.md` — Python integration and API utilization instructions

## Prototype Roadmap

- Better documentation and API reference
- Benchmarking against known frameworks (RayRLLib, Stable-Baselines3, CleanRL, etc.)
- Preservation of most critical components/logic

## RelayRL Framework Roadmap (Merged into a [monorepo](https://github.com/jrcalgo/RelayRL))

- Improved multi-agent support and orchestration by enhancing component modularity
- Enhanced stability, scalability and error handling during training through comprehensive network error handling, backpressure, configurable components, etc.
- Real-time configuration updates and shutdown/init signals through lifecycle control
- Exposing scalable components for containerized automation
- Easing execution and external application integration through compilation into CLI executables
- More built-in RL algorithms available
- And likely more to come...

## Disclaimer

This framework is a research prototype. gRPC and ZMQ transport loops are functional, but ZMQ is currently more stable. 
Features, APIs, and implementations are experimental and likely to change in future versions.
Contributions and feedback are welcome!

## License

[Apache License 2.0](LICENSE)
