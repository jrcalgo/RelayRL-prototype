# RelayRL Framework

**Core Rust Library & Python Bindings for RelayRL**

---

> **Platform Support:**
> RelayRL runs on **MacOS, Linux, and Windows** (x86_64). Some dependencies may require additional setup on Windows.

> **Warning:**  
> This is a **prototype** and is **unstable during training**.  
> For general usage and project overview, see the [root README](../README.md).

---

## Overview

The `relayrl_framework` crate provides the core infrastructure for distributed RL experiments, including:

- High-performance agent and training server implementations (gRPC & ZMQ)
- Core RL data structures (actions, trajectories, configs)
- Python bindings via PyO3 for seamless integration with Python RL algorithms
- Utilities for configuration, logging, and benchmarking

---

## Building & Development

### Prerequisites

- **Supported Platforms:** MacOS, Linux, Windows (x86_64)
- Rust (latest stable recommended)
- Python 3.8+
- PyTorch 2.5.1
- [maturin](https://github.com/PyO3/maturin) (for building Python bindings)
- Protobuf compiler (`protoc`) for gRPC

### Build the Rust Library

```sh
cargo build --release
```

### Build & Install Python Bindings

From the `relayrl_framework` directory:

```sh
pip install maturin
maturin develop
```

This will build the Rust library and install the Python bindings into your current environment.

---

## Running Benchmarks

Benchmarks are provided in the `benches/` directory:

```sh
cargo bench
```

You can also use the provided shell scripts in `scripts/` for profiling and release builds.

---

## Development Notes

- **Python Bindings:**  
  Expose `RelayRLAction`, `RelayRLTrajectory`, `ConfigLoader`, and agent/server classes to Python.
- **Configuration:**  
  Use JSON files to specify experiment parameters, network settings, and logging options.
- **Extending Algorithms:**  
  Add new Python RL algorithms in `src/native/python/algorithms/` and use them via the Python API.

---

## Limitations

- **Single-Agent Focus:**  
  The framework is designed for single-agent RL. Multi-agent support is not natively implemented, but you may launch multiple agent and training server processes for experimentation.
- **Prototype Status:**  
  The framework is unstable during training and is under active development. Expect breaking changes and incomplete features.

---

## Contributing

Contributions are welcome! Please open issues or pull requests for bug reports, feature requests, or improvements.

---

## License

[MIT](../LICENSE)