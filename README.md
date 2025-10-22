# EffiBench-X

<div>
    <h4 align="center">
        <a href="https://arxiv.org/abs/2505.13004" target="_blank">
            <img src="https://img.shields.io/badge/arXiv-2505.13004-b31b1b.svg">
        </a>
        <a href="https://huggingface.co/datasets/EffiBench/effibench-x" target="_blank">
            <img src="https://img.shields.io/badge/HF%20Dataset-effibench--x-FFD21E.svg?logo=huggingface&logoColor=black">
        </a>
        <a href="LICENSE">
            <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
        </a>
    </h4>
    
</div>

Official codebase for our paper: **EffiBench-X: A Multi-Language Benchmark for Measuring Efficiency of LLM-Generated Code**.

**EffiBench-X** is a benchmarking platform for evaluating code generation capabilities of Large Language Models (LLMs), with a focus on runtime and memory efficiency. It executes solutions in a sandboxed environment, measuring runtime, memory usage, and execution success.

<p align="left">
    ✨&nbsp;<a href="#-features">Features</a>
    | 📦&nbsp;<a href="#-installation">Installation</a>
    | 🚀&nbsp;<a href="#-quick-start">Quick Start</a>
    | 🙏&nbsp;<a href="#-acknowledgments">Acknowledgments</a>
    | ⚖️&nbsp;<a href="#️-license">License</a>
    | 📚&nbsp;<a href="#-citation">Citation</a>
</p>

## ✨ Features

- **Comprehensive Benchmarking**: Evaluate LLM code generation not only for correctness but also for efficiency metrics (runtime, memory usage)
- **Multiple Language Support**: Test solutions in Python, JavaScript, C++, Java, Go, and Ruby
- **Flexible Backends**: Run evaluations using isolated Docker execution environments
- **Model Integration**: Support for both open-source and proprietary LLMs (OpenAI, Anthropic, Google, DeepSeek, Qwen, Gemma, etc.)
- **Extensive Dataset**: Problems from multiple sources (LeetCode, AtCoder, CodeChef, Codeforces, etc.)
- **Performance Analysis**: Generate detailed reports and comparisons between different models

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/EffiBench/EffiBench-X.git
cd EffiBench-X

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Managing Datasets

```bash
# Download dataset from Hugging Face Hub
python hf_dataset.py download
```

### Start the Sandbox Backend

```bash
# Start with Docker backend
python start_sandbox.py --type docker --host 127.0.0.1 --port 8000
```

### Generate Solutions

```bash
# Generate solutions for all models in the config file
python generate_solution.py generate data/dataset data/solutions --config model_config.yaml

# Merge canonical solutions
python generate_solution.py merge-canonical-solutions
```

### Evaluate Solutions

```bash
# Evaluate solutions with multiple processes and threads
python evaluate_solution.py evaluate -o data/evaluation

# Generate evaluation report
python evaluate_solution.py report
```

## 🙏 Acknowledgments

- [llm-sandbox](https://github.com/vndee/llm-sandbox) — customized version included under `third_party/llm-sandbox` (MIT).
- Problem sources and inspirations:
  - [LeetCode](https://leetcode.com)
  - [AtCoder](https://atcoder.jp)
  - [CodeChef](https://www.codechef.com)
  - [Codeforces](https://codeforces.com)
  - [Aizu Online Judge (AOJ)](https://onlinejudge.u-aizu.ac.jp/)

## ⚖️ License

EffiBench-X is licensed under the Apache License 2.0; portions are available under separate terms. The component at [third_party/llm-sandbox](third_party/llm-sandbox) is licensed under MIT (see its LICENSE).

## 📚 Citation

Please kindly consider citing our paper if you find this repository helpful in your research and work.

```bibtex
@article{qing2025effibench,
  title={EffiBench-X: A Multi-Language Benchmark for Measuring Efficiency of LLM-Generated Code},
  author={Qing, Yuhao and Zhu, Boyu and Du, Mingzhe and Guo, Zhijiang and Zhuo, Terry Yue and Zhang, Qianru and Zhang, Jie M and Cui, Heming and Yiu, Siu-Ming and Huang, Dong and Ng, See-Kiong and Tuan, Luu Anh},
  journal={Advances in neural information processing systems},
  year={2025}
}
```
