# ctrlg_custom_vllm

Custom CTRL-G logits processors for vLLM.

## Installation

```bash
pip install -e .
```

## Usage

```python
from ctrlg_custom_vllm import Qwen25VLBaseCtrlgProcessorV0
```

## Enable ctrlg for EasyR1
You need to install these two packages

1. ctrlg_custom_vllm
```
git clone https://github.com/billkunghappy/verl.git

cd verl/custom_ctrlg_vllm
pip install -e .
```
Note that you don't need to install the verl here. You only need to install the ctrlg_custom_vllm inside it


2. ctrlg (from repo: ctrlg_for_vllm)
```
git clone https://github.com/billkunghappy/ctrlg_for_vllm.git
cd ctrlg_for_vllm
pip install -e .
```