# Lite LLaMA
Goal: Run LLaMA-65B inference on a GTX1070 with **0 quality degradation**.

## How
Transformer inference happens sequentially.
Load weights on-demand, one layer at a time.

## How to Run
```sh
# Make sure LLaMA weights are under the weights/ directory
python3 serialize.py # This will create a new serialized version of the weights
python3 model.py # Run LLaMA
```
