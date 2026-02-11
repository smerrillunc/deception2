# Deception2 / Werewolf

Standalone Werewolf environment + deception miner.

## Layout
- `src/`: Werewolf environment and miner
- `shell_scripts/`: runnable helper scripts
- `Notebooks/`: demo notebook
- `Results/`: output location (ignored in git)

## Run miner
```bash
python Werewolf/src/werewolf_deception_miner.py \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --is_reasoning_model \
  --output_dir Werewolf/Results/DeceptionMining/demo
```

or

```bash
Werewolf/shell_scripts/run_werewolf_deception_miner.sh
```

## Demo notebook
- `Werewolf/Notebooks/werewolf_environment_demo.ipynb`
