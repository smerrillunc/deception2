# Localization Sampling Budget Summary

- Examples analyzed: 100
- Reference budget: 100 continuations per prefix
- Case-study example: CarSales / 2026-03-26/shard_01/game_73/turn_0/state_48/sample_23 (9 sentences)

## 50 vs 100
- Mean absolute error vs reference: 0.0333 [0.0326, 0.0340]
- Fraction within 0.05: 0.765 [0.756, 0.773]
- Fraction within 0.10: 0.961 [0.958, 0.963]
- Peak-deception sentence agreement: 0.556 exact, 0.798 within one sentence
- Largest positive jump agreement (reference jump >= 0.3; 50/100 eligible examples): 0.930 exact, 0.972 within one sentence
