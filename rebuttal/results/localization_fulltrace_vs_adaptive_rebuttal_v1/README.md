# localization_fulltrace_vs_adaptive_rebuttal_v1

Deterministic rebuttal subset for adaptive-vs-full sentence localization.

Generated artifacts:
- `selected_examples.csv`: one row per chosen example.
- `bundle_summary.csv`: eligible vs selected counts per environment/model bundle.
- `run_manifest.csv`: all adaptive/full jobs.
- `run_manifest_full.csv`: full-only jobs.
- `run_manifest_adaptive.csv`: adaptive-only jobs.
- `bundles/<env>__<model>/examples.jsonl`: selected examples.
- `bundles/<env>__<model>/sentences.jsonl`: matching sentence records.

Local refresh:
```bash
python rebuttal/scripts/prepare_localization_fulltrace_rebuttal.py --run-name localization_fulltrace_vs_adaptive_rebuttal_v1
```

Longleaf sequence:
```bash
cd /work/users/s/m/smerrill/deception2
python rebuttal/scripts/prepare_localization_fulltrace_rebuttal.py --run-name localization_fulltrace_vs_adaptive_rebuttal_v1
bash rebuttal/slurm/submit_localization_fulltrace_rebuttal.sh localization_fulltrace_vs_adaptive_rebuttal_v1 adaptive
bash rebuttal/slurm/submit_localization_fulltrace_rebuttal.sh localization_fulltrace_vs_adaptive_rebuttal_v1 full
```
