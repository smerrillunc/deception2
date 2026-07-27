# localization_fulltrace_vs_adaptive_rebuttal_v1

Deterministic rebuttal subset for comparing dataset adaptive localization
against newly run full-trace localization.

Generated artifacts:
- `selected_examples.csv`: one row per chosen example.
- `bundle_summary.csv`: eligible vs selected counts per environment/model bundle.
- `run_manifest.csv`: full localization jobs to launch.
- `run_manifest_full.csv`: alias of `run_manifest.csv` for compatibility.
- `bundles/<env>__<model>/examples.jsonl`: selected examples.
- `bundles/<env>__<model>/sentences.jsonl`: matching sentence records.
- `selected_examples.csv[source_localization_relpath]`: dataset adaptive localization files used as the comparison baseline.

Local refresh:
```bash
python rebuttal/scripts/prepare_localization_fulltrace_rebuttal.py --run-name localization_fulltrace_vs_adaptive_rebuttal_v1
```

Longleaf sequence:
```bash
cd /work/users/s/m/smerrill/deception2
python rebuttal/scripts/prepare_localization_fulltrace_rebuttal.py --run-name localization_fulltrace_vs_adaptive_rebuttal_v1
bash rebuttal/slurm/submit_localization_fulltrace_rebuttal.sh localization_fulltrace_vs_adaptive_rebuttal_v1 full
```
