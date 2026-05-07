patent-benchmark/
├── config.yaml
├── data/
│   ├── raw/              ← what you download from USPTO/TraceParts
│   ├── processed/        ← post-normalization images
│   └── manifest.csv      ← the spine CSV (Module 1 output)
├── candidates/
│   └── candidates.csv    ← (Module 2 output)
├── prompts/
│   ├── system.txt
│   ├── zero_shot.txt
│   ├── few_shot.txt
│   └── chain_of_thought.txt
├── raw_responses/        ← (Module 4 output, per-call JSON)
├── results/
│   ├── parsed.csv        ← (Module 5)
│   └── stats/            ← (Module 6: F1, kappa, McNemar, etc.)
├── scripts/
│   ├── 01_pull_uspto.py
│   ├── 02_pull_negative.py
│   ├── 03_normalize.py
│   ├── 04_build_manifest.py
│   ├── 05_select_candidates.py
│   ├── 06_run_models.py
│   ├── 07_parse_responses.py
│   └── 08_compute_stats.py
└── README.md