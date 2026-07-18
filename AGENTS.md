# AGENTS.md

## Cursor Cloud specific instructions

This is a single-product, self-contained Python reinforcement-learning CLI (Conveyor System RL / zipper-merge). There is no web server, database, or other networked service — everything runs as a local Python process producing `.zip` models and `.gif` visualizations. See `README.md` for full usage.

### Environment / how to run
- Dependencies live in a virtualenv at `venv/` (gitignored). The startup update script keeps it current. Run everything with `venv/bin/python` (there is no bare `python` on the VM, only `python3`).
- The one-off system package `python3-venv` (and `python3-pip`) was installed via apt so `python3 -m venv` works; this persists in the VM snapshot, so it is not part of the update script.
- Entry point is `run.py` with subcommands: `train`, `demo`, `visualize`, `compare`, `evaluate`, plus network-graph commands `network-train`, `network-demo`, `network-evaluate`, `network-visualize`. Example: `venv/bin/python run.py evaluate --model models/conveyor_ppo_final`.
- Graph conveyor networks: presets `merge` / `diamond` / `triple` or custom JSON under `graphs/`. Train a small throughput policy with e.g. `venv/bin/python run.py network-demo --preset merge` (writes under `models/network/` and `network_demo.gif`).
- There is no test suite, no linter config, and no build step. The de-facto end-to-end smoke test is `venv/bin/python run.py demo` (trains 50k steps, then evaluates/visualizes/compares).

### Gotchas
- `train`/`demo` require `tensorboard`, `tqdm`, and `rich` (from `train.py` using `tensorboard_log=...` and `progress_bar=True`). These are pulled in via `stable-baselines3[extra]` in `requirements.txt`; plain `stable-baselines3` is not enough to train.
- `run.py` calls `os.chdir` to its own directory and several commands write to fixed paths in the repo: `visualize` defaults to `rollout.gif`, `compare` overwrites `comparisons/{trained,random}_policy.gif`, and `train`/`demo` overwrite `models/conveyor_ppo_final.zip` and `demo_comparison/`. To avoid dirtying committed files, point `--output`/`--model`/`save_path` at a temp location, or `git checkout --` the regenerated files afterward.
- A committed pre-trained model exists at `models/conveyor_ppo_final.zip`, so `evaluate`/`visualize`/`compare` work immediately without training (it achieves ~100% zipper ratio).
- Training runs on CPU (no GPU required); a short run (a few thousand timesteps) is enough to exercise the pipeline but will not learn the zipper behavior.
