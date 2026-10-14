# CPU continuation, 2026-10-12

User authorized completing offline preparation and requested notification when a GPU instance is required. No training, simulation or competition submission is launched.

Compliance: only pinned release clean HDF5 and seen instructions; zero simulator calls; no robotwin post-trained starting model/teacher. No new demonstrations, instructions, routing or model modifications.

The conversion space guard now checks the output data filesystem instead of the system disk. Official conversion keeps next-frame actions, three 640x480 cameras and 50 FPS. Existing outputs and raw sources are preserved.

The training TEMPLATE now explicitly sets gradient_accumulation_steps=32 for a single-process reference (32 microbatch x 1 data parallel x 32 accumulation = 1024). This fixes CPU dataclass parsing, not GPU memory feasibility. The config generator reproduces this setting. GPU launch must select microbatch and accumulation for actual hardware while checking the global batch equation; no training has used this template. Original files are retained under docs/patches/.

The background serial workflow is `bash scripts/run_cpu_continuation.sh`. It holds an exclusive lock, stops on error and writes `results/cpu_continuation_status.json`. Each completed task is checked against all original state/action rows and its per-episode seen instruction pool. Native video decode covers the first/last frame of each task, not every video frame. Reports are in `results/conversion_validation/`. Full statistics run only after all 50 tasks validate. The first task was already running when the coordinator started; its PID is only an initial-job liveness check, not a portable dependency after completion. Partial outputs are never overwritten automatically.

## Future instance transfer

Do not transfer while conversion/statistics are writing. Wait for READY_FOR_GPU, or investigate FAILED first. Preserve both `/root/lingbot_challenge` and `/mnt/lingbot-data/lingbot_challenge`; `data` and `assets` under the project are symlinks into the latter. Environment prefixes, scripts, manifests and configs currently use absolute paths. The Conda installation used for activation is `/home/vipuser/miniconda3`; preserve it or recreate the activation tooling from the recorded environment exports. Keep the existing source revision and package exports in `docs/versions/`. The new host must mount the data disk; `configs/paths.env` validates its current UUID, which must be reviewed if a different disk is used. Do not disable the mount guard to bypass a missing disk.

After the user migrates and supplies the new SSH endpoint, inspect GPU/driver, disk mounts, symlinks and current environments before running `scripts/setup_gpu.sh` and `scripts/check_gpu.sh`. CPU configuration parsing does not establish training capacity, model execution or evaluation success.
