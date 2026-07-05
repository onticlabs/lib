# ontic-lib

Shared, promoted code for Ontic experiments.

## What belongs here

`ontic-lib` is for code that has proven itself across experiments, not for
one-off experiment logic. Concretely:

- **Eval harnesses** — scoring/benchmark code reused across multiple
  experiments or projects.
- **Stable model code** — architectures, layers, or wrappers that have
  stabilized and are no longer being actively iterated on inside a single
  experiment.
- **Dataset loaders** — data loading/preprocessing code shared by more than
  one experiment.
- **Cross-cutting infrastructure**, such as the tracking shim in
  `ontic_lib.tracking`, which fans out metric logging to
  [trackio](https://pypi.org/project/trackio/) (always) and Weights & Biases
  (best-effort, optional) so training runs never crash because a metrics
  SaaS is down.

## The promotion rule

Code starts life inside an experiment's own directory (scaffolded by
`ontic-cli`). It only gets promoted into `ontic-lib` once it has been
**copied into a third experiment** — i.e. it has proven itself reusable at
least twice over. Until then, duplication across two experiments is fine
and expected; premature abstraction is not.

When code is promoted:

1. Move it into `src/ontic_lib/` here, with tests in `tests/`.
2. Update the experiments that copied it to import from `ontic_lib` instead
   of keeping their own copy.
3. Bump the version and tag a release so experiments can pin against it.

## Tracking shim

```python
from ontic_lib.tracking import init

tracker = init("my-project", config={"lr": 0.1})
tracker.log({"loss": 1.0}, step=1)
tracker.finish()
```

- Always logs to `trackio`. `TRACKIO_DIR` defaults to `./output` (relative
  to the current working directory) so the metrics DB lands inside the job
  record; set `TRACKIO_DIR` explicitly beforehand to override.
- Additionally logs to Weights & Biases iff all of the following hold:
  - `wandb` is importable (install the `wandb` extra: `pip install
    ontic-lib[wandb]`),
  - `WANDB_API_KEY` is set or a `~/.netrc` file exists (W&B credentials are
    present), and
  - `ONTIC_WANDB_RUN_ID` is set (the bootstrap sets this from
    `cert.wandb.run_id` so the run resumes by id rather than starting a new
    one).
- Every W&B call (`init`, `log`, `finish`) is wrapped so a W&B/metrics-SaaS
  failure can never crash a training run.

## Checkpointing

```python
from ontic_lib.checkpoint import CheckpointManager

mgr = CheckpointManager("./output/checkpoints", keep_last=3)

# in the training loop
mgr.save(step, {"model": model.state_dict(), "opt": opt.state_dict()},
         run_id=run_id, keep_last=3)

# on resume
result = mgr.resume()
if result is not None:
    state, meta = result
    model.load_state_dict(state["model"])
    start_step = meta["step"]
```

- Backend-agnostic: serialization defaults to `pickle`, but any framework can
  plug in its own format, e.g. `CheckpointManager(dir, save_fn=torch.save,
  load_fn=torch.load)`.
- Writes are atomic (temp file + `os.replace`) so a crash mid-save never
  leaves a corrupt checkpoint behind.
- A `latest.json` pointer file makes `resume()` and `latest_step` O(1) and
  independent of the serialization format.
- `keep_last` prunes older `step_*.ckpt` files, always keeping the one just
  written.

## Metrics

```python
from ontic_lib.metrics import MetricsAccumulator, psnr

score = psnr(prediction, target, max_val=1.0)

acc = MetricsAccumulator()
for batch in loader:
    acc.add({"loss": loss.item(), "psnr": psnr(pred, batch)}, n=batch_size)
print(acc.mean())
acc.reset()
```

- `psnr(a, b, max_val=1.0)`: peak signal-to-noise ratio in dB between two
  array-likes; returns `inf` for identical inputs.
- `MetricsAccumulator`: a weighted running mean over per-step metric dicts —
  handy for averaging batch metrics of varying batch size over an epoch.
