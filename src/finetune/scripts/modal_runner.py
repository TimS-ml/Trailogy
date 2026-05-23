"""
Modal runner for Trailogy LoRA / DoRA finetuning.

Usage:
    modal run scripts/modal_runner.py::main                    # full training (1xL40S 48GB)
    modal run scripts/modal_runner.py::main --config configs/local_sweep/r32-a32-dora-nokl-local-30k-v2mix.yaml
    modal run scripts/modal_runner.py::smoke_test              # smoke test (1xL40S, 50 steps)
    modal run scripts/modal_runner.py::download_runs           # pull outputs back to local
"""

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "trailogy-finetune"
ARTIFACT_VOLUME = "trailogy-runs"      # outputs, checkpoints, eval results

DEFAULT_CONFIG = "configs/local_sweep/r32-a32-dora-nokl-local-30k-v2mix.yaml"

REPO_URL = "https://github.com/YingCeci/Trailogy.git"
REPO_BRANCH = "feature/sft-quant-datamix"

# Local finetune source tree — mounted at runtime so edits don't need
# an image rebuild (same pattern as parameter-golf's add_local_file).
FINETUNE_DIR = Path(__file__).resolve().parent.parent  # src/finetune/

# ---------------------------------------------------------------
# Image: pre-built Docker with deps, clone repo at build time.
# Local src/ and configs/ are overlaid via add_local_dir (copy=False)
# so code + config edits take effect without image rebuild.
# ---------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "timsml/trailogy:latest",
        add_python=None,  # use Python already in the image
    )
    .apt_install("git")
    .run_commands(
        f"git clone --branch {REPO_BRANCH} --single-branch {REPO_URL} /workspace/Trailogy",
    )
    .env({
        "PYTHONUNBUFFERED": "1",
    })
    .add_local_dir(
        str(FINETUNE_DIR / "src"),
        remote_path="/workspace/Trailogy/src/finetune/src",
        copy=False,
    )
    .add_local_dir(
        str(FINETUNE_DIR / "configs"),
        remote_path="/workspace/Trailogy/src/finetune/configs",
        copy=False,
    )
)

app = modal.App(APP_NAME, image=image)

run_vol = modal.Volume.from_name(ARTIFACT_VOLUME, create_if_missing=True)


# ---------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------
def _run_training(
    config: str,
    run_name: str | None,
    max_steps: int | None,
    extra_args: list[str] | None,
    env_overrides: dict[str, str] | None,
) -> None:
    """Run training inside the container."""
    workdir = "/workspace/Trailogy/src/finetune"
    os.chdir(workdir)

    # Data lives at /workspace/data/mix-50k-v2/ in the container.
    # The config expects data/mix-50k-v2/ relative to cwd. The repo's
    # data/ dir exists (has data/sample/ tracked), so we symlink the
    # specific dataset subdirs instead of replacing the whole dir.
    data_dir = Path(workdir) / "data"
    data_dir.mkdir(exist_ok=True)
    container_data = Path("/workspace/data")
    if container_data.exists():
        for child in container_data.iterdir():
            link = data_dir / child.name
            if not link.exists():
                os.symlink(str(child), str(link),
                           target_is_directory=child.is_dir())

    # Symlink outputs to artifact volume for persistence
    outputs_link = Path(workdir) / "outputs"
    if outputs_link.exists() and not outputs_link.is_symlink():
        import shutil
        shutil.rmtree(str(outputs_link))
    if not outputs_link.exists():
        os.symlink("/vol/runs/outputs", str(outputs_link), target_is_directory=True)
    Path("/vol/runs/outputs").mkdir(parents=True, exist_ok=True)

    results_link = Path(workdir) / "results"
    if results_link.exists() and not results_link.is_symlink():
        import shutil
        shutil.rmtree(str(results_link))
    if not results_link.exists():
        os.symlink("/vol/runs/results", str(results_link), target_is_directory=True)
    Path("/vol/runs/results").mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    # Auto-disable wandb if no API key is available in the environment
    if not env.get("WANDB_API_KEY"):
        env.setdefault("WANDB_MODE", "disabled")

    cmd = ["python", "-m", "src.finetune", "--config", config, "--report_to", "none"]
    if run_name:
        cmd += ["--run_name", run_name]
    if max_steps is not None:
        cmd += ["--max_steps", str(max_steps)]
    if extra_args:
        cmd += extra_args

    print(f"launching: {' '.join(cmd)}")
    print(f"config: {config}")
    print(f"cwd: {workdir}")

    # Tee logs to the artifact volume
    log_dir = Path("/vol/runs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_name = run_name or Path(config).stem
    log_path = log_dir / f"{log_name}.log"

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        for line in iter(proc.stdout.readline, b""):
            text = line.decode("utf-8", errors="replace")
            print(text, end="")
            log_f.write(text)
        proc.wait()

    if proc.returncode != 0:
        run_vol.commit()
        raise RuntimeError(f"Training failed with exit code {proc.returncode}")

    # Auto-eval
    output_dir = Path("/vol/runs/outputs") / (run_name or Path(config).stem)
    adapter_path = output_dir / "final-adapter"
    if adapter_path.exists():
        eval_cmd = [
            "python", "-m", "src.evaluate",
            "--config", config,
            "--adapter_path", str(adapter_path),
        ]
        if run_name:
            eval_cmd += ["--run_name", run_name]
        print(f"\n=== Auto-eval: {' '.join(eval_cmd)} ===")
        subprocess.run(eval_cmd, env=env, check=False)

    run_vol.commit()


@app.function(
    gpu="L40S:1",
    cpu=8.0,
    memory=64 * 1024,
    timeout=6 * 60 * 60,  # 6 hour hard limit
    volumes={"/vol/runs": run_vol},
)
def train(
    config: str = DEFAULT_CONFIG,
    run_name: str | None = None,
    max_steps: int | None = None,
    extra_args: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
):
    _run_training(config, run_name, max_steps, extra_args, extra_env)


@app.function(
    gpu="L40S:1",
    cpu=4.0,
    memory=32 * 1024,
    timeout=30 * 60,
    volumes={"/vol/runs": run_vol},
)
def smoke(
    config: str = DEFAULT_CONFIG,
):
    """Single-GPU smoke test: 50 steps, no eval."""
    _run_training(
        config=config,
        run_name="smoke-test",
        max_steps=50,
        extra_args=["--max_train_samples", "100", "--report_to", "none"],
        env_overrides={"WANDB_MODE": "disabled"},
    )


# ---------------------------------------------------------------
# Download artifacts
# ---------------------------------------------------------------
@app.local_entrypoint()
def download_runs(dest: str = "./modal-runs"):
    """Pull outputs and logs from volume to local."""
    Path(dest).mkdir(exist_ok=True, parents=True)
    subprocess.run(
        ["modal", "volume", "get", ARTIFACT_VOLUME, "/", dest, "--force"],
        check=True,
    )
    print(f"downloaded to {dest}")


# ---------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------
@app.local_entrypoint()
def main(
    config: str = DEFAULT_CONFIG,
    run_name: str = "",
    max_steps: int = 0,
):
    """
    modal run scripts/modal_runner.py::main
    modal run scripts/modal_runner.py::main --config configs/local_sweep/r32-a32-dora-nokl-local-30k-v2mix.yaml
    """
    train.remote(
        config=config,
        run_name=run_name or None,
        max_steps=max_steps or None,
    )
    download_runs()


@app.local_entrypoint()
def smoke_test(config: str = DEFAULT_CONFIG):
    """modal run scripts/modal_runner.py::smoke_test"""
    smoke.remote(config=config)
    download_runs()
