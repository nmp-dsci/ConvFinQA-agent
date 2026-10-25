"""The student on Modal (s28), after tau-telecom-sft's `cloud/modal_app.py`.

`Student` is vLLM's OpenAI-compatible server for Qwen3-4B-Instruct-2507 in
bf16 on one L4: a 32K context, prefix caching and the hermes tool-call parser,
with no reasoning parser (the student does not think; its reasoning is the
visible ``<reasoning>`` text it writes). It serves the untrained base as
``student-qwen3-4b`` and each adapter in ``ADAPTERS`` under its own name, so the
eval loop swaps base for tuned by model name. Modal's proxy requires a token;
the runner's student provider sends it (``llm._student_model``).

`train` is ``sft_train.sft`` on one L4, shipping only that package and its
config: no ``convfinqa``, no data but the dataset on the data volume. It writes
its run folder to the runs volume, where `Student` finds the adapters and
``convfinqa-sft sync`` copies the folder home. Modal re-runs an input cut short
by a preemption with the same arguments, so the same run id, and the trainer
resumes from its newest checkpoint. No ``retries``: the job's own failures (the
data check, the mask check) would only fail again.

The weights come from tau-telecom-sft's volume, mounted read-only with the hub
offline, so nothing here can download or overwrite them.

    uv run modal deploy src/convfinqa/sft/cloud/modal_app.py                 # convfinqa-sft serve
    uv run modal run --detach src/convfinqa/sft/cloud/modal_app.py::train_job  # convfinqa-sft train
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import modal

# Self-contained on purpose: Modal imports this file inside the containers,
# where `convfinqa` (and its settings and their dependencies) is not installed.
# Deploy and run it by path (`convfinqa-sft serve|train` do).
APP = "convfinqa-sft"
BASE = "Qwen/Qwen3-4B-Instruct-2507"
BASE_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
BASE_NAME = "student-qwen3-4b"
HF_VOLUME = "tau-telecom-sft-hf"  # tau's weights, read-only here
VLLM_VOLUME = "convfinqa-sft-vllm"
DATA_VOLUME = "convfinqa-sft-data"
RUNS_VOLUME = "convfinqa-sft-runs"
GPU = "L4"
VLLM = "0.21.0"  # tau's pin
PORT = 8000
MAX_MODEL_LEN = 32768
MINUTES = 60
LORA_RANK = 16
# {served name: path under /runs} — a checkpoint per epoch, so val can pick one.
ADAPTERS: dict[str, str] = {}

TRAIN_PINS = (
    "torch==2.14.1",
    "transformers==5.20.1",
    "trl==1.16.0",
    "peft==0.21.2",
    "accelerate==1.15.0",
    "datasets==5.1.0",
    "pyyaml==6.0.3",
    "jinja2==3.1.6",
)
TRAIN_CPU = 4
TRAIN_MEMORY_GIB = 16
# Modal list prices (tau, 19 Oct): the L4, then per core and per GiB, per second
GPU_USD_PER_S = 0.000222
TRAIN_USD_PER_S = GPU_USD_PER_S + TRAIN_CPU * 0.0000131 + TRAIN_MEMORY_GIB * 0.00000222
TRAIN_HOURS = 8
REPO = "/repo"
PACKAGE = Path(__file__).resolve().parents[1]  # src/convfinqa/sft

hf_cache = modal.Volume.from_name(HF_VOLUME).read_only()
vllm_cache = modal.Volume.from_name(VLLM_VOLUME, create_if_missing=True)
data = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
runs = modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True)
OFFLINE = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(f"vllm=={VLLM}")
    .env({**OFFLINE, "VLLM_LOG_STATS_INTERVAL": "10"})
)

train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*TRAIN_PINS)
    .env({**OFFLINE, "PYTHONPATH": REPO})
    .add_local_dir(PACKAGE / "train", f"{REPO}/sft_train", ignore=["**/__pycache__"])
    .add_local_dir(PACKAGE / "configs", f"{REPO}/configs")
)

app = modal.App(APP)


def vllm_command(adapters: dict[str, str] | None = None) -> list[str]:
    """`vllm serve` with tau's flags; adapters are {served name: path under /runs}."""
    cmd = [
        "vllm", "serve", BASE,
        "--revision", BASE_REVISION,
        "--served-model-name", BASE_NAME,
        "--host", "0.0.0.0",  # noqa: S104 - inside the container; Modal's proxy is the way in
        "--port", str(PORT),
        "--dtype", "bfloat16",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.90",
        "--enable-prefix-caching",
        # report prefix-cache hits as cached tokens: the runner's per-conversation
        # token ceiling excludes cache reads (as it does for DeepSeek and Claude),
        # and without this every re-read of the context counted as new input
        "--enable-prompt-tokens-details",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
    ]  # fmt: skip
    if adapters:
        cmd += ["--enable-lora", "--max-lora-rank", str(LORA_RANK)]
        cmd += ["--max-loras", str(len(adapters)), "--lora-modules"]
        cmd += [f"{name}=/runs/{path}" for name, path in adapters.items()]
    return cmd


@app.server(
    image=image,
    gpu=GPU,
    port=PORT,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
        "/runs": runs,
    },
    scaledown_window=15 * MINUTES,
    startup_timeout=20 * MINUTES,
)
class Student:
    @modal.enter()
    def start(self) -> None:
        """Start vLLM; Modal routes the port to it once it listens."""
        self.process = subprocess.Popen(vllm_command(ADAPTERS))

    @modal.exit()
    def stop(self) -> None:
        """Stop vLLM with the container."""
        self.process.terminate()


@app.function(
    image=train_image,
    gpu=GPU,
    cpu=TRAIN_CPU,
    memory=TRAIN_MEMORY_GIB * 1024,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        f"{REPO}/data": data,
        f"{REPO}/runs": runs,
    },
    timeout=TRAIN_HOURS * 60 * MINUTES,
)
def train(
    config_file: str,
    run_id: str,
    code_sha: str,
    max_steps: int | None = None,
    longest: int | None = None,
) -> dict[str, Any]:
    """`sft_train.sft.train` on one L4, into the runs volume."""
    import time

    from sft_train.sft import train as run_train
    from sft_train.sft import write_json

    t0 = time.time()
    try:
        run_dir = run_train(
            Path(REPO) / "configs" / config_file,
            data_root=Path(REPO) / "data",
            runs_root=Path(REPO) / "runs",
            run_id=run_id,
            code_sha=code_sha,
            max_steps=max_steps,
            longest=longest,
        )
    finally:
        runs.commit()
    meta: dict[str, Any] = json.loads((run_dir / "run.json").read_text())
    meta["cost_usd"] = round((time.time() - t0) * TRAIN_USD_PER_S, 2)
    meta["cost_basis"] = (
        f"Modal list price × this attempt's wall time ({GPU}, {TRAIN_CPU} cores)"
    )
    write_json(run_dir / "run.json", meta)
    runs.commit()
    return meta


def _clean_code_sha() -> str:
    """The commit the job ships; refuses uncommitted training code (tau's rule)."""
    dirty = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            str(PACKAGE / "train"),
            str(PACKAGE / "configs"),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(
            f"commit the training code first; a Modal job ships it:\n{dirty}"
        )
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@app.local_entrypoint()
def train_job(
    config: str = "sft_v1.yaml", max_steps: int = 0, longest: int = 0
) -> None:
    """Spawn one training job; the run id is minted here so a re-run writes one folder."""
    import yaml  # type: ignore[import-untyped]

    code_sha = _clean_code_sha()
    label = yaml.safe_load((PACKAGE / "configs" / config).read_text())["label"]
    from datetime import UTC, datetime

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}_train_{label}" + (f"-step{max_steps}" if max_steps else "")
    call = train.spawn(config, run_id, code_sha, max_steps or None, longest or None)
    print(f"{run_id} at {code_sha}: call {call.object_id}")
    print(
        f"logs: uv run modal app logs {APP} · home: uv run convfinqa-sft sync --run {run_id}"
    )
    meta = call.get()
    print(
        f"{run_id}: {meta['status']}, {meta.get('steps')} steps, final loss "
        f"{meta.get('final_loss')}, about ${meta['cost_usd']}"
    )
