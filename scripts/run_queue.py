"""Run the Phase 14-16 steps unattended: resumable, memory-guarded, with a progress log.

Start it detached, inside a memory-capped cgroup, so WSL stays alive whatever happens:

    nohup setsid systemd-run --user --scope -p MemoryMax=<cap> -p MemorySwapMax=512M \\
        .venv/bin/python -u scripts/run_queue.py >> outputs/overnight.log 2>&1 &

After an interruption (crash, reboot, stopped by the watchdog) run the same command again:
finished steps are skipped (step markers), and inside a step finished GAN runs and
classifiers are skipped, so at most one unit of work is repeated.

* Steps come from scripts/reproduce_results.py (STEPS), from FIRST_BATCH_STEP to the end,
  strictly one at a time, each in its own process group.
* outputs/queue/<NN>-<step>.done is written only after a step exits successfully.
* Watchdog: every --check-every seconds it reads the free memory (MemAvailable). If it stays
  below --min-available-gb for --low-checks checks in a row (whatever is using the memory),
  the step is stopped cleanly and the queue exits with code 3. Rerun it later to continue.
* outputs/overnight_memory.csv: free memory, swap used, the step's memory and GPU memory,
  every --log-every seconds.
* outputs/queue/status.txt: what is running, what is done, what comes next.
* Before the first step it backs up outputs/experiments/ and outputs/tables/ once to
  outputs/backups/before-batch/.

Reads:  scripts/reproduce_results.py (STEPS)
Writes: outputs/queue/, outputs/overnight_memory.csv, outputs/backups/before-batch/, and
        whatever each step writes
"""

import argparse
import csv
import importlib.util
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "outputs" / "queue"
MEMORY_LOG = ROOT / "outputs" / "overnight_memory.csv"
BACKUP = ROOT / "outputs" / "backups" / "before-batch"
EXIT_LOW_MEMORY = 3


def load_pipeline():
    spec = importlib.util.spec_from_file_location(
        "reproduce", ROOT / "scripts" / "reproduce_results.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def meminfo() -> dict[str, float]:
    """MemAvailable and swap used, in GB."""
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, rest = line.split(":", 1)
        values[key] = int(rest.split()[0]) / 1024**2
    return {
        "available": values["MemAvailable"],
        "swap_used": values["SwapTotal"] - values["SwapFree"],
    }


def group_rss_gb(session: int) -> float:
    """Resident memory of every process in the step's session (the step and its children)."""
    total = 0
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
            if int(fields[3]) == session:  # field 6 of /proc/<pid>/stat: session id
                total += int(fields[21])  # field 24: resident pages
        except (OSError, IndexError, ValueError):
            continue
    return total * os.sysconf("SC_PAGE_SIZE") / 1024**3


def gpu_used_mb() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return ""


def write_status(lines: list[str]) -> None:
    QUEUE.mkdir(parents=True, exist_ok=True)
    tmp = QUEUE / "status.txt.tmp"
    tmp.write_text(f"updated {datetime.now():%Y-%m-%d %H:%M:%S}\n" + "\n".join(lines) + "\n")
    os.replace(tmp, QUEUE / "status.txt")


def record_memory(step: str, session: int | None) -> dict[str, float]:
    mem = meminfo()
    new = not MEMORY_LOG.exists()
    with open(MEMORY_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(
                ["time", "step", "available_gb", "swap_used_gb", "step_rss_gb", "gpu_mb"]
            )
        writer.writerow(
            [
                f"{datetime.now():%Y-%m-%d %H:%M:%S}",
                step,
                f"{mem['available']:.2f}",
                f"{mem['swap_used']:.2f}",
                f"{group_rss_gb(session):.2f}" if session else "",
                gpu_used_mb(),
            ]
        )
    return mem


def backup_once() -> None:
    if (QUEUE / "backup.done").exists():
        return
    log(f"Backing up outputs/experiments and outputs/tables to {BACKUP.relative_to(ROOT)}")
    for name in ("experiments", "tables"):
        target = BACKUP / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(ROOT / "outputs" / name, target)
    (QUEUE / "backup.done").write_text(f"{datetime.now():%Y-%m-%d %H:%M:%S}\n")


def stop(process: subprocess.Popen) -> None:
    """SIGTERM the step's whole process group, then SIGKILL after 30 s."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from-step", help="description of the first step (default: batch start)")
    parser.add_argument("--until-step", help="description of the last step to run")
    parser.add_argument("--min-available-gb", type=float, default=1.0)
    parser.add_argument("--low-checks", type=int, default=3, help="consecutive low checks to stop")
    parser.add_argument("--check-every", type=float, default=10.0, help="seconds")
    parser.add_argument("--log-every", type=float, default=60.0, help="seconds")
    parser.add_argument("--dry-run", action="store_true", help="list the steps and their state")
    args = parser.parse_args()

    pipeline = load_pipeline()
    steps = list(enumerate(pipeline.STEPS, start=1))
    names = [d for _, (d, _, _) in steps]
    first = names.index(args.from_step or pipeline.FIRST_BATCH_STEP)
    last = names.index(args.until_step) if args.until_step else len(steps) - 1
    selected = steps[first : last + 1]
    QUEUE.mkdir(parents=True, exist_ok=True)
    marker = {n: QUEUE / f"{n:02d}-{slug(d)}.done" for n, (d, _, _) in selected}

    if args.dry_run:
        for n, (description, command, _) in selected:
            state = "done" if marker[n].exists() else "to do"
            print(f"{n:2d} [{state}] {description}: {' '.join(command)}")
        return

    stopping = {"signal": None}

    def on_signal(signum, _frame):
        stopping["signal"] = signum

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    log(f"Queue started: steps {selected[0][0]}-{selected[-1][0]}, pid {os.getpid()}")
    backup_once()
    for i, (n, (description, command, _)) in enumerate(selected):
        label = f"[{n}/{len(pipeline.STEPS)}] {description}"
        if marker[n].exists():
            log(f"{label}: already done, skipping")
            continue
        upcoming = [d for _, (d, _, _) in selected[i + 1 :]]
        write_status([f"running: {label}", f"next: {upcoming[0] if upcoming else '(last step)'}",
                      f"remaining after this: {len(upcoming)}"])  # fmt: skip
        log(f"{label}: START {' '.join(command)}")
        began = time.time()
        script, *rest = command
        process = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "scripts" / script), *rest],
            cwd=ROOT,
            start_new_session=True,
        )
        low, last_log = 0, 0.0
        while process.poll() is None:
            time.sleep(args.check_every)
            if stopping["signal"]:
                log(f"{label}: queue received signal {stopping['signal']}; stopping the step")
                stop(process)
                write_status([f"STOPPED by signal during: {label}", "rerun to resume"])
                sys.exit(1)
            now = time.time()
            if now - last_log >= args.log_every:
                mem = record_memory(description, process.pid)
                last_log = now
            else:
                mem = meminfo()
            low = low + 1 if mem["available"] < args.min_available_gb else 0
            if low >= args.low_checks:
                record_memory(description, process.pid)
                log(
                    f"{label}: WATCHDOG: free memory {mem['available']:.2f} GB < "
                    f"{args.min_available_gb} GB for {low} checks; stopping the step"
                )
                stop(process)
                write_status([f"STOPPED by the memory watchdog during: {label}",
                              "free memory, then rerun the same command to resume"])  # fmt: skip
                sys.exit(EXIT_LOW_MEMORY)
        minutes = (time.time() - began) / 60
        if process.returncode != 0:
            log(f"{label}: FAILED (exit code {process.returncode}) after {minutes:.1f} min")
            write_status([f"FAILED: {label} (exit code {process.returncode})",
                          "see outputs/overnight.log; fix, then rerun to resume"])  # fmt: skip
            sys.exit(process.returncode)
        marker[n].write_text(f"{datetime.now():%Y-%m-%d %H:%M:%S} {minutes:.1f} min\n")
        log(f"{label}: DONE in {minutes:.1f} min")
    write_status(["ALL DONE", f"finished {datetime.now():%Y-%m-%d %H:%M:%S}"])
    log("Queue finished: all steps done")


if __name__ == "__main__":
    main()
