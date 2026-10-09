"""Single-file VIP beam experiment: optimizer, Llama agent, and GPT agent.

Python 3.10 or later. No separate beam_model.py or requirements file is needed.

Install Python dependencies in your terminal:
    python3 -m pip install scipy openai matplotlib

Install and open Ollama, then download the required model:
    ollama pull llama3.2:1b

Run all three methods (the API key prompt is hidden and the key is not saved):
    python3 avolovic_beam_ai.py --username YOUR_PURDUE_USERNAME --ask-key

Run just the numerical optimizer:
    python3 avolovic_beam_ai.py --method optimizer

Both agents start at 50 x 80 mm and each make 11 proposals in each of two runs.
Results are written to a new timestamped results/ folder beside this script,
regardless of the terminal's current working directory.
The supplied evaluator is embedded below without changing its implementation.
"""

import argparse
import csv
import getpass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time
from datetime import datetime, timezone
import urllib.request

# BEGIN ORIGINAL BEAM EVALUATOR
"""Beam-theory evaluator for the VIP individual design project.

Inputs are beam width and height in millimeters. The model represents a
simply supported rectangular steel beam under a central static point load,
with small deflections and linear elastic behavior. It excludes local
load/support contact effects, self-weight, and impact behavior.
"""

import math
from numbers import Real


# Consistent N, mm, MPa, kg system (1 MPa = 1 N/mm**2).
SPAN_MM = 1000.0
LOAD_N = 10_000.0
ELASTIC_MODULUS_MPA = 200_000.0
DENSITY_KG_PER_MM3 = 7.85e-6

WIDTH_BOUNDS_MM = (20.0, 100.0)
HEIGHT_BOUNDS_MM = (40.0, 120.0)
MAX_DEFLECTION_MM = 1.0
MAX_VON_MISES_STRESS_MPA = 120.0


def evaluate_beam(width_mm: float, height_mm: float) -> dict[str, float | bool]:
    """Evaluate one design and return its mass, deflection, stress, and feasibility.

    Args:
        width_mm: Rectangular cross-section width, in mm (20 to 100).
        height_mm: Rectangular cross-section height, in mm (40 to 120).

    Returns:
        A dictionary with ``mass_kg``, ``max_deflection_mm``,
        ``max_von_mises_stress_MPa``, and ``feasible``. The stress is the
        bending-surface von Mises stress at midspan. At this surface the
        shear traction from elementary beam theory is zero, so the von
        Mises stress equals the magnitude of the bending normal stress.

    Raises:
        ValueError: If either input is not a finite real number or is outside
            the specified design bounds. Validate agent proposals before
            passing them to this function.
    """
    for name, value, bounds in (
        ("width_mm", width_mm, WIDTH_BOUNDS_MM),
        ("height_mm", height_mm, HEIGHT_BOUNDS_MM),
    ):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite real number in mm")
        if not bounds[0] <= value <= bounds[1]:
            raise ValueError(f"{name} must be between {bounds[0]} and {bounds[1]} mm")

    width_mm = float(width_mm)
    height_mm = float(height_mm)

    second_moment_mm4 = width_mm * height_mm**3 / 12.0
    max_moment_n_mm = LOAD_N * SPAN_MM / 4.0

    mass_kg = DENSITY_KG_PER_MM3 * width_mm * height_mm * SPAN_MM
    max_deflection_mm = LOAD_N * SPAN_MM**3 / (
        48.0 * ELASTIC_MODULUS_MPA * second_moment_mm4
    )
    max_von_mises_stress_mpa = max_moment_n_mm * (height_mm / 2.0) / second_moment_mm4

    return {
        "mass_kg": mass_kg,
        "max_deflection_mm": max_deflection_mm,
        "max_von_mises_stress_MPa": max_von_mises_stress_mpa,
        "feasible": (
            max_deflection_mm <= MAX_DEFLECTION_MM
            and max_von_mises_stress_mpa <= MAX_VON_MISES_STRESS_MPA
        ),
    }
# END ORIGINAL BEAM EVALUATOR

START = (50.0, 80.0)
RUNS = 2
PROPOSALS = 11
LLAMA_MODEL = "llama3.2:1b"
GPT_MODEL = "gpt-4.1-mini-2025-04-14"
TEMPERATURE = 0.5
TOP_P = 0.9
MAX_TOKENS = 200

SYSTEM_PROMPT = f"""You are a beam design agent. Minimize mass_kg while satisfying
max_deflection_mm <= {MAX_DEFLECTION_MM} and
max_von_mises_stress_MPa <= {MAX_VON_MISES_STRESS_MPA}.
Width must be between {WIDTH_BOUNDS_MM[0]} and {WIDTH_BOUNDS_MM[1]} mm inclusive.
Height must be between {HEIGHT_BOUNDS_MM[0]} and {HEIGHT_BOUNDS_MM[1]} mm inclusive.
Use the supplied evaluation history to propose one next design. An external
black-box evaluator calculates the results; you only choose the dimensions.
Reply with exactly one JSON object having exactly two keys, width_mm and
height_mm, with finite JSON numbers as values. Do not use strings for numbers,
Markdown fences, comments, additional keys, explanations, or multiple designs.
Invalid replies consume a proposal and are not evaluated or repaired.
Repeated valid designs are evaluated and consume a proposal too.
"""

FIELDS = [
    "method", "model", "run", "iteration", "origin", "response_status",
    "width_mm", "height_mm", "mass_kg", "max_deflection_mm",
    "max_von_mises_stress_MPa", "feasible", "deflection_margin_mm",
    "stress_margin_MPa", "evaluated", "beam_evaluations", "repeated_design",
    "best_feasible_mass_kg", "elapsed_seconds", "raw_response", "error",
    "returned_model", "response_id",
]


def strict_json(text):
    def object_from_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError("Non-finite JSON number: " + value)

    return json.loads(text, object_pairs_hook=object_from_pairs,
                      parse_constant=reject_constant)


def validate_proposal(text):
    """No regex extraction, fence removal, unit conversion, clipping, or retry."""
    data = strict_json(text)
    if not isinstance(data, dict) or set(data) != {"width_mm", "height_mm"}:
        raise ValueError("Expected exactly width_mm and height_mm in a JSON object")
    for key, bounds in (("width_mm", WIDTH_BOUNDS_MM),
                        ("height_mm", HEIGHT_BOUNDS_MM)):
        value = data[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(key + " must be a finite number (not a string or boolean)")
        if not bounds[0] <= value <= bounds[1]:
            raise ValueError(f"{key} must be in {bounds} mm; received {value}")
    return data["width_mm"], data["height_mm"]


def messages_for(history, iteration):
    # Each run supplies only its own history. No optimizer results are included.
    keys = ("iteration", "width_mm", "height_mm", "mass_kg",
            "max_deflection_mm", "max_von_mises_stress_MPa", "feasible",
            "response_status", "raw_response", "error")
    records = [{k: row[k] for k in keys if row.get(k) not in (None, "")}
               for row in history]
    prompt = (f"Proposal {iteration} of {PROPOSALS}. "
              "History begins with the common starting design at iteration 0.\n"
              + json.dumps(records, allow_nan=False, separators=(",", ":"))
              + "\nPropose the next width_mm and height_mm.")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}]


def safe_error(exc):
    # SDK error strings can contain credential fragments; log safe metadata only.
    parts = [type(exc).__name__]
    for attr in ("status_code", "code", "type"):
        value = getattr(exc, attr, None)
        if value is not None and re.fullmatch(r"[A-Za-z0-9_.-]+", str(value)):
            parts.append(f"{attr}={value}")
    if isinstance(exc, (ImportError, RuntimeError)):
        parts.append(str(exc))
    return "; ".join(parts)


def ollama_json(base, endpoint, payload=None, timeout=60):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(base.rstrip("/") + endpoint, data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


class OllamaBackend:
    model = LLAMA_MODEL

    def __init__(self, base, timeout):
        self.base, self.timeout = base, timeout
        tags = ollama_json(base, "/api/tags", timeout=5)
        match = next((m for m in tags.get("models", [])
                      if m.get("name") == self.model or m.get("model") == self.model), None)
        if match is None:
            raise RuntimeError("Run: ollama pull llama3.2:1b")
        self.metadata = {"model": self.model, "installed_model": match,
                         "server": ollama_json(base, "/api/version", timeout=5),
                         "num_ctx": 8192, "seed": "not set", "retries": 0}

    def __call__(self, messages):
        data = ollama_json(self.base, "/api/chat", {
            "model": self.model, "messages": messages, "stream": False,
            "options": {"temperature": TEMPERATURE, "top_p": TOP_P,
                        "num_predict": MAX_TOKENS, "num_ctx": 8192},
        }, timeout=self.timeout)
        return {"text": data.get("message", {}).get("content", ""),
                "complete": bool(data.get("done")) and data.get("done_reason") != "length",
                "returned_model": data.get("model", ""), "response_id": "",
                "details": {k: v for k, v in data.items() if k != "message"}}

    def close(self):
        pass


class OpenAIBackend:
    model = GPT_MODEL

    def __init__(self, timeout):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY or use --ask-key in a terminal")
        from openai import OpenAI
        self.client = OpenAI(max_retries=0, timeout=timeout)
        self.metadata = {"model": self.model, "api": "Responses", "retries": 0,
                         "seed": "not set", "store": False}

    def __call__(self, messages):
        response = self.client.responses.create(
            model=self.model, input=messages, temperature=TEMPERATURE,
            top_p=TOP_P, max_output_tokens=MAX_TOKENS, store=False,
        )
        return {"text": response.output_text or "",
                "complete": response.status == "completed",
                "returned_model": response.model, "response_id": response.id,
                "details": {"status": response.status,
                            "usage": response.usage.model_dump() if response.usage else None,
                            "incomplete_details": (response.incomplete_details.model_dump()
                                                   if response.incomplete_details else None)}}

    def close(self):
        self.client.close()


class RunLog:
    def __init__(self, method, model, run, sink=None):
        self.method, self.model, self.run = method, model, run
        self.sink = sink or (lambda row: None)
        self.rows, self.seen = [], set()
        self.evaluations, self.best = 0, None
        self.started = time.perf_counter()

    def record(self, iteration, origin, status, dimensions=None, **extra):
        row = {key: None for key in FIELDS}
        row.update(method=self.method, model=self.model, run=self.run,
                   iteration=iteration, origin=origin, response_status=status,
                   evaluated=False, repeated_design=False, **extra)
        if dimensions is not None:
            width, height = dimensions
            row.update(width_mm=width, height_mm=height,
                       repeated_design=(width, height) in self.seen)
            outputs = evaluate_beam(width, height)
            self.evaluations += 1
            self.seen.add((width, height))
            row.update(outputs, evaluated=True,
                       deflection_margin_mm=MAX_DEFLECTION_MM - outputs["max_deflection_mm"],
                       stress_margin_MPa=MAX_VON_MISES_STRESS_MPA - outputs["max_von_mises_stress_MPa"])
            # Strict evaluator feasibility is authoritative for best-design selection.
            if outputs["feasible"] and (self.best is None or outputs["mass_kg"] < self.best["mass_kg"]):
                self.best = row
        row.update(beam_evaluations=self.evaluations,
                   best_feasible_mass_kg=self.best["mass_kg"] if self.best else None,
                   elapsed_seconds=time.perf_counter() - self.started)
        self.rows.append(row)
        self.sink(row)
        return row

    def summary(self, status, **extra):
        best_keys = ("width_mm", "height_mm", "mass_kg", "max_deflection_mm",
                     "max_von_mises_stress_MPa", "deflection_margin_mm", "stress_margin_MPa")
        return {"method": self.method, "model": self.model, "run": self.run,
                "status": status, "beam_evaluations": self.evaluations,
                "proposal_attempts": sum(r["origin"] == "proposal" for r in self.rows),
                "invalid_replies": sum(r["response_status"] == "invalid_reply" for r in self.rows),
                "service_errors": sum(r["response_status"] == "service_error" for r in self.rows),
                "repeated_proposals": sum(r["repeated_design"] and r["origin"] == "proposal" for r in self.rows),
                "elapsed_seconds": time.perf_counter() - self.started,
                "best_feasible": {k: self.best[k] for k in best_keys} if self.best else None,
                **extra}


def run_agent(backend, run, sink=None, audit=None, method="agent"):
    """The same agent loop is used for both models; it never calls SciPy."""
    log = RunLog(method, backend.model, run, sink)
    log.record(0, "starting_design", "starting_design", START)
    audit = audit or (lambda event: None)
    for iteration in range(1, PROPOSALS + 1):
        messages = messages_for(log.rows, iteration)
        event = {"method": method, "model": backend.model, "run": run,
                 "iteration": iteration, "messages": messages}
        print(f"{backend.model}: run {run}/{RUNS}, proposal {iteration}/{PROPOSALS}", flush=True)
        try:
            reply = backend(messages)
        except Exception as exc:
            error = safe_error(exc)
            log.record(iteration, "proposal", "service_error", error=error)
            event["error"] = error
        else:
            event["reply"] = reply
            extra = {"raw_response": reply["text"],
                     "returned_model": reply.get("returned_model", ""),
                     "response_id": reply.get("response_id", "")}
            try:
                if not reply.get("complete", True):
                    raise ValueError("Model response was incomplete or truncated")
                dimensions = validate_proposal(reply["text"])
            except (ValueError, TypeError, OverflowError, RecursionError) as exc:
                # Retain readable dimensions even for an invalid, out-of-bounds reply.
                # These values are logged only; they are never repaired or evaluated.
                try:
                    attempted = strict_json(reply["text"])
                    if isinstance(attempted, dict):
                        for key in ("width_mm", "height_mm"):
                            value = attempted.get(key)
                            if type(value) in (int, float) and math.isfinite(value):
                                extra[key] = value
                except (ValueError, TypeError, OverflowError, RecursionError):
                    pass
                log.record(iteration, "proposal", "invalid_reply", error=str(exc), **extra)
            else:
                log.record(iteration, "proposal", "valid", dimensions, **extra)
        audit(event)
        latest = log.rows[-1]
        print(f"  {latest['response_status']}; best feasible mass = "
              f"{latest['best_feasible_mass_kg']:.6f} kg", flush=True)
    status = "completed_with_service_errors" if any(r["response_status"] == "service_error" for r in log.rows) else "completed"
    return log, log.summary(status)


def run_optimizer(sink=None):
    from scipy.optimize import minimize
    log = RunLog("optimizer", "SLSQP", 1, sink)
    log.record(0, "starting_design", "starting_design", START)

    def evaluate(x, origin):
        # Log EVERY black-box call, including constraint and derivative probes.
        return log.record(len(log.rows), origin, "evaluated", tuple(x))

    result = minimize(
        fun=lambda x: evaluate(x, "objective")["mass_kg"], x0=START,
        method="SLSQP", bounds=[WIDTH_BOUNDS_MM, HEIGHT_BOUNDS_MM],
        constraints=[
            {"type": "ineq", "fun": lambda x: 1 - evaluate(x, "deflection_constraint")["max_deflection_mm"] / MAX_DEFLECTION_MM},
            {"type": "ineq", "fun": lambda x: 1 - evaluate(x, "stress_constraint")["max_von_mises_stress_MPa"] / MAX_VON_MISES_STRESS_MPA},
        ], options={"maxiter": 200, "ftol": 1e-9},
    )
    final = evaluate(result.x, "returned_design")
    tolerant = (final["deflection_margin_mm"] / MAX_DEFLECTION_MM >= -1e-6
                and final["stress_margin_MPa"] / MAX_VON_MISES_STRESS_MPA >= -1e-6)
    summary = log.summary("completed", solver_success=bool(result.success),
                          solver_message=str(result.message), solver_iterations=int(result.nit),
                          scipy_objective_calls=int(result.nfev),
                          returned_design={k: final[k] for k in ("width_mm", "height_mm", "mass_kg", "feasible")},
                          returned_feasible_with_numerical_tolerance=tolerant)
    return log, summary


def save_plots(logs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    agents = [log for log in logs if log.method != "optimizer"]
    if not agents:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for log in agents:
        y = [r["best_feasible_mass_kg"] for r in log.rows]
        label = f"{log.model}, run {log.run}"
        axes[0].step([r["iteration"] for r in log.rows], y, where="post", label=label)
        evaluated = [r for r in log.rows if r["evaluated"]]
        axes[1].step([r["beam_evaluations"] for r in evaluated],
                     [r["best_feasible_mass_kg"] for r in evaluated], where="post", label=label)
    for ax in axes:
        for log in logs:
            if log.method == "optimizer" and log.best:
                ax.axhline(log.best["mass_kg"], color="black", ls="--", label="SLSQP best feasible")
        ax.set_ylabel("Best feasible mass (kg)")
        ax.grid(alpha=0.25)
    axes[0].set_xlabel("Proposal number (0 = starting design)")
    axes[1].set_xlabel("Beam evaluations (includes starting design)")
    axes[0].set_title("All attempts, including invalid replies")
    axes[1].set_title("Evaluated designs only")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("all", "optimizer", "ollama", "openai"), default="all")
    parser.add_argument("--username", default="username", help="Your Purdue username for output filenames")
    parser.add_argument("--ask-key", action="store_true", help="Prompt privately for an OpenAI API key")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path,
                        help="Optional output folder; default is results/<timestamp> beside this script")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.username):
        parser.error("username must contain only letters, numbers, underscores, or hyphens")
    if args.ask_key and args.method in ("all", "openai") and not os.environ.get("OPENAI_API_KEY"):
        if not sys.stdin.isatty():
            parser.error("--ask-key needs an interactive terminal")
        os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API key (hidden, not saved): ").strip()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    script_dir = Path(__file__).resolve().parent
    out = args.output_dir.expanduser() if args.output_dir else script_dir / "results" / timestamp
    try:
        out.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        parser.error(f"Cannot create output folder '{out}': {exc.strerror}. "
                     "Use --output-dir with a new folder in a writable location.")
    prefix = args.username + "avolovic_beam_ai"
    metadata = {
        "started_utc": timestamp, "python": sys.version, "platform": platform.platform(),
        "packages": {}, "starting_design_mm": START, "runs_per_model": RUNS,
        "proposals_per_run": PROPOSALS, "temperature": TEMPERATURE, "top_p": TOP_P,
        "max_output_tokens": MAX_TOKENS, "timeout_seconds": args.timeout,
        "prompt": SYSTEM_PROMPT, "response_format": "plain text with strict local JSON validation",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "evaluator_packaging": "Original evaluator embedded without implementation changes",
        "selected_method": args.method, "backends": {}, "setup_errors": {},
        "timing": "Per-run wall clock, includes evaluation, API wait and logging; excludes setup and plots. No warm-up.",
        "feasibility": "Strict evaluator flag selects best designs; optimizer tolerance is reported separately.",
    }
    for package in ("scipy", "numpy", "openai", "matplotlib"):
        try:
            metadata["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            metadata["packages"][package] = "not installed"
    logs, summaries = [], []

    def checkpoint():
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (out / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    checkpoint()
    with (out / (prefix + ".csv")).open("w", newline="", encoding="utf-8") as csv_file, \
         (out / "agent_transcripts.jsonl").open("w", encoding="utf-8") as transcript:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()

        def sink(row):
            writer.writerow(row)
            csv_file.flush()

        def audit(event):
            transcript.write(json.dumps(event, allow_nan=False) + "\n")
            transcript.flush()

        try:
            if args.method in ("all", "optimizer"):
                log, summary = run_optimizer(sink)
                logs.append(log)
                summaries.append(summary)
                checkpoint()
            for name, factory in (
                ("ollama", lambda: OllamaBackend(args.ollama_url, args.timeout)),
                ("openai", lambda: OpenAIBackend(args.timeout)),
            ):
                if args.method not in ("all", name):
                    continue
                try:
                    backend = factory()
                except Exception as exc:
                    metadata["setup_errors"][name] = safe_error(exc)
                    print(f"{name} unavailable: {safe_error(exc)}", flush=True)
                    checkpoint()
                    continue
                metadata["backends"][name] = backend.metadata
                checkpoint()
                try:
                    for run in range(1, RUNS + 1):
                        log, summary = run_agent(backend, run, sink, audit, method=name)
                        logs.append(log)
                        summaries.append(summary)
                        checkpoint()
                finally:
                    backend.close()
        except KeyboardInterrupt:
            metadata["interrupted"] = True
            print("Interrupted. Completed rows and transcripts have been saved.")
        finally:
            checkpoint()
    save_plots(logs, out / "search_history.png")
    for summary in summaries:
        best = summary["best_feasible"]
        print(f"{summary['model']} run {summary['run']}: {summary['status']}; "
              f"best feasible mass = {best['mass_kg']:.6f} kg; "
              f"evaluations = {summary['beam_evaluations']}")
    print(f"Saved results to: {out.resolve()}")
    if metadata["setup_errors"]:
        print("Incomplete experiment: fix the listed model setup errors and run again.")
        return 1
    if metadata.get("interrupted") or any(s["service_errors"] for s in summaries):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
