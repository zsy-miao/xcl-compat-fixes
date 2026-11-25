#!/usr/bin/env python3
"""Direct-DXIL coverage baseline and gate.

`xcl shaders <pack> --direct --sample 0 --json` sweeps every console program in a shader
pack through the direct translator and prints per-stage outcome counts. Run that sweep (or
hand in its saved JSON) and compare the counts against a committed baseline: a stage that
translated fewer programs than the baseline, or gained validator failures, panicked or
timed out, is a coverage regression - the class of bug a sibling project lost 323 shader
translations to unnoticed. The baseline holds counts only: no program names, no game data.

  python scripts/coverage_gate.py --xcl target/release/xcl.exe --rpf PACK [--threads N]
  python scripts/coverage_gate.py --update ...    rewrite the baseline from a fresh sweep
  python scripts/coverage_gate.py --run-json F    compare a saved sweep (no game needed)
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_BASELINE = Path(__file__).resolve().parent.parent / "tests" / "data" / "coverage-baseline.json"
WORSEN = ("validator_failed", "panicked", "timeout")


def sweep_json(stdout):
    """The report line of `xcl shaders --direct --json` output (the last JSON line)."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError("no JSON report in the sweep output")


def compare(baseline, run):
    """(regressions, notes) of a sweep against the baseline; regressions are (stage, why)."""
    bad, notes = [], []
    for stage, want in sorted(baseline["stages"].items()):
        got = run["stages"].get(stage)
        if got is None:
            bad.append((stage, "stage missing from the sweep"))
            continue
        if got["translated"] < want["translated"]:
            bad.append((stage, f"translated {got['translated']} < baseline {want['translated']}"))
        for k in WORSEN:
            if got[k] > want[k]:
                bad.append((stage, f"{k} {got[k]} > baseline {want[k]}"))
        if got["total"] != want["total"]:
            notes.append(f"{stage}: total {want['total']} -> {got['total']} (corpus changed)")
    for stage in sorted(set(run["stages"]) - set(baseline["stages"])):
        notes.append(f"{stage}: new stage in the sweep")
    return bad, notes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--xcl", type=Path, help="xcl binary (required unless --run-json)")
    ap.add_argument("--rpf", type=Path, help="shader pack (required unless --run-json)")
    ap.add_argument("--threads", type=int, help="sweep threads (xcl default: up to 8)")
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--run-json", type=Path, help="compare a saved sweep report instead of running one")
    ap.add_argument("--update", action="store_true", help="write the baseline from the sweep and exit 0")
    args = ap.parse_args(argv)
    if args.run_json:
        run = json.loads(args.run_json.read_text(encoding="utf-8"))
    else:
        if not (args.xcl and args.rpf):
            ap.error("--xcl and --rpf are required unless --run-json is given")
        cmd = [str(args.xcl), "shaders", str(args.rpf), "--direct", "--sample", "0", "--json"]
        if args.threads:
            cmd += ["--threads", str(args.threads)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        try:
            run = sweep_json(proc.stdout)
        except ValueError as e:
            print(f"{e}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}", file=sys.stderr)
            return 2
    if args.update:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps({"stages": run["stages"]}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"baseline written: {args.baseline}")
        return 0
    if not args.baseline.is_file():
        print(f"no baseline at {args.baseline} - run with --update on a known-good build", file=sys.stderr)
        return 2
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    bad, notes = compare(baseline, run)
    for n in notes:
        print(f"note: {n}")
    for stage, why in bad:
        print(f"coverage regression: {stage}: {why}")
    if bad:
        print(f"{len(bad)} regression(s) against {args.baseline}")
        return 1
    print("coverage gate: no regressions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
