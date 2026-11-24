"""What a pull request needs before it can be merged (CONTRIBUTING.md, "What a PR needs to be
merged"), checked from its description and its files – run by .github/workflows/pr-check.yml
(`pull_request_target`: this file as it is on main, never the PR's copy) and locally:

    python scripts/pr_check.py --body body.md --head <sha> --files files.tsv [--local] [--test-only]

files.tsv: one line per file of the PR – path, status, binary (true/false), blob sha. --local
asks git instead of the GitHub API (whether a commit is on main, a blob's size).
--test-only skips only the smoke check. GitHub Actions passes it after a test-only label approval.
"""
import argparse, os, re, subprocess, sys

SECTIONS = ("Tested", "Not tested", "AI use")
# small / open-weight models are not accepted for code (CONTRIBUTING.md, "AI-assisted contributions")
BANNED_MODELS = r"\b(qwen|deepseek|llama|mistral|mixtral|gemma|phi-?\d)"
# game data never goes into the repository (CONTRIBUTING.md, "Licence and provenance")
GAME_DATA = re.compile(
    r"\.(rpf|xvd|xvc|msixvc|appx|msix|appxbundle|xex|xbe|pak|bk2|bik|ttpf|dds|ktx2?|wem|bnk|"
    r"exe|dll|pdb|sys|dmp|mdmp|bin|dat|img|iso|pkg|xsp|cab)$", re.I)
GAME_FILES = re.compile(r"(^|/)(MicrosoftGame\.config|appxmanifest\.xml|resources\.pri)$", re.I)
# where test data of XCL's own may live (synthetic, small)
TEST_DATA = ("tests/data/",)
MAX_BINARY = 1 << 20
FRAMES_KEPT = 0.8   # the PR's frames at least this share of main's
COMMIT_GROWTH = 1.25  # the PR's peak commit at most this times main's


def smoke_blocks(body):
    """The `xcl-smoke v1` blocks of `body`: each a dict of its `key: value` lines. A block ends at
    a code fence or at the next block's header (both blocks pasted in one fence)."""
    out = []
    for m in re.finditer(r"xcl-smoke v1\s*\n(.*?)(?=xcl-smoke v1|```|\Z)", body, re.S):
        block = {}
        for line in m.group(1).splitlines():
            k, sep, v = line.partition(":")
            if sep:
                block[k.strip()] = v.strip()
        out.append(block)
    return out


def number(s):
    m = re.match(r"\s*(\d+)", s or "")
    return int(m.group(1)) if m else None


def runtime(path):
    return path.startswith("crates/xcl/src/") and "/tests" not in path and not path.endswith("tests.rs")


def check(body, head, files, is_main_ancestor, *, test_only=False):
    """The problems (empty: mergeable as far as this check goes). `files`: (path, status, binary,
    size) with `size` a function giving the file's bytes; `is_main_ancestor(sha)`: whether the
    commit is on main."""
    body = body.replace("\r\n", "\n")
    problems = []
    for section in SECTIONS:
        if not re.search(r"\*\*" + section + r"\*\*\s*\n\s*\S", body):
            problems.append(f"the **{section}** section is missing or empty")

    ai = re.search(r"\*\*AI use\*\*(.*?)(\n\*\*|\Z)", body, re.S)
    banned = re.findall(BANNED_MODELS, ai.group(1), re.I) if ai else []
    if banned:
        problems.append("**AI use** names a model this repository does not accept for code ("
                        + ", ".join(sorted({b.lower() for b in banned})) + "): see CONTRIBUTING.md")

    for path, status, binary, size in files:
        if status == "removed":
            continue
        in_test_data = path.startswith(TEST_DATA) and size() <= MAX_BINARY
        if GAME_FILES.search(path) and path != "tests/data/testgame/MicrosoftGame.config":
            problems.append(f"`{path}` is a game package file: game files never go into the repository")
        elif GAME_DATA.search(path) and not in_test_data:
            problems.append(f"`{path}` looks like game data or a binary: game files never go into the repository")
        elif binary and not in_test_data:
            problems.append(f"`{path}` is a binary file outside tests/data/ or over 1 MB")

    if not test_only and any(runtime(p) for p, *_ in files):
        blocks = smoke_blocks(body)
        mine = [b for b in blocks if len(b.get("commit", "")) >= 7 and head.startswith(b["commit"].split()[0])]
        clean = [b for b in mine if "uncommitted" not in b.get("commit", "")]
        base = [b for b in blocks if b not in mine and len(b.get("commit", "").split()[0] if b.get("commit") else "") >= 7]
        if not clean:
            problems.append(f"the PR changes the runtime: **Regression run** needs an `xcl-smoke` block made on this PR's "
                            f"head commit {head[:12]} with no uncommitted changes (rerun scripts/smoke.ps1 after every push)")
        main = [b for b in base if is_main_ancestor(b["commit"].split()[0])]
        if not main:
            problems.append("**Regression run** needs an `xcl-smoke` block made on a commit of main (not the PR's), "
                            "with the same title and time as the PR's")
        if clean:
            pr = clean[-1]
            for key in ("device removals", "crashes"):
                if (number(pr.get(key)) or 0) > 0:
                    problems.append(f"the PR's regression run shows {key}: {pr.get(key)}")
            if pr.get("exit", "").startswith("killed"):
                problems.append("the PR's regression run did not end by itself (killed)")
            if main:
                m = next((b for b in main if b.get("title") == pr.get("title") and b.get("seconds") == pr.get("seconds")), None)
                if m is None:
                    problems.append("the main and PR regression runs are not of the same title for the same time")
                else:
                    pf, mf = number(pr.get("frames shown")), number(m.get("frames shown"))
                    if mf and (pf is None or pf < FRAMES_KEPT * mf):
                        problems.append(f"the PR's run shows fewer frames than main's ({pf} against {mf})")
                    pc, mc = number(pr.get("peak commit")), number(m.get("peak commit"))
                    if pc and mc and pc > COMMIT_GROWTH * mc:
                        problems.append(f"the PR's run commits much more memory than main's ({pc} MB against {mc} MB)")
    return problems


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--body", required=True)
    a.add_argument("--head", required=True)
    a.add_argument("--files", required=True)
    a.add_argument("--repo", default=os.environ.get("REPO", ""))
    a.add_argument("--local", action="store_true")
    a.add_argument("--test-only", action="store_true", help="skip the retail smoke check for an approved test-only diff")
    o = a.parse_args()
    body = open(o.body, encoding="utf-8").read()
    def blob_size(sha):
        if o.local:
            r = subprocess.run(["git", "cat-file", "-s", sha], capture_output=True, text=True)
        else:
            r = subprocess.run(["gh", "api", f"repos/{o.repo}/git/blobs/{sha}", "--jq", ".size"], capture_output=True, text=True)
        return int(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip().isdigit() else 1 << 62

    files = []
    for line in open(o.files, encoding="utf-8"):
        parts = line.rstrip("\r\n").split("\t")
        if len(parts) >= 4:
            sha = parts[3]
            files.append((parts[0], parts[1], parts[2] == "true", lambda sha=sha: blob_size(sha)))

    def is_main_ancestor(sha):
        if o.local:
            return subprocess.run(["git", "merge-base", "--is-ancestor", sha, "origin/main"], capture_output=True).returncode == 0
        r = subprocess.run(["gh", "api", f"repos/{o.repo}/compare/{sha}...main", "--jq", ".status"], capture_output=True, text=True)
        return r.returncode == 0 and r.stdout.strip() in ("ahead", "identical")

    problems = check(body, o.head, files, is_main_ancestor, test_only=o.test_only)
    if problems:
        print("This PR cannot be merged yet (see CONTRIBUTING.md, 'What a PR needs to be merged'):")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    print("description ok")


if __name__ == "__main__":
    main()
