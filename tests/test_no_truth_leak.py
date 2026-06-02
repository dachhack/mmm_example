"""
Guard test: the modeling pipeline must NEVER read the sealed ground truth.
Only src/draftzone_mmm/evaluate.py is permitted to reference data_sealed/.

This enforces the project's core honesty constraint. If this test fails, some pipeline
module has gained access to the answer key and the recovery results can no longer be trusted.
"""
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "draftzone_mmm"
ALLOWED = {"evaluate.py"}                 # only this module may touch the sealed truth
FORBIDDEN = re.compile(r"data_sealed|ground_truth")


def test_no_truth_leak():
    offenders = []
    if not SRC.exists():
        # Pipeline not built yet; nothing to check. Skip gracefully.
        return
    for py in SRC.glob("*.py"):
        if py.name in ALLOWED:
            continue
        text = py.read_text(encoding="utf-8", errors="ignore")
        # strip comments to avoid false positives on docstrings mentioning the rule
        code = "\n".join(
            line for line in text.splitlines() if not line.strip().startswith("#")
        )
        if FORBIDDEN.search(code):
            offenders.append(py.name)
    assert not offenders, (
        f"Forbidden reference to sealed truth in: {offenders}. "
        "Only evaluate.py may read data_sealed/ground_truth.json."
    )


if __name__ == "__main__":
    test_no_truth_leak()
    print("no-truth-leak guard: PASS")
