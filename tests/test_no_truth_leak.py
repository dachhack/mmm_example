"""
Guard test: the modeling pipeline must NEVER READ the sealed ground truth.

  - evaluate.py  is the ONLY module permitted to READ data_sealed/ground_truth.json
                 (it grades recovery against the answer key).
  - datagen.py   GENERATES and seals the truth, so it must reference the path — but it is
                 verified to be WRITE-ONLY here (it never opens the sealed file for reading).
  - every other module must not reference the sealed truth at all.

This enforces the project's core honesty constraint. If it fails, some pipeline module has
gained read access to the answer key and the recovery results can no longer be trusted.
"""
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "draftzone_mmm"

READERS = {"evaluate.py"}        # may read the sealed truth
WRITERS = {"datagen.py"}         # may create/seal it, but must not read it
FORBIDDEN = re.compile(r"data_sealed|ground_truth")
# Patterns that constitute READING a file.
READ_PATTERNS = re.compile(
    r"json\.load|read_text|np\.load|pd\.read|open\([^)]*['\"]r['\"]|open\([^,]*\)"
)


def _strip_comments(text):
    return "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))


def test_no_truth_leak():
    if not SRC.exists():
        return  # pipeline not built yet; nothing to check
    offenders = []
    for py in sorted(SRC.glob("*.py")):
        code = _strip_comments(py.read_text(encoding="utf-8", errors="ignore"))
        if py.name in READERS:
            continue
        if not FORBIDDEN.search(code):
            continue
        if py.name in WRITERS:
            # the generator may reference the path but only to WRITE it
            for line in code.splitlines():
                if FORBIDDEN.search(line) and READ_PATTERNS.search(line):
                    offenders.append(f"{py.name} (reads sealed truth: {line.strip()!r})")
        else:
            offenders.append(py.name)
    assert not offenders, (
        f"Forbidden access to sealed truth in: {offenders}. "
        "Only evaluate.py may read data_sealed/ground_truth.json; datagen.py may only write it."
    )


def test_datagen_opens_sealed_write_only():
    """Belt-and-suspenders: every open() of the sealed file in datagen is write mode."""
    dg = (SRC / "datagen.py")
    if not dg.exists():
        return
    for line in _strip_comments(dg.read_text()).splitlines():
        if "ground_truth.json" in line and "open(" in line:
            assert '"w"' in line or "'w'" in line, f"datagen must open sealed truth write-only: {line!r}"


if __name__ == "__main__":
    test_no_truth_leak()
    test_datagen_opens_sealed_write_only()
    print("no-truth-leak guard: PASS")
