"""Materialize and audit the staff assets in a bounded remote job."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


COURSE_REPO = "https://github.com/AbDu11aHHH/atml-pa4-datasets.git"
COURSE_REVISION = "6a0a9ec9da7ef3d313ee3036cfec493886723644"
REMOVED_LEGACY_CHECKER = "scripts/check_release_STAFF_LEGACY.py"


def run(command, directory, environment):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=directory, env=environment, check=True)


def copy_and_hash(source, destination):
    destination.parent.mkdir(exist_ok=True, parents=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)
    with destination.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": destination.stat().st_size, "sha256": digest}


def verify_staff_checksums(staff, scratch, environment):
    command = [sys.executable, "scripts/verify_checksums.py"]
    result = subprocess.run(
        command, cwd=staff, env=environment, capture_output=True, text=True, check=False,
    )
    print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    if not result.returncode:
        return []

    # This exact staff revision removed the legacy checker but kept its checksum entry.
    only_known_failure = f"CHECKSUM FAILURES:\n - {REMOVED_LEGACY_CHECKER}"
    if result.stdout.strip() != only_known_failure or (staff / REMOVED_LEGACY_CHECKER).exists():
        result.check_returncode()
    checksums = json.loads((staff / "SHA256SUMS.json").read_text(encoding="utf-8"))
    del checksums[REMOVED_LEGACY_CHECKER]
    manifest = scratch / "effective-checksums.json"
    manifest.write_text(json.dumps(checksums), encoding="utf-8")
    run([*command, "--manifest", str(manifest)], staff, environment)
    return [REMOVED_LEGACY_CHECKER]


def main():
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/public-data").resolve()
    output.mkdir(exist_ok=True, parents=True)
    evidence = {"staff_revision": COURSE_REVISION, "status": "running", "artifacts": {}}
    try:
        with tempfile.TemporaryDirectory(prefix="pa4-data-") as scratch:
            root = Path(scratch)
            staff = root / "staff"
            dependencies = root / "dependencies"
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(dependencies)
            environment["HF_HOME"] = str(root / "hf")
            environment["PYTHONUNBUFFERED"] = "1"

            run(["git", "clone", COURSE_REPO, str(staff)], root, environment)
            run(["git", "checkout", "--detach", COURSE_REVISION], staff, environment)
            evidence["checksum_excluded_missing_paths"] = verify_staff_checksums(
                staff, root, environment
            )
            run([sys.executable, "scripts/validate_assets.py"], staff, environment)
            run([
                sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                "--target", str(dependencies), "-r", "requirements-public.txt",
            ], staff, environment)

            try:
                run([sys.executable, "scripts/setup_data.py", "--public"], staff, environment)
                run([sys.executable, "scripts/check_setup.py"], staff, environment)
                evidence["status"] = "passed"
            finally:
                files = list((staff / "data/public_locked").glob("*.jsonl.gz"))
                files.extend(staff.glob("data/local_source_revisions.json"))
                for path in sorted(files):
                    relative = path.relative_to(staff)
                    evidence["artifacts"][relative.as_posix()] = copy_and_hash(
                        path, output / relative
                    )
    except Exception as error:
        evidence.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        manifest = output / "manifest.json"
        manifest.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(evidence, indent=2), flush=True)


if __name__ == "__main__":
    main()
