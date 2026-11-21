"""Run a small CUDA check and persist evidence even when the check fails."""

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


def installed_version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/preflight.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "packages": {
            name: installed_version(name)
            for name in ("torch", "transformers", "huggingface_hub", "datasets", "peft")
        },
        "free_output_bytes": shutil.disk_usage(output.parent).free,
        "source_access": {},
    }
    try:
        for url in (
            "https://github.com/AbDu11aHHH/atml-pa4-datasets.git",
            "https://github.com/mbh2006/ATML-PA_4.git",
        ):
            result = subprocess.run(
                ["git", "ls-remote", url, "HEAD"],
                capture_output=True, text=True, timeout=30, check=False,
            )
            evidence["source_access"][url] = {
                "exit_code": result.returncode,
                "head": result.stdout.split()[0] if result.stdout.strip() else None,
            }

        import torch
        from torch.nn.functional import scaled_dot_product_attention

        evidence["cuda_available"] = torch.cuda.is_available()
        evidence["torch_cuda_version"] = torch.version.cuda
        if not evidence["cuda_available"]:
            raise RuntimeError("CUDA is unavailable")

        evidence["devices"] = [
            {
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
        torch.manual_seed(6304)
        query = torch.randn((2, 4, 256, 64), device="cuda", dtype=torch.float16)
        tensor = scaled_dot_product_attention(query, query, query, is_causal=True)
        torch.cuda.synchronize()

        elapsed = []
        for _ in range(5):
            start = time.perf_counter()
            tensor = scaled_dot_product_attention(query, query, query, is_causal=True)
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - start)
        if not torch.isfinite(tensor).all().item():
            raise ArithmeticError("SDPA returned nonfinite values")

        evidence.update(
            shape=list(query.shape),
            dtype=str(query.dtype),
            sdpa_elapsed_seconds=elapsed,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            status="passed",
        )
    except Exception as error:
        evidence.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)
        print(json.dumps(evidence, indent=2), flush=True)


if __name__ == "__main__":
    main()
