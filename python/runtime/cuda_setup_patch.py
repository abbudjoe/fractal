from __future__ import annotations

import argparse
from pathlib import Path


def sm_from_arch(arch: str) -> str:
    major, minor = arch.strip().split(".", 1)
    return f"{major}{minor}"


def patch_cuda_setup_text(setup_text: str, arch: str) -> str:
    lines = setup_text.splitlines()
    start = None
    end = None

    for index, line in enumerate(lines):
        if 'cc_flag.append("-gencode")' in line and start is None:
            start = index
        if start is not None and line.lstrip().startswith("# HACK:"):
            end = index
            break

    if start is None or end is None or end <= start:
        raise ValueError("failed to locate CUDA arch block in setup.py")

    indent = lines[start][: len(lines[start]) - len(lines[start].lstrip())]
    sm = sm_from_arch(arch)
    replacement = [
        f'{indent}cc_flag.append("-gencode")',
        f'{indent}cc_flag.append("arch=compute_{sm},code=sm_{sm}")',
        "",
    ]
    patched = lines[:start] + replacement + lines[end:]
    return "\n".join(patched) + "\n"


def patch_cuda_setup_file(setup_path: Path, arch: str) -> None:
    patched = patch_cuda_setup_text(setup_path.read_text(encoding="utf-8"), arch)
    setup_path.write_text(patched, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch a CUDA setup.py to build only one compute capability."
    )
    parser.add_argument("setup_path", type=Path)
    parser.add_argument("--arch", required=True, help="CUDA arch like 8.9")
    args = parser.parse_args()

    patch_cuda_setup_file(args.setup_path, args.arch)
    print(f"patched {args.setup_path} to build only sm_{sm_from_arch(args.arch)}")


if __name__ == "__main__":
    main()
