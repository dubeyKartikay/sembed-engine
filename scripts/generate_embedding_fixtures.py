#!/usr/bin/env python3

import argparse
import shutil
import subprocess
from pathlib import Path


def run(cmd, *, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True)


def compile_converter(source: Path, output: Path, extra_args=None):
    extra_args = extra_args or []
    output.parent.mkdir(parents=True, exist_ok=True)
    run(["cc", "-O2", "-std=c11", str(source), "-o", str(output), *extra_args])


def glove_dimensions(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline().strip().split()
    if len(first) < 2:
        raise RuntimeError(f"unexpected glove subset format: {path}")
    return len(first) - 1


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def generate_glove_fixture(repo_root: Path, output_dir: Path, tools_dir: Path):
    source = repo_root / "testdata/embeddings/glove/glove.6B.50d.subset.txt"
    fixture = output_dir / "gvec.bin"
    words = output_dir / "gvec.words.bin"
    converter = tools_dir / "convertGloveTXT"

    compile_converter(repo_root / "convertUtils/convertGloveTXT.c", converter)
    run(
        [
            str(converter),
            str(source),
            str(fixture),
            str(words),
            str(line_count(source)),
            str(glove_dimensions(source)),
        ]
    )


def generate_word2vec_fixture(repo_root: Path, output_dir: Path, tools_dir: Path):
    source = repo_root / "testdata/embeddings/word2vec/google-word2vec.subset.bin"
    if not source.exists():
        return

    fixture = output_dir / "w2v.bin"
    words = output_dir / "w2v.words.bin"
    converter = tools_dir / "convertWord2vectDS"

    compile_converter(
        repo_root / "convertUtils/convertWord2vectDS.c",
        converter,
        extra_args=["-lm"],
    )
    run([str(converter), str(source), str(fixture), str(words)])


def main():
    parser = argparse.ArgumentParser(
        description="Generate deterministic embedding fixtures used by tests."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build"),
        help="Directory where converted fixtures should be written.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (repo_root / args.output_dir).resolve()
    tools_dir = output_dir / "fixture-tools"

    output_dir.mkdir(parents=True, exist_ok=True)
    if tools_dir.exists():
        shutil.rmtree(tools_dir)
    tools_dir.mkdir(parents=True, exist_ok=True)

    generate_glove_fixture(repo_root, output_dir, tools_dir)
    generate_word2vec_fixture(repo_root, output_dir, tools_dir)


if __name__ == "__main__":
    main()
