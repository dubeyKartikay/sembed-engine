#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path


def run(*command):
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description="Generate embedding fixtures.")
    parser.add_argument("--output-dir", type=Path, default=Path("build"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    glove = root / "testdata/embeddings/glove/glove.6B.50d.subset.txt"
    rows = sum(1 for _ in glove.open())
    dimensions = len(glove.open().readline().split()) - 1
    glove_tool = output / "convertGloveTXT"
    run("cc", "-O2", "-std=c11", root / "convertUtils/convertGloveTXT.c",
        "-o", glove_tool)
    run(glove_tool, glove, output / "gvec.bin", output / "gvec.words.bin",
        str(rows), str(dimensions))

    word2vec_tool = output / "convertWord2vectDS"
    run("cc", "-O2", "-std=c11", root / "convertUtils/convertWord2vectDS.c",
        "-o", word2vec_tool, "-lm")
    run(word2vec_tool,
        root / "testdata/embeddings/word2vec/google-word2vec.subset.bin",
        output / "w2v.bin", output / "w2v.words.bin")


if __name__ == "__main__":
    main()
