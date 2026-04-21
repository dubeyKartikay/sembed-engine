# sembed-engine

## Requirements

- CMake
- A C++17-capable compiler
- A CMake generator such as `make` or `ninja`
- [Allure](https://allurereport.org) (for HTML test reports)

## Build

```sh
cmake -S . -B build
cmake --build build
```

Produces the main executable at `build/sembed`.
The test build also generates deterministic embedding fixtures at `build/gvec.bin`
and `build/w2v.bin` from the checked-in subsets under `testdata/embeddings/`.

## Test

Run the suite with ctest:

```sh
ctest --test-dir build --output-on-failure
```

Or generate an HTML report:

```sh
cmake --build build --target test_html_report
```

To also open the report in a browser after generation:

```sh
cmake --build build --target test_html_report_open
```

The GoogleTest runner is also available directly at `build/Test`.

## Notes

- `build/gvec.bin` is generated automatically from a reduced `glove.6B.50d` subset.
- `build/w2v.bin` is generated automatically from a reduced GoogleNews word2vec subset.
- Dataset fixtures use a binary layout of `int64_t record_id` followed by `storedDimentions - 1` floats per record.
