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

- The `LoadingBinary.*` tests expect a dataset fixture at `build/gvec.bin`.
