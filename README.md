# Matrix Algorithms

An in-progress academic project focused on implementing, testing, and benchmarking
matrix algorithms from scratch in Rust. The project compares classical and
fast matrix multiplication methods and explores how their theoretical operation
counts translate into real execution time.

> **Project status:** active development. The first assignments are implemented;
> later workspace members are being developed incrementally.

## What this project demonstrates

- Implementing numerical algorithms without relying on a ready-made matrix
  multiplication routine.
- Applying divide-and-conquer techniques to matrix multiplication.
- Comparing algorithmic complexity, arithmetic operation counts, and measured
  runtime.
- Designing reusable Rust modules with unit and integration tests.
- Building reproducible benchmarks with deterministic input generation.
- Handling non-ideal matrix sizes by padding recursive algorithm inputs when
  necessary.

## Implemented algorithms

### Matrix multiplication (`z1`)

The first assignment contains implementations of:

- **Naive (Binet) multiplication** — the classical \(O(n^3)\) approach.
- **Recursive Binet multiplication** — a recursive formulation of the
  classical algorithm.
- **Strassen multiplication** — a divide-and-conquer algorithm using seven
  recursive multiplications.
- **Laderman multiplication** — a three-by-three block algorithm using 23
  recursive multiplications.

The implementations use `ndarray` for matrix storage and slicing, while the
algorithmic operations themselves are written in project code rather than
delegated to `ndarray`'s built-in `dot` operation.

### Matrix inversion (`z2`)

The second assignment builds on `z1` and currently includes recursive matrix
inversion based on block matrices and the Schur complement. Arithmetic
operations are tracked through the same instrumentation used by the
multiplication algorithms.

## Benchmarking and analysis

The project separates algorithm logic from measurement concerns:

- `OperationCounter` records additions, multiplications, and divisions.
- `EmptyCounter` provides a zero-overhead counter for timing runs.
- Criterion benchmarks compare Binet, Strassen, Laderman, and a dot-product
  reference implementation across multiple matrix sizes.
- Deterministic pseudo-random input generation makes benchmark runs
  reproducible.
- CSV utilities and analysis examples support precision checks and empirical
  complexity analysis.

Criterion reports are generated in `target/criterion` after running the
benchmark suite.

## Repository structure

```text
.
├── z1/                 # Matrix multiplication algorithms and tests
├── z2/                 # Second assignmnet (in development)
├── z3/                 # Third assignment (in development)
├── z4/                 # Fourth assignment (in development)
├── benchmarks/         # Criterion benchmark suite
├── src/macierzowe/     # Supporting Python package and analysis scaffold
├── Cargo.toml          # Rust workspace and shared dependencies
└── pyproject.toml      # Python tooling configuration
```

Each Rust assignment is an independent workspace package. Public algorithms
are exposed through `lib.rs`, while `main.rs` files provide small executable
examples. Integration tests live in the `tests/` directory of the relevant
assignment.

## Getting started

### Prerequisites

- Rust toolchain with Cargo
- Python 3.12+ and `uv` (only needed for the supporting Python analysis
  scaffold)

### Run an example

Run the binary for a selected assignment:

```bash
cargo run -p z1
cargo run -p z2
```

Run one of the `z1` analysis examples:

```bash
cargo run -p z1 --example precision_comparison
cargo run -p z1 --example operation_comparison
cargo run -p z1 --example time_complexity_calculator
```

### Test the workspace

```bash
cargo test --workspace
```

### Run benchmarks

```bash
cargo bench -p benchmarks
```

After the benchmark completes, open the generated Criterion report from
`target/criterion/report/index.html`.

## Engineering decisions

- **Rust workspace:** shared dependency versions and consistent project
  structure across assignments.
- **`ndarray`:** used as a safe, convenient storage and slicing layer rather
  than as a replacement for the algorithms being studied.
- **Generic operation counting:** algorithms accept an `OperationCounter`,
  allowing the same implementation to be used for both instrumentation and
  performance measurements.
- **Test-first verification:** results are checked against reference
  multiplication and known operation counts, including matrix sizes that are
  not powers of two or three.

## Roadmap

- Complete the remaining assignments (`z2`, `z3` and `z4`).
- Extend inversion support to additional multiplication strategies.
- Add more cross-assignment integration tests.
- Expand the benchmark suite and document measured results.
- Improve the Python analysis and visualization workflow.

## License

This repository is an educational project developed as part of university
coursework.
