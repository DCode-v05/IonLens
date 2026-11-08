# IonLens — Virtual Plasma Laboratory

![Python](https://img.shields.io/badge/Python_3.12-3776AB?style=flat&logo=python&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white) ![SymPy](https://img.shields.io/badge/SymPy-3B5526?style=flat&logo=sympy&logoColor=white) ![FEniCSx](https://img.shields.io/badge/FEniCSx-1F4E79?style=flat&logo=python&logoColor=white) ![JAX](https://img.shields.io/badge/JAX-A8B9CC?style=flat&logo=google&logoColor=black) ![uv](https://img.shields.io/badge/uv-DE5FE9?style=flat&logo=uv&logoColor=white) ![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=flat&logo=pytest&logoColor=white)

## Overview

Ion energy flux is the energy that ions carry into a surface facing a plasma. It decides how fast a wafer etches or a coating grows, but it cannot be measured directly. A probe that can see it disturbs it, and a probe that leaves it alone sees only part of the picture.

IonLens is a physics-constrained inverse diagnostics framework for plasma sheaths. It asks one question: given only non-perturbing optical signals (optical emission spectra, laser-induced fluorescence lineshapes, Thomson scattering spectra and interferometric phase), can the ion energy flux be reconstructed in space and time, with honest uncertainty?

It treats this as a regularised, Bayesian inverse problem. A forward chain built from sheath and kinetic physics turns a known plasma state into synthetic measurements. The inverse solver recovers the state from those measurements alone, without seeing the truth, and a validation engine compares the result with the hidden truth. Without a laboratory, that closed loop is the evidence for the accuracy claims.

Built by Team Astrophage for Techathon 2.0 (Naari Shakti), track *Plasma Technologies for Industry & Society*, hosted by AIC RRCAT π-Hub Foundation, DAE. It is not a plasma simulator or a digital twin. Everything runs on synthetic data.

## Key Features

- **Four-stage forward chain** `F = F₄∘F₃∘F₂∘F₁`: sheath physics, emission and scattering response, optical transport, and detector noise.
- **Sealed-truth barrier.** The inverse layer depends on no forward model, and a test checks the import graph so it stays that way.
- **Physics hierarchy:** L0 analytic sheath models (the verification anchor), L1 fluid solver on FEniCSx, and an L2 particle-in-cell level that targets JAX on the GPU.
- **Real atomic data loaders** for LXCat cross sections, NIST ASD transition data and OpenADAS, plus Boltzmann / EEDF integration.
- **Parameter registry with a literal-lint rule.** Every number traces to a published source, a standard database or a declared design parameter. A guess is tagged `ASSUMED`.
- **Manifest engine:** one YAML file per experiment, run with `ionlens run`, re-checked bit-identical with `ionlens reproduce`, and diffed with `ionlens compare`.
- **Provenance embedded in every artifact** written to HDF5, Zarr or Parquet.
- **Verification and validation kept apart**, with separate suites, reports and gates.

## How It Works

### The closed loop

```
TRUE PLASMA STATE x                (known only because we synthesised it)
   │
   ▼  F₁  sheath / fluid / kinetic (PIC) physics
   ▼  F₂  collisional-radiative emission, LIF, Thomson, refractive index
   ▼  F₃  optical transport: geometry, apertures, aberration, stray light
   ▼  F₄  detector: QE, shot / read / dark noise, ADC, dead pixels
   │
MEASUREMENTS y = F(x) + ε          ◄── the truth x is now hidden
   │
   ▼  physics-constrained inverse solver (regularised / Bayesian)
   │
RECOVERED STATE x̂, posterior p(x|y)  →  ION ENERGY FLUX Γ_E(x̂) ± σ
   │
   ▼  validation engine: compare with hidden truth, decompose the error budget
```

### Specification first

The plan was written before the code. Fifteen numbered documents in [`docs/`](docs/) cover the charter, requirements, system design, physics models, measurement models, inverse problem, error budget, V&V, architecture, data provenance, compute plan, roadmap, commercialisation, reproducibility and risk register. The code is written against them section by section. When the code shows a document is wrong, the fix goes into an architecture decision record in [`docs/adr/`](docs/adr/) instead of being applied silently.

### Hard constraints

1. **No hardware, no laboratory, no real experimental data.** All measurements are synthetic.
2. **No unsolved problems.** Every component is a known, published method. The work is in integrating, verifying and validating them.
3. **One NVIDIA RTX A4000 (16 GB).** Every headline result has to be reachable on it.
4. **No hidden assumptions.** Every assumption is a named, versioned, logged parameter.

## Results / Highlights

- **Phase 1 (Foundation) complete, gate G-1 met**, with one criterion amended by ADR-010.
- **1,488 tests** in the workspace environment plus **92** under FEniCSx.
- **`mypy --strict` clean**, **98 % core coverage**, and an `ASSUMED`-parameter count of **0**.
- Verification checks **V-01, V-02, V-03 and V-04 pass**; manifest reproduction gate **G-1.3 passes**.
- Verified on two machines: macOS/arm64 and the reference RTX A4000 under WSL2 Ubuntu 24.04.
- Building it corrected the baseline documents in places. For example, the self-consistent Child-Langmuir sheath thickness is 1.14 mm, not 0.89 mm (ADR-007).
- A research monograph and a conference paper are in [`docs/papers/`](docs/papers/), and the Techathon deck is in [`docs/presentation/`](docs/presentation/).

These figures are from the end of phase 1 and have not been re-run since.

## Tech Stack

- **Language:** Python 3.12 (core and physics also run on 3.11 for the GPU JAX build)
- **Scientific stack:** NumPy, SciPy (CODATA constants, special functions, linear algebra), SymPy (manufactured solutions), Pint (units)
- **PDE / kinetics:** FEniCSx with PETSc and MPI (L1 fluid), JAX on CUDA (L2 PIC)
- **Configuration:** OmegaConf, PyYAML
- **Storage:** HDF5 (h5py), Zarr, Apache Arrow / Parquet
- **Tooling:** uv workspace, Hatchling, Ruff, mypy (strict), pytest, Hypothesis
- **Papers:** LaTeX

## Getting Started

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Optional: FEniCSx / PETSc for the L1 fluid tests, and a CUDA GPU for the L2 PIC level

### Installation

```bash
git clone https://github.com/DCode-v05/IonLens.git
cd IonLens
uv sync
```

### Running

```bash
uv run ionlens run packages/ionlens-experiment/examples/b02-reference-operating-point.yaml
uv run ionlens reproduce <run-id>          # re-runs from the archived manifest and verifies
uv run ionlens compare <run-id-a> <run-id-b>
```

Run the checks:

```bash
uv run pytest                  # FEniCSx and GPU tests skip where the stack is missing
uv run mypy packages
uv run ruff check .
```

## Usage

- Describe an experiment in one YAML manifest — operating point, solver, instruments. Two examples are in [`packages/ionlens-experiment/examples/`](packages/ionlens-experiment/examples/).
- `ionlens run` executes it and writes artifacts that carry their own provenance.
- `ionlens reproduce` proves a past run is bit-identical; `ionlens compare` diffs two runs.
- Read the documents in [`docs/`](docs/) in order, starting with the [Charter](docs/00-charter.md), to see what each part is meant to do.

## Project Structure

```
IonLens/
├── packages/
│   ├── ionlens-core/          # Protocols, state model, units, parameter registry, provenance
│   ├── ionlens-physics/       # F₁: L0 analytic, L1 fluid (FEniCSx), L2 kinetic sheath models
│   ├── ionlens-instruments/   # F₂: collisional-radiative emission, lineshapes, spectrograph
│   ├── ionlens-inverse/       # Control parameters, priors, per-channel likelihoods
│   ├── ionlens-validation/    # Convergence studies, error norms, sealed truth, benchmarks
│   └── ionlens-experiment/    # Manifest engine and the ionlens run/reproduce/compare CLI
├── docs/
│   ├── 00-charter.md … 14-risk-register.md   # The specification
│   ├── adr/                   # Architecture decision records
│   ├── papers/                # Research monograph and conference paper (LaTeX + PDF)
│   ├── presentation/          # Techathon deck and presenter guide
│   └── commercialization/     # Commercialisation assessment
├── metrics/                   # ASSUMED-parameter baseline
├── refs/                      # Techathon thematic-domain deck and original proposal
├── pyproject.toml             # uv workspace, lint, type and test configuration
└── uv.lock
```

Licence: to be decided before any public release (see [Charter section 9](docs/00-charter.md)). Until then, all rights are reserved by Team Astrophage.

---

## Contact

<table>
  <tr><td><b>Portfolio:</b> <a href="https://www.denistan.me">Denistan</a></td><td><b>LinkedIn:</b> <a href="https://www.linkedin.com/in/denistanb">denistanb</a></td></tr>
  <tr><td><b>GitHub:</b> <a href="https://github.com/DCode-v05">DCode-v05</a></td><td><b>LeetCode:</b> <a href="https://leetcode.com/u/Denistan_B">Denistan_B</a></td></tr>
  <tr><td colspan="2" align="center"><b>Email:</b> <a href="mailto:denistanb05@gmail.com">denistanb05@gmail.com</a></td></tr>
</table>

Made with ❤️ by **Denistan B**
