# IonLens

IonLens is a framework for reconstructing the ion energy flux at a plasma sheath from non-perturbing optical measurements only: optical emission spectra, laser-induced fluorescence lineshapes, Thomson scattering spectra and interferometric phase. It is built as a regularised, Bayesian inverse problem, and everything runs on synthetic data. It is not a plasma simulator or a digital twin.

Ion energy flux decides how fast a wafer etches or a coating grows, but it is hard to measure. Probes that see it disturb it, and probes that do not disturb it see only part of the picture. The question here is whether the flux can be recovered in space and time from optical signals alone, with honest uncertainty.

It was built by Team Astrophage for Techathon 2.0 (Naari Shakti), track "Plasma Technologies for Industry & Society", hosted by AIC RRCAT π-Hub Foundation, DAE.

## How it works

A forward chain turns a known plasma state into synthetic measurements, and the inverse solver recovers the state from those measurements without seeing the truth. A validation engine then compares the result with the hidden truth and breaks down the error budget. Without a laboratory, that closed loop is the evidence for the accuracy claims.

```
true plasma state x                (known because we synthesised it)
   |
   v  F1  sheath / fluid / kinetic (PIC) physics
   v  F2  collisional-radiative emission, LIF, Thomson, refractive index
   v  F3  optical transport: geometry, apertures, aberration, stray light
   v  F4  detector: QE, shot / read / dark noise, ADC, dead pixels
   |
measurements y = F(x) + noise      <- the truth x is now hidden
   |
   v  physics-constrained inverse solver (regularised / Bayesian)
   |
recovered state, posterior p(x|y)  ->  ion energy flux with uncertainty
   |
   v  validation engine: compare with hidden truth, decompose the error budget
```

Main design points:

- The inverse layer depends on no forward model (the "sealed truth" barrier), and a test checks the import graph so it stays that way.
- Physics comes in levels: L0 analytic sheath models as the verification anchor, an L1 fluid solver on FEniCSx, and an L2 particle-in-cell level that targets JAX on the GPU.
- Atomic data loaders cover LXCat cross sections, NIST ASD transition data and OpenADAS, plus Boltzmann / EEDF integration.
- A parameter registry with a literal-lint rule: every number traces to a published source, a standard database or a declared design parameter, and guesses are tagged `ASSUMED`.
- One YAML manifest describes an experiment. `ionlens run` executes it, `ionlens reproduce` re-checks a past run bit for bit, and `ionlens compare` diffs two runs. Provenance is embedded in every artifact written to HDF5, Zarr or Parquet.
- Verification and validation are kept apart, with separate suites, reports and gates.

The plan was written before the code. The numbered documents in [`docs/`](docs/) (charter through risk register) are the specification, and code is written against them. When the code shows a document is wrong, the fix goes into an architecture decision record in [`docs/adr/`](docs/adr/) instead of being applied silently.

Constraints the project works under: no hardware or real experimental data, only known published methods, and every headline result has to be reachable on one NVIDIA RTX A4000 (16 GB).

## Results

From the previous README, as of the end of phase 1:

- Phase 1 (foundation) gate G-1 met, with one criterion amended by ADR-010.
- 1,488 tests in the workspace environment plus 92 under FEniCSx.
- `mypy --strict` clean, 98 % core coverage and an `ASSUMED`-parameter count of 0.
- Verification checks V-01 to V-04 pass, and the manifest reproduction gate G-1.3 passes.
- Run on macOS/arm64 and on the RTX A4000 under WSL2 Ubuntu 24.04.
- The self-consistent Child-Langmuir sheath thickness came out at 1.14 mm rather than the 0.89 mm in the baseline documents (ADR-007).

A research monograph and a conference paper are in [`docs/papers/`](docs/papers/), and the Techathon deck is in [`docs/presentation/`](docs/presentation/).

## Tech stack

- Python 3.12 (the GPU JAX build also runs on 3.11)
- NumPy, SciPy, SymPy, Pint
- FEniCSx with PETSc and MPI for the fluid level, JAX on CUDA for PIC
- OmegaConf, PyYAML
- HDF5 (h5py), Zarr, Apache Arrow / Parquet
- uv workspace, Hatchling, Ruff, mypy (strict), pytest, Hypothesis
- LaTeX for the papers

## Setup

You need Python 3.12 and [uv](https://docs.astral.sh/uv/). FEniCSx / PETSc is optional (L1 fluid tests) and so is a CUDA GPU (L2 PIC).

```bash
git clone https://github.com/DCode-v05/IonLens.git
cd IonLens
uv sync
```

## Running

```bash
uv run ionlens run packages/ionlens-experiment/examples/b02-reference-operating-point.yaml
uv run ionlens reproduce <run-id>          # re-runs from the archived manifest and verifies
uv run ionlens compare <run-id-a> <run-id-b>
```

Checks:

```bash
uv run pytest                  # FEniCSx and GPU tests skip where the stack is missing
uv run mypy packages
uv run ruff check .
```

Two example manifests are in [`packages/ionlens-experiment/examples/`](packages/ionlens-experiment/examples/).

## Project structure

```
IonLens/
├── packages/
│   ├── ionlens-core/          # protocols, state model, units, parameter registry, provenance
│   ├── ionlens-physics/       # F1: L0 analytic, L1 fluid (FEniCSx), L2 kinetic sheath models
│   ├── ionlens-instruments/   # F2: collisional-radiative emission, lineshapes, spectrograph
│   ├── ionlens-inverse/       # control parameters, priors, per-channel likelihoods
│   ├── ionlens-validation/    # convergence studies, error norms, sealed truth, benchmarks
│   └── ionlens-experiment/    # manifest engine and the ionlens CLI
├── docs/
│   ├── 00-charter.md ... 14-risk-register.md   # the specification
│   ├── adr/                   # architecture decision records
│   ├── papers/                # research monograph and conference paper (LaTeX + PDF)
│   ├── presentation/          # Techathon deck and presenter guide
│   └── commercialization/     # commercialisation assessment
├── metrics/                   # ASSUMED-parameter baseline
├── refs/                      # original proposal and Techathon deck
├── pyproject.toml             # uv workspace, lint, type and test configuration
└── uv.lock
```

Licence: to be decided before any public release (see [Charter section 9](docs/00-charter.md)). Until then all rights are reserved by Team Astrophage.

---

## Contact

<table>
  <tr><td><b>Portfolio:</b> <a href="https://www.denistan.me">Denistan</a></td><td><b>LinkedIn:</b> <a href="https://www.linkedin.com/in/denistanb">denistanb</a></td></tr>
  <tr><td><b>GitHub:</b> <a href="https://github.com/DCode-v05">DCode-v05</a></td><td><b>LeetCode:</b> <a href="https://leetcode.com/u/Denistan_B">Denistan_B</a></td></tr>
  <tr><td colspan="2" align="center"><b>Email:</b> <a href="mailto:denistanb05@gmail.com">denistanb05@gmail.com</a></td></tr>
</table>
