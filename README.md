# Reciprocal Transform for Painlevé-Type Equations

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23814431.svg)](https://doi.org/10.5281/zenodo.23814431)

Code and data for the paper:

> **Reciprocal Transform for Painlevé-Type Equations: An Empirical Study of PINN Performance Near Movable Singularities**
> Maryam Jalili (2026). Submitted for publication.

## Abstract

We investigate the performance of physics-informed neural networks (PINNs) for second-order ODEs with movable singularities, focusing on the prototype y'' = y^3 and Painlevé II. We compare three representations: direct, rapidity, and reciprocal transform w = 1/y. The reciprocal representation reduces median near-pole error by up to three orders of magnitude but exhibits a 20% seed failure rate and a weak residual-error correlation. A double-pole test (y'' = 6y^2) yields a fitted scaling of δ^{-2.06}. The PINN is not competitive with DOP853 for single solves. These findings highlight both potential and limitations of PINNs for singular ODEs.
## Reproduce

```bash
pip install -r requirements.txt
python experiments/M1_ablation.py                # 2.5 h  -- Table M1
python experiments/phase2_minimal_prototype.py   # 4 h    -- Table 1
python experiments/p2_multiseed_full.py          # 7-12 h -- Table 3 (resume-capable)
python experiments/generate_figures.py           # 30 s   -- figures
```

## Requirements

- Python 3.14 (tested on Windows 10)
- PyTorch 2.x (CPU, no GPU required)
- NumPy 2.5.1
- SciPy 1.14 (DOP853 via solve_ivp)
- matplotlib 3.9

Exact versions: see requirements.txt.

## Hardware

All experiments ran on:
- CPU: Intel Core i7 (exact model -- to be added)
- RAM: 8 GB
- OS: Windows 10
- No GPU used

Training time for one seed: ~10-30 minutes depending on epochs.
DOP853 reference: 18-46 ms per solve.

## Data

All numerical results are in results/:

| File | Contents |
|---|---|
| M1_ablation.log | M1 ablation (Table M1, 5 configs, seed 42) |
| phase2_proto_5seed.npz | Prototype, 5 seeds (Table 1 preliminary) |
| phase2_proto_10seed_extended.npz | Prototype, 10 seeds (Table 1 final) |
| p2_multiseed_full.npz | P-II, 5 seeds x 3 alpha x 3 reps (Table 3) |
| ablation_metrics.npz | Full ablation |
| seed_robustness.npz | Multi-seed for prototype |
| residual_gap_scatter.pdf | Residual-error gap scatter (Spearman rho=0.00, 5 seeds) |

## Code Structure

- src/substitutions.py -- residual functions for the three representations (prototype)
- src/painleve2.py -- residual functions for Painlevé II
- src/pinn_solver.py -- PINN architecture and training loop
- src/reference_dop853.py -- DOP853 reference solution
- experiments/M1_ablation.py -- M1: transform vs reweighting
- experiments/phase2_minimal_prototype.py -- 5-seed prototype
- experiments/p2_multiseed_full.py -- P-II multi-seed
- experiments/generate_figures.py -- figure generation

## Citation

```bibtex
@article{Jalili2026Reciprocal,
  author  = {Jalili, Maryam},
  title   = {Reciprocal Transform for {Painleve}-Type Equations:
             An Empirical Study of PINN Performance Near Movable Singularities},
  journal = {(to be determined)},
  year    = {2026},
  note    = {submitted}
}
```

## License

MIT (see LICENSE).

## Contact

Maryam Jalili -- mrmjlili@iau.ac.ir
