# Reciprocal Transform for Painlevé-Type Equations

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23814431.svg)](https://doi.org/10.5281/zenodo.23814431)

Code and data for the paper:

> **Reciprocal Transform for Painlevé-Type Equations: A Mathematical Regularization with Physics-Informed Neural Network Verification**
> Maryam Jalili, *Communications in Nonlinear Science and Numerical Simulation (CNSNS)* (submitted)

## Abstract

We propose the reciprocal transform w = 1/y as a regularization for second-order ODEs with movable poles, focusing on the prototype y'' = y^3 and Painlevé II. The transformation, together with the w^3-weighted residual (which is the natural residual for the reciprocal equation), reduces the median near-pole relative error by a factor of ~1450 for the prototype (10 seeds; median 517× vs 5-seed baseline) and ~139–523 for Painlevé II (5 seeds × 3 α, median). An ablation separates the transformation from the reweighting: the transformation alone and the reweighting alone both fail; only their combination produces the reciprocal-PINN result.

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
             A Mathematical Regularization with Physics-Informed
             Neural Network Verification},
  journal = {Communications in Nonlinear Science and Numerical Simulation},
  year    = {2026},
  note    = {submitted}
}
```

## License

MIT (see LICENSE).

## Contact

Maryam Jalili -- mrmjlili@iau.ac.ir
