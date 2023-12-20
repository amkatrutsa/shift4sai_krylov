# Practical shift choice in Shift-And-Invert Krylov method for matrix exponential evaluation

Prerequisites:
- NumPy >= 1.16, SciPy >= 1.1, Matplotlib >= 3
- PyAmg >= 4.0

## Examples

* Convection-diffusion equation with piecewise-constant coefficients:
  - [Optimize-and-run method](./convection_diffusion/opt-and-run.ipynb)
  - [Inceremental method](./convection_diffusion/incremental.ipynb)
* Anisotropic diffusion equation
  - [Optimize-and-run method](./anisotropic_diffusion/opt-and-run.ipynb)
  - [Inceremental method](./anisotropic_diffusion/incremental.ipynb)

* Source code for the SAI Krylov method is available [here](./src/sai.py)


## Citing
If you use this code in your research, please cite our preprint:
```
@article{katrutsa2019practical,
  title={Practical shift choice in the shift-and-invert Krylov subspace evaluations of the matrix exponential},
  author={Katrutsa, Alexandr and Botchev, Mike and Oseledets, Ivan},
  journal={arXiv preprint arXiv:1909.13059},
  year={2019}
}
```
