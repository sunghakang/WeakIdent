# WeakIdent
Script for predicting differential equations using WeakIdent.

This repository provides implementation details of WeakIdent using Matlab and using Python. 

Copyright 2022, All Rights Reserved

Code author:  Mengyi Tang Rajchel

For Paper, ["WeakIdent: Weak formulation for Identifying Differential Equation using Narrow-fit and Trimming" by Mengyi Tang, Wenjing Liao, Rachel Kuske and Sung Ha Kang](https://arxiv.org/abs/2211.03134).

If you found WeakIdent useful in your research, please consider cite 

```
@misc{tang2022weakident,
      title={WeakIdent: Weak formulation for Identifying Differential Equations using Narrow-fit and Trimming}, 
      author={Mengyi Tang and Wenjing Liao and Rachel Kuske and Sung Ha Kang},
      year={2022},
      eprint={2211.03134},
      archivePrefix={arXiv},
      primaryClass={math.NA}
}
```


## Goal
This works focused on data-driven identifcation of differential equations when the given data are corrupted by noise. When the governing diferential equation is a linear combination of various differential terms, the identifcation problem can be formulated as solving a linear system, with the feature matrix consisting of linear and nonlinear terms multiplied by a coeffcient vector. The goal is to identify the correct terms that form the equation to capture the dynamics of the given data. We propose a general and robust framework to recover differential equations using a weak formulation, for both ordinary and partial di erential equations (ODEs and PDEs).  We use weak formulation to facilitate an effcient and robust way to handle noise. For a robust recovery against noise and the choice of hyper-parameters, we introduce two new mechanisms, narrow-fit and trimming, for the coeffcient support and value recovery, respectively. 

## The structure of this repository.
```
├── README.md
├── WeakIdent-Matlab
└── WeakIdent-Python
    ├── README.md
    ├── configs
    ├── environment.yml
    ├── main.py
    ├── model.py
    ├── output
    └── utils.py
```

## WeakIdent - Matlab 
run `main.m` to use WeakIdent on partial differential equations and and ode systems.

`weakIdentV4`: script for predicting PDE equations using WeakIdent
`datasetV2`: folder containing datasets used in the paper.

## WeakIdent - Python

### Environment set-up
run `conda env create -f environment.yml` to create the environment.

run `conda activate test_env1` to activate the environment.

### Run WeakIdent on provided datasets

run `python main.py --config configs/$configuration file name$.yaml` to predict differential equation using a pre-simulated dataset. For example, to predict Transport Equation (with diffusion), run `python main.py --config configs/config_1.yaml` to see the output:

```
Start loading arguments and dataset for Transport Equation
Start building feature matrix W:
[===========================================] 100.0% 
Start building scale matrix S:
[===========================================] 100.0% 
The number of rows in the highly dynamic region is  933

 Start finding support: 
[=========] 100.0% 
Finished support trimming and narrow fit for variable no.1 . A support is found.

 ------------- coefficient vector overview ------noise-signal-ratio : 0.5  -------
╒════╤════════════════╤════════════╤════════════╕
│    │ feature        │   true u_t │   pred u_t │
╞════╪════════════════╪════════════╪════════════╡
│  0 │ 1              │       0    │  0         │
├────┼────────────────┼────────────┼────────────┤
│  1 │ u              │       0    │  0         │
├────┼────────────────┼────────────┼────────────┤
│  2 │ u_{x}          │      -1    │ -0.986752  │
├────┼────────────────┼────────────┼────────────┤
│  3 │ u_{xx}         │       0.05 │  0.0502687 │
├────┼────────────────┼────────────┼────────────┤
│  4 │ u_{xxx}        │       0    │  0         │
├────┼────────────────┼────────────┼────────────┤
│  5 │ u_{xxxx}       │       0    │  0         │
├────┼────────────────┼────────────┼────────────┤
│    │   ......       │     ...    │     ...    │
├────┼────────────────┼────────────┼────────────┤
│ 40 │ (u^6)_{xxxx}   │       0    │  0         │
├────┼────────────────┼────────────┼────────────┤
│ 41 │ (u^6)_{xxxxx}  │       0    │  0         │
├────┼────────────────┼────────────┼────────────┤
│ 42 │ (u^6)_{xxxxxx} │       0    │  0         │
╘════╧════════════════╧════════════╧════════════╛

 ------------- equation overniew ------noise-signal-ratio : 0.5  -------------------
╒════╤═════════════════════════════════╤═══════════════════════════════════╕
│    │ True equation                   │ Predicted equation                │
╞════╪═════════════════════════════════╪═══════════════════════════════════╡
│  0 │ u_t = - 1.0 u_{x} + 0.05 u_{xx} │ u_t = - 0.987 u_{x} + 0.05 u_{xx} │
╘════╧═════════════════════════════════╧═══════════════════════════════════╛

 ------------------------------ CPU time: 0.83 seconds ------------------------------

 Identification error for Transport Equation from WeakIdent: 
╒════╤═══════════╤════════════════╤═════════════╤═════════╤═════════╕
│    │     $e_2$ │   $e_{\infty}$ │   $e_{res}$ │   $tpr$ │   $ppv$ │
╞════╪═══════════╪════════════════╪═════════════╪═════════╪═════════╡
│  0 │ 0.0132347 │      0.0132485 │    0.449226 │       1 │       1 │
╘════╧═══════════╧════════════════╧═════════════╧═════════╧═════════╛

```


### Datasets
We use folder `dataset-Python` to store sample dataset for multiple equations including true coefficients.
Each configuration file in `configs` is associated with one experiments. See the following table for type of equations:

| config file  index       | Equation name      | 
|:-------------:|-------------|
|1     |  Transport Equation |  
| 2     | Reaction Diffusion Equation    | 
| 3 | Anisotropic Porous Medium (PM) Equation    |
| 4 | Heat Equation | 
| 5 | Korteweg-de Vires (KdV) Equation | 
| 6 | Kuramoto-Sivashinsky (KS) Equation | 
| 7 | Nonlinear Schrodinger (NLS) | 
| 8 | 2D Linear System | 
| 9 | Nonlinear System (Van der Pol) | 
| 10 | Nonlinear System (Duffing) | 
| 11 | Noninear System (Lotka-Volterra) | 
|12| Nonlinear System (Lorenz) | 

We refer details of each dataset to the experimental result section in *WeakIdent: Weak formulation for Identifying
Differential Equation using Narrow-fit and Trimming*


Remark: In order to run WeakIdent on reaction diffusion type equation, please click [here](https://www.dropbox.com/t/TKK9U1ttVwX2mfHP) to download the 
dataset into folder`dataset-Python`, or run `simulate_reaction_diffusion_eqn.py` before running 
`python main.py --config configs/config_2.yaml`. 


### Sample output
We provide sample output for each equation(dataset) in  `output`.

### Credit/note
Build feature matrix through convolution (using fft), this part of the code is modified from `get_lib_columns()` (originally Matlab version) from [WeakSindyPde](https://github.com/dm973/WSINDy_PDE).

