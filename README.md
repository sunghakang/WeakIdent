# WeakIdent
Script for predicting differential equations using WeakIdent.

This repository provides implementation details of WeakIdent using Matlab and using Python. 

Script for **identifying differential equations** using WeakIdent.

This repo provides implementation details of WeakIdent using Python. 

Copyright 2022, All Rights Reserved

**Code author:  Mengyi Tang Rajchel**

For Paper, "[:link:WeakIdent: Weak formulation for Identifying Differential Equation using Narrow-fit and Trimming](https://arxiv.org/abs/2211.03134)" by Mengyi Tang, Wenjing Liao, Rachel Kuske and Sung Ha Kang.

:blush: If you found WeakIdent useful in your research, please consider cite us:

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

## What  does WeakIdent do?
WeakIdent is a general and robust framework to recover differential equations using a weak formulation, for both ordinary and partial differential equations (ODEs and PDEs). 
Noisy time series data are taken in with spacing as input and output a governing equation for this data.


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

**Required packages**
`sys, yaml, argparse, time, typing, pandas, tabular, numpy, numpy_index, Scipy`

**Set-up**

[Option 1] If you do not have `conda` installed, you can use `pip install` to install the packages listed above.

[Option 2] (1) run `conda env create -f environment.yml` to create the environment. (2) run `conda activate test_env1` to activate the environment.


### Datasets
Sample datasets from various type of equations including true coefficients. can be found in folder `dataset-Python`. For each dataset, there exists a 
configuration file in `configs` that specifies the input argument to run WeakIdent. The following table provide equations names of each dataset:

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
|13| Noninear System 2D (Lotka-Volterra) |

We refer details of each dataset to the experimental result section in *WeakIdent: Weak formulation for Identifying Differential Equation using Narrow-fit and Trimming*

**Remark**

The dataset for reaction diffusion type equation and Nonlinear Lotka-Volterro equation is sligher larger (100-200 M). They are not provided in `dataset-Python`.
-  In order to run WeakIdent on reaction diffusion type equation

   [option 1] please click [here](https://www.dropbox.com/t/TKK9U1ttVwX2mfHP) to download the dataset into folder`dataset-Python`, or 

   [option 2] run `simulate_reaction_diffusion_eqn.py` to simulate the dataset before running 
`python main.py --config configs/config_2.yaml`.  

- To run WeakIdent on Nonlinear System 2D (Lotka-Volterra), please run `simulate_lotka_volterra_2d_ode_eqn.py` to simulate the dataset before running `python main.py --config configs/config_13.yaml`.

### Run WeakIdent on provided datasets
There are 13 datasets provided in `dataset-Python`. To run WeakIdent on each indivial dataset, 
run `python main.py --config configs/config_$n$ file name$.yaml` to identify differential equation using a pre-simulated dataset specified in `configs/config_$n$.yaml`. 


**An example of running WeakIdent on Transport Equation with diffusion**

Run `python main.py --config configs/config_1.yaml` to see the output:

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
                      ......
│ 41 │ (u^6)_{xxxxx}  │       0    │  0         │
├────┼────────────────┼────────────┼────────────┤
│ 42 │ (u^6)_{xxxxxx} │       0    │  0         │
╘════╧════════════════╧════════════╧════════════╛

 ------------- equation overview ------noise-signal-ratio : 0.5  -------------------
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

### More sample output for each dataset
We provide sample output for each equation(dataset) in  `output`.

## Credit/note
Build feature matrix through convolution (using fft), this part of the code is modified from `get_lib_columns()` (originally Matlab version) from [WeakSindyPde](https://github.com/dm973/WSINDy_PDE).
