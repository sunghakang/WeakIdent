#!/usr/bin/env python
# coding: utf-8
import argparse
import yaml
import numpy as np
from utils import load_data, add_noise
from model import weak_ident_pred
from tabulate import tabulate

"""

Main function of weakident to predict partial/ordinary different equations.

Copyright 2022, All Rights Reserved
Code by Mengyi Tang Rajchel
For Paper, "WeakIdent: Weak formulation for Identifying
Differential Equation using Narrow-fit and Trimming"
by Mengyi Tang, Wenjing Liao, Rachel Kuske and Sung Ha Kang

"""
parser = argparse.ArgumentParser(
    description='Implementation of WeakIdent in Python')
parser.add_argument('--config', default='./config_1.yaml')


def main():
    run()


def run():
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    print('Start loading arguments and dataset for', args.equation)
    u, xs, true_coefficients = load_data('dataset-Python', args.filename)
    u_hat = add_noise(u, args.sigma_SNR)
    identification_err_table, eqn_table, coe_table, run_time = weak_ident_pred(
        u_hat, xs, true_coefficients, args.max_dx, args.max_poly, args.stride_x,
        args.stride_t, args.use_cross_der, args.Tau)

    print(
        "\n ------------- coefficient vector overview ------noise-signal-ratio : %s  -------" % (args.sigma_SNR))
    print(tabulate(coe_table, headers= coe_table.columns, tablefmt="grid"))
    print(
        "\n ------------- equation overview ------noise-signal-ratio : %s  -------------------"
        % (args.sigma_SNR))
    print(tabulate(eqn_table, headers= eqn_table.columns, tablefmt="grid"))
    
    print("\n ------------------------------ CPU time: %s seconds ------------------------------" %
          (round(run_time, 2)))
    print("\n Identification error for", args.equation, "from WeakIdent: ")
    print(tabulate(identification_err_table, headers= identification_err_table.columns, tablefmt="grid"))

if __name__ == '__main__':
    main()
