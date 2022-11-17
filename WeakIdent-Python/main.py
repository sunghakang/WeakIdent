import argparse
import yaml
from utils.data import load_data, add_noise
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
parser.add_argument('--config', default='./configs/config_1.yaml')


def main():
    run()


def run():
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    try:
        print('Start loading arguments and dataset', args.filename, 'for', args.equation)
    except AttributeError:
        print('NameError occurred. The name of given data / equation is not defined in config file.') 
        return

    try: 
        u, xs, true_coefficients = load_data('dataset-Python', args.filename)
    except FileNotFoundError:
        print('FileNotFoundError occurred. Dataset for ', args.equation, 'is missing')
        return
    
    try: 
        sigma_SNR = args.sigma_SNR
    except AttributeError:
        print('Noise signal ratio is missing. It is set to be 0.') 
        sigma_SNR = 0
    u_hat = add_noise(u, sigma_SNR)
    
    try:
        use_cross_der = args.use_cross_der
    except AttributeError:
        use_cross_der = False
        print('use_cross_der is set to be False.') 

    try:
        stride_x = args.stride_x
    except AttributeError:
        stride_x = 5
        print('Set up a defalt stride in space in feature subsampling') 

    try:
        stride_t = args.stride_t
    except AttributeError:
        stride_t = 5
        print('Set up a defalt stride in time in feature subsampling') 

    try:
        tau = args.Tau
    except AttributeError:
        tau = 0.05
        print('Set up a defalt trimming score in support trimming') 

    # set up default argument for dictionary building.    
    try:
        max_dx = args.max_dx
    except AttributeError:
        max_dx = 6
    
    try: 
        assert max_dx <=8, "The feature library is suggested not to be too large. Reset max_dx = 6."
    except AssertionError as msg:
        print(msg)
        max_dx = 6

    try:
        max_poly = args.max_poly
    except AttributeError:
        max_poly = 6

    try:
        identification_err_table, eqn_table, coe_table, run_time = weak_ident_pred(
            u_hat, xs, true_coefficients, max_dx, max_poly, stride_x,
            stride_t, use_cross_der, tau)
    except AttributeError:
        print('Additional necessary arguments are missing. Please make sure config file has all the arguments.')
        return
    
    print(
        "\n ------------- coefficient vector overview ------noise-signal-ratio : %s  -------" % (sigma_SNR))
    print(tabulate(coe_table, headers= coe_table.columns, tablefmt="grid"))
    print(
        "\n ------------- equation overview ------noise-signal-ratio : %s  -------------------"
        % (sigma_SNR))
    print(tabulate(eqn_table, headers= eqn_table.columns, tablefmt="grid"))
    
    print("\n ------------------------------ CPU time: %s seconds ------------------------------" %
          (round(run_time, 2)))
    print("\n Identification error for", args.equation, "from WeakIdent: ")
    print(tabulate(identification_err_table, headers= identification_err_table.columns, tablefmt="grid"))

if __name__ == '__main__':
    main()
