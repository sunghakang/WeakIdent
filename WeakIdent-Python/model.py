from utils.feature_library_building import build_feature_vector_tags, compute_test_function_para
from utils.output_writing import write_output_tables
from utils.calculations import circshift, two_piece_fit_v2, least_square_adp, compute_cross_validation_error, compute_cross_validation_err_v2

from utils.helpers import set_hist_bins, set_sparsity_level
import numpy as np
import scipy.linalg
import numpy_indexed as npi
import sys
import pandas as pd
import time
from typing import Tuple
"""

Modeling functions of WeakIdent to identify partial/ordinary different equations.

Code author:  Mengyi Tang Rajchel.

Copyright 2022, All Rights Reserved

For Paper, "WeakIdent: Weak formulation for Identifying
Differential Equation using Narrow-fit and Trimming"
by Mengyi Tang, Wenjing Liao, Rachel Kuske and Sung Ha Kang

"""


def weak_ident_pred(
        u_hat: np.array, xs: np.array, true_coefficients: np.array,
        max_dx: int, max_poly: int, skip_x: int, skip_t: int,
        use_cross_der: bool,
        tau) -> Tuple[np.array, np.array, pd.core.frame.DataFrame, float]:
    """
    Identify a system of partial differential equations or 
    ordinary differential equations using weak_ident. 
    More details of this work (WeakIdent) can be found through this link: https://arxiv.org/abs/2211.03134.

    For better visualization of input and output , I use an example of Transport 
    equation with diffusions. $u_t = -u_x + 0.05u_{xx}$ in my illustration.
    For this example, an expected good identification result should be only feature u_x 
    and u_{xx} existing in the identified support with associated coefficient values 
    being -1 and 0.05.

    Args:
        u_hat (np.array): array of shape (n,). This is the given noisy data with n variables.
        #################### an example input of transport eqn ######################
        #  e.g. u_hat[0] has shape (257, 300) where u_hat[0][i][j] represents given #
        #  noisy data at x = x_i and t = t_j for i = 0,1,...,256 and j = 0, 1 ,..., #
        #  299. In this example, the given noisy data is                            #
        #    u_hat = [array([[-0.00753292,  0.00909885, ...,  -0.02589293],...,     #
        #             [-0.01762188, -0.00837559,  ...,  0.04464247]])      ]        #
        #############################################################################

        xs (np.array): array of shape (dim_x + 1,). This is the spacing of given data.
        #################### an example input of transport eqn ######################
        # In this case, dim_x = 1, hence xs can be expressed by                     #
        # xs = [(x_1, x_2, ...)^T, (t_1, t_2,...)].                                 #
        # xs = [array([[0.        ],[0.00390625],[0.0078125 ], ..., [1.        ]])  #
        #    array([[0.    , 0.0001, 0.0002, ... 0.0299]])      ]                   #
        # Remark: in the case of dim_x = 2, xs = [(x_1, x_2, ...)^T,                #
        #                                   (x_1, x_2, ...)^T, (t_1, t_2,...)].     #
        #  In this work, equal spacing is required for given data.                  #
        #############################################################################

        true_coefficients: array of shape (n,). This variable stores the true coefficient 
        along with appropriate tags for given data. This information is used for validating
        identification results. true_coefficients[i] has shape (l, n + dim_x + 1). 
        #################### an example input of transport eqn ######################
        # true_coefficient =  [array([[ 1.  ,  1.  ,  0.  , -1.  ],                 #
        #                             [ 1.  ,  2.  ,  0.  ,  0.05]])]               #
        # true_coefficient[0] denotes true feature - u_x                            #
        # true_coefficient[1] denotes true feature + 0.05 u_{xx}                    #
        # Remark: the first n column denote the monomial base power of each feature.#
        # The n+1 - n+dim_x the column denotes the derivative of monomial along x   #
        # direction. The n+ column denotes the derivative of monommial along t      #
        # direction. The last column denotes the coefficient value for each feature #
        #############################################################################

        max_dx (int): maximum total order of partial derivatives.
        max_poly (int): maximum total order of monomials.
        #################### an example input of transport eqn ######################
        # max_dx = 6, max_poly = 6. Suggested values: max_dx = 4-6, max_poly = 3-6  #
        # e.g. In this example, there are 43 features in the dictionary.            #
        # They are 1, u, u_x, u_{xx}, ..., u_{xxxxxx}, u^2, (u^2)_x, ...,           #
        # (u^2)_{xxxxxx}, ..., u^6, ....,  (u^6)_{xxxxxx}                           #
        #############################################################################

        skip_x (int): number of skipped steps in spatial domain when downsampling 
        feature matrix.
        skip_t (int): number of skipped steps in temporal domain when downsampling 
        feature matrix.
        #################### an example input of transport eqn ######################
        # skip_x = 5, skip_t = 6 .                                                  #
        # To improve computation efficiency, the built feature matrix is downsampled#
        # W  with rate 1/5 and 1/6 in space and time respectively.                  #
        #############################################################################

        use_cross_der (bool): whether to allow partial derivatives exist in the dictionary.
        #################### an example input of transport eqn ######################
        # use_cross_der = False .                                                   #
        # use_cross_der is consider only when dealing with 2d spatial domain.       #
        # Transport equation is on 1d. Hence, use_cross_der = False in this example.#
        #############################################################################

        tau (float): trimming threshold.
        #################### an example input of transport eqn ######################
        # tau = 0.05.                                                               #
        # tau is the trimming score for a given support A recovered from Subspace   #
        # Persuit. The support is trimmed by removing features with contributions   #
        # below a threshold. The default number is 0.05.                            #
        #############################################################################     

    Returns:
        Tuple[pd.core.frame.DataFrame, pd.core.frame.DataFrame, pd.core.frame.DataFrame, float]: 
        Identification error table,  Equation table, Coefficient vector table,
        running time   
        #### an example output of df_identification_errors for transport eqn #######
        +----+-----------+----------------+-------------+---------+---------+
        |    |     $e_2$ |   $e_{\infty}$ |   $e_{res}$ |   $tpr$ |   $ppv$ |
        +====+===========+================+=============+=========+=========+
        |  0 | 0.0102789 |      0.0102866 |    0.263103 |       1 |       1 |
        +----+-----------+----------------+-------------+---------+---------+ 
        Remark: here $e_2$ denotes relative l_2 norm error, 
                     $e_{\infty}$ denotes relative l_infty norm error,
                     $e_{res}$ denotes residual error,
                     $tpr$ denotes true positive rate, and
                     $ppv$ denotes positive predictive rate.
                A good recovery should have small $e_2$ and $tpr=1$, $ppv=1$. 
                This means a correct support is found with accurate coefficients.

        #### an example output of df_identified_eqns for transport eqn #############
        +----+---------------------------------+----------------------------------+
        |    | True equation                   | Predicted equation               |
        +====+=================================+==================================+
        |  0 | u_t = - 1.0 u_{x} + 0.05 u_{xx} | u_t = - 0.99 u_{x} + 0.05 u_{xx} |
        +----+---------------------------------+----------------------------------+

        #### an example output of df_coefficient_values for transport eqn #############
                +----+----------------+------------+------------+
                |    | feature        |   true u_t |   pred u_t |
                +====+================+============+============+
                |  0 | 1              |       0    |  0         |
                +----+----------------+------------+------------+
                |  1 | u              |       0    |  0         |
                +----+----------------+------------+------------+
                |  2 | u_{x}          |      -1    | -0.989713  |
                +----+----------------+------------+------------+
                |  3 | u_{xx}         |       0.05 |  0.0496751 |
                +----+----------------+------------+------------+
                |  4 | u_{xxx}        |       0    |  0         |
                                    ......
                +----+----------------+------------+------------+
                | 42 | (u^6)_{xxxxxx} |       0    |  0         |
                +----+----------------+------------+------------+
    """
    start_time = time.time()

    # determin number of variables of given data.
    num_of_variables = u_hat.shape[0]

    # determine spatial dimension of given data.
    dim_x = len(u_hat[0].shape) - 1

    # determine whether given dadta is 1-dimensional ODE.
    is_1d_ode = u_hat[0].shape[0] == 1

    # build feature tags for each left-hand-side feature and right-hand-side feature.
    dictionary_list, idx_of_lhs_feature, idx_of_rhs_feature, true_coefficient_vector = build_feature_vector_tags(
        num_of_variables, dim_x, max_dx, max_poly, use_cross_der,
        true_coefficients, is_1d_ode)

    # build feature matrix including left-hand-side feature and right-hand-side features and corresponding scale matrix
    feature_matrix, scale_matrix = build_feature_matrix_and_scale_matrix(
        u_hat, xs, max_dx, skip_x, skip_t, dictionary_list, is_1d_ode)

    # find a highly dynamic region from given data. idx_highly_dynamic_region stores the row index of feature
    # matrix that contains spatial-temporal features inside a highly dynamic region. This step helps with better
    # coefficient recover and idx_highly_dynamic_region will be used during narrow-fit.
    idx_highly_dynamic_region = find_highly_dynamic_region(dictionary_list,
                                                           scale_matrix, is_1d_ode,
                                                           num_of_variables, dim_x, max_dx)
    # identify differential equation(s) from feature matrix.
    w, b, c_pred = diff_eqn_identification(tau, num_of_variables, dim_x,
                                           is_1d_ode, idx_of_lhs_feature,
                                           idx_of_rhs_feature, feature_matrix,
                                           scale_matrix, idx_highly_dynamic_region)

    run_time = time.time() - start_time

    # write output tables.
    # (1) a table of identified equations and true equations ;
    # (2) a table of identified features with associated coefficient values
    # (3) a table of identification error (relative l_2 norm error, relative l_infty norm error, residual error,
    #     true positive rate, and positive predictive value)

    df_identified_eqns, df_coefficient_values, df_identification_errors = write_output_tables(
        num_of_variables, true_coefficient_vector, dim_x, is_1d_ode,
        dictionary_list, idx_of_lhs_feature, idx_highly_dynamic_region, w, b, c_pred)

    return df_identification_errors, df_identified_eqns, df_coefficient_values, run_time

def build_feature_matrix_and_scale_matrix(
        u_hat: np.array, xs: np.array, max_dx: int, skip_x: int, skip_t: int,
        dict_list: np.array, is_1d_ode: bool) -> Tuple[np.array, np.array]:
    '''
    This function builds a feature matrix W and a scale matrix S for given data.

    Args:
        u_hat (np.array): array of shape (n,) , this is given noisy data with n variables.
        xs (np.array): array of shape (dim_x + 1,), spacing of given data.
        max_dx (int): maximum total order of partial derivatives (including multiple dimension derivatives).
        skip_x (int): number of skipped steps in spatial domain when downsampling the feature matrix.
        skip_t (int): number of skipped steps in temporal domain when downsampling the feature matrix.
        dict_list(np.ndarray): array of shape (L, n + dim_x + 1). Here each row represents a feature,
                            column 1 - column n represent the degree of monomial for each feature,
                            column n+1 - column n+dim_x represent the order of partial derivatives along each spatial domain,
                            and column n+dim_x+1 represents the order of partial derivatives along temporal
                            domain.  
        is_1d_ode (bool): whether given data is 1d-ode type.
    Returns:
        Tuple[np.array, np.array]: feature matrix W and scale matrix S (including lhs).                        

    '''
    m_x, m_t, p_x, p_t = compute_test_function_para(u_hat, xs, max_dx)
    dx = 0
    if not is_1d_ode:
        dx = xs[0].flatten()[1] - xs[0].flatten()[0]
    dt = xs[-1].flatten()[1] - xs[-1].flatten()[0]

    if is_1d_ode:
        w_b_large, s_b_large = compute_feature_and_scale_matrix_ode(
            m_x, p_x, skip_x, dict_list, u_hat, dt)
    else:
        w_b_large, s_b_large = compute_feature_and_scale_matrix_pde(
            m_x, m_t, p_x, p_t, skip_x, skip_t, dict_list, u_hat, dx, dt,
            max_dx)

    return w_b_large, s_b_large

def find_highly_dynamic_region(dict_list: np.array, s_b: np.array,
                               is_1d_ode: bool, num_of_variables: int,
                               dim_x: int,
                               max_dx: int) -> np.array:
    """
    This function returns the index of rows in the feature matrix where (x_i,t_n) is located in the 
    highly dynamic region.

    Args:
        dict_list(np.ndarray): array of shape (L, n + dim_x + 1). Here each row represents a feature.
                               Column 1 - column n represent the degree of monomial for each feature,
                               column n+1 - column n+dim_x represent the order of partial derivatives 
                               along each spatial domain, and column n+dim_x+1 represents the order of 
                               partial derivatives along temporal domain. 
        idx_interesting_features (np.array): array of shape (m, n + 2) where each row provides the a tag 
                                             for an interesting feature.
        s_b (np.array): scale matrix S (inlcuding lhs).
        is_1d_ode (bool): whether given data is 1d-ode type.
        num_of_variables(int) : number of variables of given data.
        dim_x(int): spatial dimension of given data.
        max_dx (int): maximum total order of partial derivatives.

    Returns:
        np.array: row index of features located in highly dynamic region
    """
    idx_interesting_features = find_idx_of_interesting_feature(
        num_of_variables, dim_x, is_1d_ode, max_dx)
    num_of_bins = set_hist_bins(is_1d_ode)
    idxs = np.zeros(idx_interesting_features.shape[0])
    dict_list = dict_list.astype(int)
    for i in range(idx_interesting_features.shape[0]):
        idxs[i] = np.flatnonzero(
            npi.contains([idx_interesting_features[i, :]], dict_list))
    idxs = idxs.astype(int)
    scales_sum = np.sum(np.abs(s_b[:, idxs]), axis=1)
    ind = scales_sum.argsort()
    hist, _ = np.histogram(scales_sum, bins=num_of_bins)
    hist = np.cumsum(hist)
    y = np.block([1, hist])
    transition_group_idx = two_piece_fit_v2(hist)
    idx_highly_dynamic = ind[y[transition_group_idx] - 1:y[-1]]
    return idx_highly_dynamic

def diff_eqn_identification(
        tau: float, num_of_variables: int, dim_x: int, is_1d_ode: bool,
        lhs_ind: np.array, rhs_ind: np.array, w_b_large: np.array,
        s_b_large: np.array,
        idx_highly_dynamic: np.array) -> Tuple[np.array, np.array, np.array]:
    """
    This function performs identification using given feature matrix (including lhs and rhs), scale matrix and row index of highly dynamic region.

    Args:
        tau (float): trimming score. Defaults to 0.05.
        num_of_variables(int) : number of variables of given data.
        dim_x(int): spatial dimension of given data.
        is_1d_ode (bool): whether given data is 1d-ode type.
        lhs_ind(np.ndarray): shape of (n,) , row index of left-hand-side feature.
        rhs_ind(np.ndarray): shape of (L,) , row index of right-hand-side features.
        w_b_large (np.array): original feature matrix.
        s_b_large (np.array): original scale matrix.
        idx_highly_dynamic (np.array): row index of features located in highly dynamic region.

    Returns:
        w(np.array):  original feature matrix.
        b(np.array):  original dynamic variable, it can be u_t or v_t in pdes.
        c(np.array):  identified sparse coefficient vector.
    """
    print("The number of rows in the highly dynamic region is ",
          idx_highly_dynamic.shape[0])

    scales_w_b = np.mean(np.abs(s_b_large[idx_highly_dynamic, :]), axis=0)
    scales_lhs_features, scales_rhs_features = scales_w_b[
        lhs_ind], 1 / scales_w_b[rhs_ind].flatten()

    w, w_tilda, w_narrow_tilda = compute_features_matrix_and_rescaled_matrix(
        rhs_ind, w_b_large, idx_highly_dynamic, scales_rhs_features)
    b, b_tilda, b_narrow_tilda = compute_features_matrix_and_rescaled_matrix(
        lhs_ind, w_b_large, idx_highly_dynamic, scales_lhs_features)

    c_pred = np.zeros(
        (w_b_large.shape[1] - num_of_variables, num_of_variables))

    # set up maximum sparsity level. For a predetermined dictionary, one can use a number <= # total features
    # as maximum sparsity level. This number is suggested to be an appropriate number to reduce computation cost.
    sparsity = set_sparsity_level(is_1d_ode, num_of_variables, dim_x)

    for num_var in range(num_of_variables):
        support_pred, coeff_sp = weak_ident_feature_selection(
            w, b, b_tilda, w_tilda, sparsity, scales_rhs_features,
            scales_lhs_features, w_narrow_tilda, b_narrow_tilda, num_var, tau)
        print(" ")
        print(
            'WeakIdent finished support trimming and narrow-fit for variable no.%d . A support is found this variable.'
            % (num_var + 1))
        c_pred[support_pred, num_var] = coeff_sp
    return w, b, c_pred

def weak_ident_feature_selection(w: np.array,
                                 b: np.array,
                                 b_tilda: np.array,
                                 w_tilda: np.array,
                                 k: int,
                                 s_rhs: np.array,
                                 s_lhs: np.float64,
                                 w_nar_tilda: np.array,
                                 b_nar_tilda: np.array,
                                 num_var: int,
                                 tau=0.05) -> np.ndarray:
    """
    This function returns the support identified by WeakIdent. See details in paper "WeakIdent: Weak formulation for Identifying
    Differential Equation using Narrow-fit and Trimming" by Mengyi Tang, Wenjing Liao, Rachel Kuske and Sung Ha Kang

    Note: There is another version (2.0) of this function coming very soon which reduced cpu by caching the considered supports.

    Args:
        w (np.array): (feature matrix array) of shape (N, L) where L is the number of rhs features and n 
                      is the number of variables.
        b (np.array): (dynamic variable u_t) of shape (N, 1).
        b_tilda (np.array): error normalized vector b of shape (N, 1).
        w_tilda (np.array): error normalized matrix w of shape (N, L).
        k (np.array): maximum allowed sparsity level.
        s_rhs (np.array): scale vector s for rhs features, array of shape (L, ).
        s_lhs (np.float64): scale number for lhs feature (u_t).
        w_nar_tilda (np.array): \Tilda{W}_narrow, error normalized feature matrix, array of shape (NxNt, L).
        b_nar_tilda (np.array): \Tilda{b}_narrow, error normalized dynamic variable, array of shape (NxNt, 1).
        tau (float, optional): trimming score. Defaults to 0.05.

    Returns:
        np.ndarray: a support vector that stores the index of finalized candiate features for each variable.
    """

    b_one_var = b[:, num_var].reshape(-1, 1)
    b_tilda_one_var = b_tilda[:, num_var].reshape(-1, 1)
    s_lhs_one_var = s_lhs[num_var]
    b_nar_tilda_one_var = b_nar_tilda[:, num_var].reshape(-1, 1)

    # this variable can be used to control whether you want to display the trimming process.
    display_tex = False

    support_list = {}
    cv_err_list = []
    w_column_norm = np.linalg.norm(w, axis=0).reshape(1, -1)
    column_normalized_w = w / w_column_norm
    print("\n Start finding support: ")
    for i in range(k):
        sys.stdout.write('\r')
        sys.stdout.write("[{:{}}] {:.1f}%".format("=" * i, k - 1,
                                                  (100 / (k - 1) * i)))
        sys.stdout.flush()
        support = subspace_persuit(column_normalized_w,
                                   b_one_var / np.linalg.norm(b_one_var, 2), i)
        if display_tex:
            print('SP -- Sparsity level is    :', i + 1)
            print('SP -- Support (Coef index) :', support)
        c_pred = narrow_fit(w_nar_tilda, b_nar_tilda_one_var, support, s_rhs,
                            s_lhs_one_var)
        trim_score = compute_trim_score(w_column_norm, support, c_pred)
        if display_tex:
            print('The trimming scores are: ', trim_score)
        support_list[i] = support
        cv_err_list.append(
            compute_cross_validation_err_v2(support, w_tilda, b_tilda_one_var))
        if display_tex:
            print('The CV residual error is: ', cv_err_list)
        while len(support) > 1:
            idx_least_imp_feature = np.where(trim_score == trim_score.min())[0]
            if trim_score[idx_least_imp_feature[0]] > tau:
                break
            idx_least_imp_feature = idx_least_imp_feature[0]
            support = np.delete(support, idx_least_imp_feature)
            c_pred = narrow_fit(w_nar_tilda, b_nar_tilda_one_var, support,
                                s_rhs, s_lhs_one_var)
            trim_score = compute_trim_score(w_column_norm, support, c_pred)
            if display_tex:
                print('For a smaller support  ', support, ',')
                print('the trimming score vector is ', trim_score)
            cv_err_list[i] = compute_cross_validation_err_v2(
                support, w_tilda, b_tilda_one_var)
            support_list[i] = support
            if display_tex:
                print(
                    'The current smallest residual error for sparsity level',
                    i + 1, ' is updated to be ', cv_err_list[i])
    cv_err_list = np.array(cv_err_list)
    cross_idx = np.argmin(cv_err_list)
    support_pred = support_list[int(cross_idx)]

    relative_scale = s_rhs / s_lhs_one_var
    coeff_sp = least_square_adp(
        w_nar_tilda[:, support_pred],
        b_nar_tilda_one_var) * (relative_scale[support_pred].reshape(-1, 1))

    return support_pred, coeff_sp.flatten()

def compute_features_matrix_and_rescaled_matrix(
        rhs_ind: np.array, w_b_large: np.array, idx_highly_dynamic: np.array,
        scales_rhs_features: np.array) -> Tuple[np.array, np.array, np.array]:
    """This function builds feature matrix, rescaled feature matrix, and rescaled narrower feature matrix (for narrow-fit)
       using scaling factor for ech feature.
    Args:
        rhs_ind(np.ndarray): shape of (L,) , row index of right-hand-side features.
        w_b_large (np.array): original feature matrix.
        idx_highly_dynamic (np.array): row index of features located in highly dynamic region
        scales_rhs_features (np.array): rescaling factor for right-hand-side features.

    Returns:
        Tuple[np.array, np.array, np.array]: feature matrix (rhs), rescaled feature matrix, rescaled narrower feature matrix 
    """
    w = w_b_large[:, rhs_ind]
    w_tilda = w * scales_rhs_features
    w_narrow_tilda = w_tilda[idx_highly_dynamic, :]
    return w, w_tilda, w_narrow_tilda

def find_idx_of_interesting_feature(n: int, dim: int,
                                    is_1d_ode: bool, max_dx: int) -> np.array:
    """
    This function returns the tags for interesting features which are used in error-normalization
    for Narrow-fit.

    Args:
        n (int): number of variable
        dim (int): spatial dimension (1 or 2)
        is_1d_ode (bool): whether or not given data is 1d ode data.
        max_dx (int): maximum total order of partial derivatives. This is used to distinguish interesting features 
        for multi-dimensional pde and multi-dimensional ode

    Returns:
        np.array: array which each row represents the tag of one interesting feature.
    """
    if is_1d_ode:
        idx = np.block([np.diag(np.ones(n)), np.zeros((n, 2))])
        idx = idx.astype(int)
    else:
        if max_dx == 0:
            idx = np.block([np.diag(np.ones(n)), np.zeros((n, dim + 1))])
            idx = idx.astype(int)
        else:
            if n == 1 and dim == 1:
                idx = np.array([[2, 1, 0]])
            elif n == 1 and dim == 2:
                idx = np.array([[2, 1, 0, 0], [2, 0, 1, 0], [3, 1, 1, 0]])
            elif n == 2 and dim == 1:
                idx = np.array([[0, 2, 1, 0], [2, 0, 1, 0]])
            elif n == 2 and dim == 2:
                idx = np.array([[0, 2, 0, 1, 0], [2, 0, 0, 1, 0], [0, 2, 1, 0, 0],
                                [2, 0, 1, 0, 0], [2, 1, 1, 0, 0], [1, 2, 0, 1, 0]])

    return idx

def narrow_fit(w: np.array, b: np.array, support: np.array, s_rhs: np.array,
               s_lhs: np.float64) -> np.array:
    """
    This function perform narrow least square fit for the problem wb = c.
    Notice that here w and c have to be error normalized features. Rescaling step should be done before this
    function is called. Narrow-fit is proposed in WeakIdent to help increase the accuracy of coefficients. A 
    hyghly dynamic region is found and scaling matrix is calculated to compute error normalized feature matrix 
    and normalized dynamic variable.

    Args:

        w (np.array): \Tilda{W}_narrow, error normalized feature matrix, array of shape (NxNt, L).
        b (np.array): \Tilda{b}_narrow, error normalized dynamic variable, array of shape (NxNt, 1).
        support (np.array): a support vector that stores the index of candiate features for each variable.
        s_rhs (np.array): scale vector s for rhs features, array of shape (L, ).
        s_lhs (np.float64): scale number for lhs feature (u_t).

    Returns:
        np.array: the coefficient values for a given support
    """
    c_pred = least_square_adp(w[:, support], b)
    c_pred = np.absolute(c_pred * s_rhs[support].reshape(-1, 1) / s_lhs)
    return c_pred

def compute_trim_score(w_column_norm: np.array, support: np.array,
                       c_pred: np.array) -> np.array:
    """
    This function computes the trimming score vector for given support and coefficient values.
    Args:
        w_column_norm (np.array): column norm vector of feature matrix w
        support (np.array): a support vector that stores the index of candiate features for each variable.
        c_pred (np.array): a coefficient vector that stores the values of candiate features for each variable.

    Returns:
        np.array: a trimming score vector for each candidate feature
    """
    trim_score = w_column_norm.flatten()[support] * c_pred.flatten()
    trim_score = trim_score / np.max(trim_score)
    return trim_score

def subspace_persuit(phi: np.array, b: np.array, k: int) -> np.array:
    """
    This function finds the support for a sparse vector c such that phi * x = b where k numbers in c 
    are nonzero. Note: Subspace Persuit is a greedy algorithm which minimizes residual error when searching
    for a support.

    Args:
        phi (np.array): array of shape (N, L)
        b (np.array): array of shape (N, 1)
        k (int): sparse level

    Returns:
        np.array: array of shape (k,). a support vector that stores the index of candiate features for each variable.
    """
    itermax = 15
    n = len(phi[0])
    is_disp = 0
    b_t = np.transpose(b)
    cv = np.absolute(np.matmul(b_t, phi))
    cv_index = np.argsort(-cv).flatten()
    lda = cv_index[:(k + 1)]
    phi_lda = phi[:, lda]
    if is_disp:
        print(np.sort(lda))
    x = least_square_adp(np.matmul(np.transpose(phi_lda), phi_lda),
                         np.matmul(np.transpose(phi_lda), b).reshape(-1, 1))
    r = b - np.matmul(phi_lda, x)
    res = np.linalg.norm(r, 2)
    if res < 10**(-12):
        X = np.zeros(n)
        X[lda] = np.transpose(x)
        supp = [i for i in range(phi.shape[1]) if X[i] != 0]
        return np.array(supp)
    usedlda = np.zeros(n)
    usedlda[lda] = 1
    for _ in range(itermax):
        res_old = res
        cv = np.absolute(np.matmul(np.transpose(r), phi))
        cv_index = np.argsort(-cv).flatten()
        sga = np.union1d(lda, cv_index[:k + 1])
        phi_sga = phi[:, sga]
        x_temp = least_square_adp(np.matmul(np.transpose(phi_sga), phi_sga),
                                  np.matmul(np.transpose(phi_sga), b))
        x_temp_index = np.argsort(-np.absolute(x_temp.flatten())).flatten()
        lda = sga[x_temp_index[:k + 1]]
        phi_lda = phi[:, lda]
        usedlda[lda] = 1
        x = least_square_adp(np.matmul(np.transpose(phi_lda), phi_lda),
                             np.matmul(np.transpose(phi_lda), b))
        r = b - np.matmul(phi_lda, x)
        res = np.linalg.norm(r, 2)
        X = np.zeros(n)
        X[lda] = np.transpose(x)
        if res / res_old >= 1 or res < 10**(-12):
            supp = [i for i in range(phi.shape[1]) if X[i] != 0]
            return np.array(supp)

def compute_discrete_phi(m: int, d: int, p: int) -> np.array:
    """
    This function will compute the discretized 0th - dth order partial derivative of 
    1d test function f(x) = (1-x^2)^p on a localized domain centered around 0 with grid
    size 2m+1.

    Args:
        m (int): 2*m+1 is the local intergration grid size for weak features in one dimension.
        d (int): the highest order of partial derivatives allowed in the dictionary.
        p (int): a smoothness parameter in the test function f(x) = (1-x^2)^p.

    Remark: The script is modified from Matlab code for Paper, "Weak SINDy for Partial Differential
            Equations" by D. A. Messenger and D. M. Bortz.

    Returns:
        np.array: array of shape (d+1, 2*m+1) where i-th (i = 1,...,d, d+1) row represents
                  d^(i-1)(f)/dx^(i-1).
    """
    t = np.arange(m + 1) / m
    t_l = np.zeros((d + 1, m + 1))
    t_r = np.zeros((d + 1, m + 1))
    for j in range(m):
        t_l[:, j] = np.power(
            1 + t[j],
            np.fliplr(np.arange(p - d, p + 1).reshape(1, -1)).flatten())
        t_r[:, j] = np.power(
            1 - t[j],
            np.fliplr(np.arange(p - d, p + 1).reshape(1, -1)).flatten())
    ps = np.ones((d + 1, 1))
    for q in range(d):
        ps[q + 1] = (p - (q + 1) + 1) * ps[q]
    t_l = ps * t_l
    t_r = (np.power(-1, np.arange(d + 1)).reshape(-1, 1) * ps) * t_r
    cfs = np.zeros((d + 1, 2 * m + 1))
    cfs[0, :] = np.block([
        np.fliplr((t_l[0, :] * t_r[0, :]).reshape(1, -1)),
        t_l[0, 1:] * t_r[0, 1:]
    ])
    p = np.fliplr(scipy.linalg.pascal(d + 1))
    for k in range(d):
        binoms = np.diag(p, d - k - 1)
        cfs_temp = np.zeros((1, m + 1))
        for j in range(k + 2):
            cfs_temp = cfs_temp + binoms[j] * t_l[k + 1 - j, :] * t_r[j, :]
        cfs[k + 1, :] = np.block([
            (-1)**(k + 1) * np.fliplr(cfs_temp.reshape(1, -1)),
            cfs_temp[0, 1:].reshape(1, -1)
        ])
    return cfs

def compute_feature_and_scale_matrix_ode(
        m: int, p: int, skip: int, dict_list: np.array, u_hat: np.array,
        dt: np.array) -> Tuple[np.array, np.array]:
    """
    This function computes the feature matrix W and scale matrix S for both lhs and rhs features for ODEs.

    Note: (1) the current version of ode equations refer to 1d ode problems where spatial derivatives do not exits.
          (2) for odes, the scale matrix S is considered to be identical to feature matrix W.
    Args:
        m (int): one-side length of a local integration area.
        p (int): parameter in the test function.
        skip (int): skipping steps when downsampling a feature matrix
        dict_list(np.ndarray): array of shape (L, n + 2). Here each row represents a feature,
                               column 1 - column n represents the degree of monomial in each feature
                               column n + 1 has 0s.
                               column n + 2 represents the order of partial derivatives along temporal
                               domain.  
        u_hat (np.array): array of shape (n,) (given noisy data).
        dt (np.array): delta t (temporal increase) of given data .

    Returns:
        np.array, np.array: feature matrix W and sclae matrix S (including lhs)
    """
    n_t = u_hat[0].shape[1]
    subsampling_idx = np.arange(0, n_t - 2 * m, skip)
    phi_xs = compute_discrete_phi(m, 1, p)
    mm, nn = phi_xs.shape[0], phi_xs.shape[1]
    temp = np.block([
        np.zeros((mm, n_t - nn)),
        np.power(m * dt, -np.arange(mm)).reshape(-1, 1) * phi_xs / nn
    ])
    fft_phis = np.fft.fft(temp)
    w_b_large = compute_w_ode(dict_list, u_hat, fft_phis, subsampling_idx)
    s_b_large = w_b_large.copy()
    return w_b_large, s_b_large

def compute_feature_and_scale_matrix_pde(
        m_x: int, m_t: int, p_x: int, p_t: int, skip_x: int, skip_t: int,
        dict_list: np.array, u_hat: np.array, dx: np.array, dt: np.array,
        max_dx: int) -> Tuple[np.array, np.array]:
    """
    This function computes the feature matrix W and scale matrix S (including lhs and rhs) for given data.

    Args:
        m_x (int): 1-side size of integrating region in spatial domain.
        m_t (int): 1-side size of integrating region in temporal domain.
        p_x (int): parameter in test function in terms of x.
        p_t (int): parameter in test function in terms of t.
        skip_x (int): # skipping steps in spatial domain when downsampling feature matrix
        skip_t (int): # skipping steps in temporal domain when downsampling feature matrix
        dict_list(np.ndarray): array of shape (L, n + dim_x + 1). Here each row represents a feature.
                               Column 1 - column n represents the degree of monomial for each variable
                               column n+1 - column n+dim_x represents the order of partial derivatives 
                               along each spatial domain,
                               and column n+dim_x+1 represents the order of partial derivatives along temporal
                               domain.  
        u_hat (np.array): given data of shape (n,).
        dx (np.array): delta x (spatial increase) of given data.
        dt (np.array): delta t (temporal increase) of given data.
        max_dx (int): maximum total order of partial derivatives.

    Returns:
        Tuple[np.array, np.array]: feature matrix W and scale matrix S (including lhs).
    """
    dims = u_hat[0].shape
    dim_x = len(dims) - 1
    subsampling_idx = []
    shrink_size = np.ones((1, dim_x + 1))
    shrink_size = np.block(
        [np.ones((1, dim_x)) * m_x * 2,
         np.array([[m_t * 2]])]).flatten()
    ss = np.block([np.ones((1, dim_x)) * skip_x,
                   np.array([[skip_t]])]).flatten()
    for j in range(len(dims)):
        subsampling_idx.append(np.arange(0, dims[j] - shrink_size[j], ss[j]))
    phi_xs, phi_ts = compute_test_funs(m_x, m_t, p_x, p_t, max_dx)
    fft_phis = compute_fft_test_fun(dims, phi_xs, phi_ts, m_x, m_t, dx, dt)
    w_b_large = compute_w(dict_list, u_hat, fft_phis, subsampling_idx)
    s_b_large = compute_s_v2(w_b_large, dict_list, u_hat, fft_phis,
                             subsampling_idx)
    return w_b_large, s_b_large

def compute_w_ode(dict_list: np.array, u_hat: np.array, fft_phis: np.array,
                  subsampling_idx: np.array) -> np.array:
    """
    Args:
        dict_list(np.ndarray): array of shape (L, n + 2). Here each row represents a feature,
                               column 1 - column n represents the degree of monomial in each feature
                               column n + 1 has 0s.
                               column n + 2 represents the order of partial derivatives along temporal
                               domain.  
        u_hat (np.array): given noisy data with shape (n,).
        fft_phis (np.array): array of shape (2, mathbbNx). This is the fast fourier transform of the test function and
                             it's first order derivative. 
        subsampling_idx (np.array): array of shape (Nx , ) which stores the index of subsampled feature matrix.

    Returns:
        np.array: array of shape (N, L + n)
    """
    print('The number Nx  in the subsampled feature matrix is',
          subsampling_idx.shape[0], '.')
    print('Start building feature matrix W:')
    L = dict_list.shape[0]
    w = np.zeros((subsampling_idx.shape[0], L))
    n = u_hat.shape[0]
    ind = 0
    while ind < L:
        tags = dict_list[ind, :n]
        fcn = np.power(u_hat[0], tags[0])
        for k in range(1, n):
            fcn = fcn * np.power(u_hat[k], tags[k])
        while np.all(dict_list[ind, :n] == tags):
            temp = fft_phis
            test_conv_cell = temp[int(dict_list[ind, n + 1]), :].reshape(-1, 1)
            fcn_conv = conv_fft_v2(fcn, test_conv_cell, subsampling_idx)
            w[:, ind] = fcn_conv.flatten()
            sys.stdout.write('\r')
            sys.stdout.write("[{:{}}] {:.1f}%".format("=" * ind, L - 1,
                                                      (100 / (L - 1) * ind)))
            sys.stdout.flush()
            ind += 1
            if ind >= L:
                break
    return w

def compute_w(dict_list: np.array, u_hat: np.array, fft_phis: np.array,
              subsampling_idx: np.array):
    """
    This function computes the feature matrix W (including lhs) for given data to identify pde equations.

    Args:
        dict_list(np.ndarray): array of shape (L, n + dim_x + 1). Here each row represents a feature,
                               column 1 - column n represents the degree of monomial for each variable,
                               column n+1 - column n+dim_x represents the order of partial derivatives 
                               along each spatial domain,
                               and column n+dim_x+1 represents the order of partial derivatives along temporal
                               domain.
        u_hat (np.array): array of shape (n,) (given data).
        fft_phis (np.array): array of shape (max_dx+1, mathbbNx). This is the fft of a test functions and it's derivatives. 
        subsampling_idx (np.array): array of shape (Nx , ) which stores the index of subsampled feature matrix.

    Returns:
        np.array: (feature matrix including lhs) array of shape (N, L + n). 
    """
    print('Start building feature matrix W:')
    n = u_hat.shape[0]
    dim = len(u_hat[0].shape)
    L = dict_list.shape[0]
    if dim == 2:
        W = np.zeros((len(subsampling_idx[0]) * len(subsampling_idx[1]), L))
    elif dim == 3:
        W = np.zeros((len(subsampling_idx[0])**2 * len(subsampling_idx[2]), L))
    ind = 0
    while ind < L:
        tags = dict_list[ind, :n]
        fcn = np.power(u_hat[0], tags[0])
        for k in range(1, n):
            fcn = fcn * np.power(u_hat[k], tags[k])
        while np.all(dict_list[ind, :n] == tags):
            test_conv_cell = []
            # test_conf_cell store the fft of a test function(or it's derivatives) in all spatial
            # and temporal dimension
            for k in range(dim):
                temp = fft_phis[k]
                test_conv_cell.append(temp[int(dict_list[ind,
                                                         n + k]), :].reshape(
                                                             -1, 1))
            fcn_conv = conv_fft(fcn, test_conv_cell, subsampling_idx)
            if dim == 2:
                W[:, ind] = np.transpose(fcn_conv, (1, 0)).flatten()
            elif dim == 3:
                W[:, ind] = np.transpose(fcn_conv, (2, 1, 0)).flatten()
            if dim == 3 or L > 100:
                print(" Progress {:2.1%}".format((ind + 1) / L), end="\r")
            else:
                sys.stdout.write('\r')
                sys.stdout.write("[{:{}}] {:.1f}%".format(
                    "=" * ind, L - 1, (100 / (L - 1) * ind)))
                sys.stdout.flush()
            ind += 1
            if ind >= L:
                break
    return W

def compute_base_of_s(u_hat: np.array, tags: np.array) -> np.array:
    """
    This function computes the base of the scales (for error normalization) for a given features. 
    The definition of base follows the idea of leading coefficient of epsilon in WeakIdent paper.

    Args:
        u_hat (np.array): array of shape (n,) (given data).
        tags (np.array): the tags of betas (monomial order) of a given feature.

    Returns:
        np.array: base of scale for a given feature.
    """
    n = u_hat.shape[0]
    if n == 1:
        fcn = tags[0] * np.power(u_hat[0], tags[0] - 1)
    elif n >= 2:
        fcn = np.zeros_like(u_hat[0])
        for ii in range(n):
            if tags[ii] == 0:
                temp = 0
            else:
                if tags[ii] == 1:
                    temp = 1
                else:
                    temp = tags[ii] * np.power(u_hat[ii], tags[ii] - 1)
                for k in range(n):
                    if k != ii and tags[k] > 0:
                        temp = temp * np.power(u_hat[k], tags[k])
            fcn = fcn + temp
    return fcn

def compute_s_v2(w: np.array, dict_list: np.array, u_hat: np.array,
                 fft_phis: np.array, subsampling_idx: np.array) -> np.array:
    """
    This function computes the scale matrix S (including lhs) for given data.

    Args:
        w (np.array): (feature matrix array) of shape (N, L + n) where L is the number of rhs features and n 
                      is the number of variables
        dict_list(np.ndarray): array of shape (L, n + dim_x + 1). Here each row represents a feature.
                               Column 1 - column n represents the degree of monomial for each variable, 
                               column n+1 - column n+dim_x represents the order of partial derivatives 
                               along each spatial domain, 
                               and column n+dim_x+1 represents the order of partial derivatives along temporal
                               domain. 
        u_hat (np.array): array of shape (n,) (given noisy data).
        fft_phis (np.array): array of shape (max_dx+1, mathbbNx). This is the fast fourier transform of a test function
                             and it's derivatives. 
        subsampling_idx (np.array): array of shape (Nx , ) which stores the index of subsampled feature matrix.

    Returns:
        np.array: (scale matrix) array of shape (N, L + n). 
    """
    print(" ")
    print('Start building scale matrix S:')
    n = u_hat.shape[0]
    dim = len(u_hat[0].shape)
    l = dict_list.shape[0]
    if dim == 2:
        s = np.ones((len(subsampling_idx[0]) * len(subsampling_idx[1]), l))
    elif dim == 3:
        s = np.ones((len(subsampling_idx[0])**2 * len(subsampling_idx[2]), l))
    ind = 1
    while ind < l:
        tags = dict_list[ind, :n]
        beta_u_plus_beta_v = np.sum(tags)
        if beta_u_plus_beta_v > 1:
            fcn = compute_base_of_s(u_hat, tags)
        while np.all(dict_list[ind, :n] == tags):
            if beta_u_plus_beta_v == 1:
                s[:, ind] = w[:, ind]
            else:
                test_conv_cell = []
                if np.sum(dict_list[ind, n:n + dim]) == 0:
                    for k in range(dim):
                        temp = fft_phis[k]
                        test_conv_cell.append(temp[0, :].reshape(-1, 1))
                else:
                    for k in range(dim):
                        temp = fft_phis[k]
                        test_conv_cell.append(
                            temp[int(dict_list[ind, n + k]), :].reshape(-1, 1))
                fcn_conv = conv_fft(fcn, test_conv_cell, subsampling_idx)
                if dim == 2:
                    s[:, ind] = np.transpose(fcn_conv, (1, 0)).flatten()
                elif dim == 3:
                    s[:, ind] = np.transpose(fcn_conv, (2, 1, 0)).flatten()
            if dim == 3 or l > 100:
                print(" Progress {:2.1%}".format((ind + 1) / l), end="\r")
            else:
                sys.stdout.write('\r')
                sys.stdout.write("[{:{}}] {:.1f}%".format(
                    "=" * ind, l - 1, (100 / (l - 1) * ind)))
                sys.stdout.flush()
            ind += 1
            if ind >= l:
                break
    print(" ")
    return s

def conv_fft(fu: np.array, fft_phis: np.array, subsampling_idx: np.array):
    """
    This function takes f(u) and fft(d^i(phi)) as input and returns the 
    multi-dimennional convolution u * d^i(phi) as as a weak feature.
    Note: this function is called only when the equation to be identified is not a
          1d ode systems. See conv_fft for multi-dimensional
          convlution for the case of PDEs or multi-dimensional ODEs. 
          See conv_fft_v2() for the case of 1d-ode equations.
    Args:
        fu (np.array): the base of feature (the product of monomials).
        fft_phi (np.array): This is the fft of test functions (derivative) fft(d^i(phi)) in each spatial
                            and temporal domain.
        subsampling_idx (np.array): stores the index of subsampled feature matrix in each spatial and 
                                    and temporal domain.

    Returns:
        np.array: convolution u * d^i(phi)
    """
    Ns = fu.shape
    dim = len(Ns)
    X = np.copy(fu)
    for k in range(dim):
        col_ifft = fft_phis[k]
        shift = circshift([i for i in range(dim)], 1 - (k + 1))
        shift_back = circshift([i for i in range(dim)], -1 + (k + 1))
        Y = np.fft.fft(X.transpose(tuple(shift)), axis=0)
        if dim == 3:
            col_ifft = col_ifft.reshape(-1, 1, 1)
        X = np.fft.ifft(col_ifft * Y, axis=0)
        inds = subsampling_idx[k].astype(int)
        if dim == 2:
            X = X[inds, :]
        elif dim == 3:
            X = X[inds, :, :]
        X = X.transpose(tuple(shift_back))
    return np.real(X)

def conv_fft_v2(fu: np.array, fft_phi: np.array,
                subsampling_idx: np.array) -> np.array:
    """
    This function takes f(u) and fft(d^i(phi)) as input and returns the convolution u * d^i(phi) as
    as a result of weak feature.
    Note: this function is called only when the sytem to be identified is 1d ode. See conv_fft for multi-dimensional
          convlution for the case of PDEs or multi-dimensional ODEs. 

    Args:
        fu (np.array): the base of a feature (the product of monomials).
        fft_phi (np.array): This is the fft of test functions (derivative) fft(d^i(phi)).
        subsampling_idx (np.array): array of shape (Nx, ) which stores the index of subsampled feature matrix.

    Returns:
        np.array: convolution u * d^i(phi)
    """
    ifft_phi = fft_phi.flatten()
    temp = np.fft.fft(fu).flatten()
    fu = np.fft.ifft(ifft_phi * temp, axis=0)
    inds = subsampling_idx.astype(int)
    fu = fu[inds]
    return np.real(fu)

def compute_test_funs(m_x: int, m_t: int, p_x: int, p_t: int,
                      max_dx: np.array) -> Tuple[np.array, np.array]:
    """
    This function computes the discretized test function for both spatial dimension and temporal 
    dimension.

    Args:
        m_x (int): 1-side size of integrating region in spatial domain.
        m_t (int): 1-side size of integrating region in temporal domain.
        p_x (int): parameter in test function in terms of x.
        p_t (int): parameter in test function in terms of t.
        max_dx (int): maximum total order of partial derivatives.

    Returns:
        [np.array, np.array] : phi(x) = (1-x^2)^p_x and phi(t) (1-t^2)^p_t.
    """
    phi_xs = compute_discrete_phi(m_x, max_dx, p_x)
    phi_ts = compute_discrete_phi(m_t, 1, p_t)
    return phi_xs, phi_ts

def compute_fft_test_fun(dims: tuple, phi_xs: np.array, phi_ts: np.array,
                         m_x: int, m_t: int, dx: np.array,
                         dt: np.array) -> list:
    """
    This function computes fast fourier transform of test function phi and it's partial derivatives w.r.t x and t.

    Args:
        dims (tuple): (N_x, N_t) or (N_x, N_y, N_t).
        phi_xs (np.array): array of shape (max_dx +1, 2m_x + 1), fourier transform phi and phi^(i)(x) for i = 0,1,...,max_dx.
        phi_ts (np.array): array of shape (2, 2m_x + 1), fourier transform phi and phi^(i)(t) for i = 0,1,...,max_dt.
        m_x (int): 1-side size of integrating region in spatial domain.
        m_t (int): 1-side size of integrating region in temporal domain.
        dx (np.array): delta x (spatial increase) of given data.
        dt (np.array): delta t (temporal increase) of given data.

    Returns:
        list: [np.array, np.array] where each element is fft(phi_xs) and fft(phi_ts).
    """
    dimxandt = len(dims)
    fft_phis = []
    mm, nn = phi_xs.shape[0], phi_xs.shape[1]
    for k in range(dimxandt - 1):
        temp = np.block([
            np.zeros((mm, dims[k] - nn)),
            np.power(m_x * dx, -np.arange(mm)).reshape(-1, 1) * phi_xs / nn
        ])
        fft_phis.append(np.fft.fft(temp))
    mm, nn = phi_ts.shape[0], phi_ts.shape[1]
    temp = np.block([
        np.zeros((mm, dims[dimxandt - 1] - nn)),
        np.power(m_t * dt, -np.arange(mm)).reshape(-1, 1) * phi_ts / nn
    ])
    fft_phis.append(np.fft.fft(temp))
    return fft_phis
