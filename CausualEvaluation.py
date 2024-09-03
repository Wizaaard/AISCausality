import pandas as pd 
import numpy as np
from sklearn.metrics import mean_squared_error
from maq import get_ipw_scores
import tensorflow as tf

# Copied from Shwab et. al 2019's Source code
# SOURCE: https://github.com/d909b/perfect_match/blob/master/perfect_match/models/pehe_loss.py

# SOURCE: https://github.com/clinicalml/cfrnet, MIT-License
def pdist2(X, Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2*tf.matmul(X, tf.transpose(Y))
    nx = tf.reduce_sum(tf.square(X), 1, keepdims=True)
    ny = tf.reduce_sum(tf.square(Y), 1, keepdims=True)
    D = (C + tf.transpose(ny)) + nx

    return tf.sqrt(D + 1e-8)


# SOURCE: https://github.com/clinicalml/cfrnet, MIT-License
def cf_nn(x, t):
    It = tf.where(tf.equal(t, 1))[:, 0]
    Ic = tf.where(tf.equal(t, 0))[:, 0]

    x_c = tf.gather(x, Ic)
    x_t = tf.gather(x, It)

    D = pdist2(x_c, x_t)

    nn_t = tf.gather(Ic, tf.argmin(D, 0))
    nn_c = tf.gather(It, tf.argmin(D, 1))

    return tf.stop_gradient(nn_t), tf.stop_gradient(nn_c)


# SOURCE: https://github.com/clinicalml/cfrnet, MIT-License
def pehe_nn(yf_p, ycf_p, y, x, t, nn_t=None, nn_c=None):
    if nn_t is None or nn_c is None:
        nn_t, nn_c = cf_nn(x, t)

    It = tf.where(tf.equal(t, 1))[:, 0]
    Ic = tf.where(tf.equal(t, 0))[:, 0]

    ycf_t = 1.0*tf.cast(tf.gather(y, nn_t), tf.float64)
    eff_nn_t = ycf_t - 1.0*tf.cast(tf.gather(y, It), tf.float64)
    eff_pred_t = tf.gather(ycf_p, It) - tf.gather(yf_p, It)

    eff_pred = eff_pred_t
    eff_nn = eff_nn_t

    pehe_nn = tf.sqrt(tf.reduce_mean(tf.square(eff_pred - eff_nn)))
    return pehe_nn


def pehe_loss(y_true, y_pred, t, x, num_treatments):
    total, num_elements = 0, 0.
    for i in range(num_treatments):
        for j in range(num_treatments):
            if j >= i:
                continue

            t1_indices = tf.where(tf.equal(t, i))[:, 0]
            t2_indices = tf.where(tf.equal(t, j))[:, 0]

            these_x = tf.concat([tf.gather(x, t1_indices), tf.gather(x, t2_indices)], axis=0)
            y_pred_these_treatments = tf.concat([tf.gather(y_pred, t1_indices),
                                                 tf.gather(y_pred, t2_indices)], axis=0)
            y_true_these_treatments = tf.concat([tf.gather(y_true, t1_indices),
                                                 tf.gather(y_true, t2_indices)], axis=0)

            these_treatments = tf.concat([tf.ones((tf.shape(t1_indices)[0],), dtype="int32") * i,
                                          tf.ones((tf.shape(t2_indices)[0],), dtype="int32") * j],
                                         axis=0)

            these_y_pred_f = tf.gather(y_pred_these_treatments,
                                       tf.concat([tf.range(tf.shape(y_pred_these_treatments)[0]),
                                                  these_treatments], axis=-1))
            these_y_true_f = y_true_these_treatments

            inverse_treatments = tf.concat([tf.ones((tf.shape(t1_indices)[0],), dtype="int32") * j,
                                            tf.ones((tf.shape(t2_indices)[0],), dtype="int32") * i],
                                           axis=0)

            these_y_pred_cf = tf.gather(y_pred_these_treatments,
                                        tf.concat([tf.range(tf.shape(y_pred_these_treatments)[0]),
                                                   inverse_treatments], axis=-1))

            these_treatments = tf.concat([tf.zeros((tf.shape(t1_indices)[0],), dtype="int32"),
                                          tf.ones((tf.shape(t2_indices)[0],), dtype="int32")],
                                         axis=0)
            total += pehe_nn(these_y_pred_f, these_y_pred_cf, these_y_true_f, these_x, these_treatments)
            num_elements += 1.
    return total / num_elements

# Based on Tran, Burghardt, et. al 2019 Eqs. 27 and 28
# Returns the SQUARE ROOTED version of Eq 28
def mse_of_hte_thru_ipw(y_test, pred_y_test, treatment_flag_test):
  IPW_scores = get_ipw_scores(y_test, treatment_flag_test.astype(int))
  return mean_squared_error(IPW_scores, pred_y_test)