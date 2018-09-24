from __future__ import division

import tensorflow as tf

from .ops import dot


################################################################################
# Dimension-wise kernel implementations

def dimwise_mixrbf_kernels(X, bws=(.01, .1, .2, 1, 5, 10, 100), wts=None):
    """Mixture of RBF kernels between each dimension of X.
    If X is shape (n, d), returns shape (n, n, d).
    Kernel is sum_i wt_i exp(- (x - y)^2 / (2 bw^2)).
    If wts is not passed, uses 1 for each alpha.
    """
    shp = X.get_shape()
    shp.assert_has_rank(2)

    bws = tf.constant(bws, dtype=X.dtype)
    wts = tf.constant(1 if wts is None else wts, dtype=X.dtype, shape=bws.shape)

    sqdists = tf.square(tf.expand_dims(X, 0) - tf.expand_dims(X, 1))
    bws_e = tf.expand_dims(tf.expand_dims(tf.expand_dims(bws, 1), 2), 3)
    wts_e = tf.expand_dims(tf.expand_dims(tf.expand_dims(wts, 1), 2), 3)

    parts = tf.exp(tf.expand_dims(sqdists, 0) / (-2 * bws_e ** 2))
    return tf.reduce_sum(wts_e * parts, 0)


def dimwise_mixrq_kernels(X, alphas=(.2, .5, 1, 2, 5), wts=None):
    """Mixture of RQ kernels between each dimension of X.
    If X is shape (n, d), returns shape (n, n, d).
    Kernel is sum_i wt_i (1 + (x - y)^2 / (2 alpha_i))^{-alpha_i}.
    If wts is not passed, uses 1/n_alphas for each alpha.
    """
    shp = X.get_shape()
    shp.assert_has_rank(2)

    alphas = tf.constant(alphas, dtype=X.dtype)
    wts = tf.constant(1 / len(alphas) if wts is None else wts,
                      dtype=X.dtype, shape=alphas.shape)

    sqdists = tf.square(tf.expand_dims(X, 0) - tf.expand_dims(X, 1))
    alphas_e = tf.expand_dims(tf.expand_dims(tf.expand_dims(alphas, 1), 2), 3)

    logs = tf.log1p(tf.expand_dims(sqdists, 0) / (2 * alphas_e))
    return tf.einsum('w,wijd->ijd', wts, tf.exp(-alphas_e * logs))


################################################################################
# HSIC estimators

def dhsic(kernels, logspace=False):
    """(Biased) estimator of total independence.
    kernels should be shape (n, n, d) to test for total
    independence among d variables with n paired samples.
    """
    # formula from section 4.4 of
    # https://papers.nips.cc/paper/4893-a-kernel-test-for-three-variable-interactions.pdf
    shp = kernels.get_shape()
    shp.assert_has_rank(3)
    shp[0].assert_is_compatible_with(shp[1])

    shp = tf.shape(kernels)
    n = tf.cast(shp[0], kernels.dtype)
    d = tf.cast(shp[2], kernels.dtype)

    # t1: 1/n^2      sum_a  sum_b  prod_i K_ab^i
    # t2: -2/n^(d+1) sum_a  prod_i sum_b  K_ab^i
    # t3: 1/n^(2d)   prod_i sum_a  sum_b  K_ab^i

    if not logspace:
        sum_b = tf.reduce_sum(kernels, axis=1)

        t1 = tf.reduce_mean(tf.reduce_prod(kernels, axis=2))
        t2 = tf.reduce_sum(tf.reduce_prod(sum_b, axis=1)) * (-2 / (n ** (d+1)))
        t3 = tf.reduce_prod(tf.reduce_sum(sum_b, dim=0)) / (n ** (2 * d))
        return t1 + t2 + t3
    else:
        log_n = tf.log(n)
        log_2 = tf.log(tf.constant(2, dtype=kernels.dtype))
        log_kernels = tf.log(kernels)
        log_sum_b = tf.reduce_logsumexp(log_kernels, axis=1)

        l1 = tf.reduce_logsumexp(tf.reduce_sum(log_kernels, axis=2)) - 2*log_n
        l2 = tf.reduce_logsumexp(tf.reduce_sum(log_sum_b, axis=1)) + (
            log_2 - (d+1) * log_n)
        l3 = tf.reduce_sum(tf.reduce_logsumexp(log_sum_b, axis=0)) - 2*d*log_n

        # dhsic = exp(l1) - exp(l2) + exp(l3)
        #    = exp(a) (exp(l1 - a) - exp(l2 - a) + exp(l3 - a))  for any a
        a = tf.maximum(tf.maximum(l1, l2), l3)
        return tf.exp(a) * (tf.exp(l1 - a) - tf.exp(l2 - a) + tf.exp(l3 - a))


def center_kernel(K):
    """Center a kernel matrix of shape (n, n, ...).

    If there are trailing dimensions, assumed to be a list of kernels and
    centers each one separately.
    """
    # (I - 1/n 1 1^T) K (I - 1/n 1 1^T)
    #  = K - 1/n 1 1^T K - 1/n K 1 1^T + 1/n^2 1 1^T K 1 1^T
    row_means = tf.reduce_mean(K, axis=0, keepdims=True)
    col_means = tf.transpose(row_means)
    grand_mean = tf.reduce_mean(row_means, axis=1, keepdims=True)
    return K - row_means - col_means + grand_mean


def hsic(kernel_x, kernel_y, biased=False, center_x=False):
    """Independence estimator.

    By default, uses the unbiased estimator.

    If biased=True, center_x determines which kernel will be centered; might
    result in slightly simpler gradients for the kernel which is not centered.
    Ignored if biased=False.
    """
    shp = kernel_x.get_shape()
    shp.assert_has_rank(2)
    shp[0].assert_is_compatible_with(shp[1])
    shp.assert_is_compatible_with(kernel_y.get_shape())

    n = tf.cast(tf.shape(kernel_x)[0], kernel_x.dtype)

    if biased:
        if center_x:
            kernel_x = center_kernel(kernel_x)
        else:
            kernel_y = center_kernel(kernel_y)
        return tf.einsum('ij,ij->', kernel_x, kernel_y) / (n - 1)**2

    # HSIC_1 from http://www.jmlr.org/papers/volume13/song12a/song12a.pdf
    z = tf.zeros(tf.shape(kernel_x)[0], dtype=kernel_x.dtype)
    Kt = tf.matrix_set_diag(kernel_x, z)
    Lt = tf.matrix_set_diag(kernel_y, z)

    Kt_sums = tf.reduce_sum(Kt, axis=0)
    Lt_sums = tf.reduce_sum(Lt, axis=0)

    return 1 / (n * (n - 3)) * (
        tf.einsum('ij,ij->', Kt, Lt)
        + tf.reduce_sum(Kt_sums) * tf.reduce_sum(Lt_sums) / ((n - 1) * (n - 2))
        - 2 / (n - 2) * dot(Kt_sums, Lt_sums))


def average_pairwise_hsic(kernels, include_self=False, biased=True):
    """Sum of estimators of pairwise independence.
    kernels should be shape (n, n, d) when testing
    among d variables with n paired samples.

    If include_self = False, don't include terms comparing a variable to itself.
    """
    shp = kernels.get_shape()
    shp.assert_has_rank(3)
    shp[0].assert_is_compatible_with(shp[1])

    n = tf.cast(tf.shape(kernels)[0], kernels.dtype)
    d = tf.cast(tf.shape(kernels)[2], kernels.dtype)

    if biased:
        cent = center_kernel(kernels)

        # HSIC on dims (i, j) is  1/n^2 sum_{a, b} K[a, b, i] K[a, b, j]
        # sum over dimensions is 1/n^2 sum_{i, j, a, b} K[a, b, i] K[a, b, j]
        s = tf.einsum('abi,abj->', cent, cent)
        if include_self:
            return s / n**2 / d**2
        else:
            self_terms = tf.einsum('abi,abi->', cent, cent)
            return (s - self_terms) / n**2 / (d * (d - 1))

    raise NotImplementedError("haven't implemented unbiased here yet")
