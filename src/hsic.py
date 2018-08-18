import tensorflow as tf

################################################################################
# Dimension-wise kernel implementations.

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
    If wts is not passed, uses 1 for each alpha.
    """
    shp = X.get_shape()
    shp.assert_has_rank(2)

    alphas = tf.constant(alphas, dtype=X.dtype)
    wts = tf.constant(1 if wts is None else wts,
                      dtype=X.dtype, shape=alphas.shape)

    sqdists = tf.square(tf.expand_dims(X, 0) - tf.expand_dims(X, 1))
    alphas_e = tf.expand_dims(tf.expand_dims(tf.expand_dims(alphas, 1), 2), 3)
    wts_e    = tf.expand_dims(tf.expand_dims(tf.expand_dims(wts,    1), 2), 3)

    logs = tf.log1p(tf.expand_dims(sqdists, 0) / (2 * alphas_e))
    return tf.reduce_sum(wts_e * tf.exp(-alphas_e * logs), 0)


################################################################################
# HSIC estimators

def total_hsic(kernels):
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

    sum_b = tf.reduce_sum(kernels, axis=1)
    t1 = tf.reduce_mean(tf.reduce_prod(kernels, axis=2))
    t2 = - 2 / (n ** (d + 1)) * tf.reduce_sum(tf.reduce_prod(sum_b, axis=1))

    # do this third term in logspace:
    sum_ab = tf.reduce_sum(sum_b, axis=0)
    t3 = tf.exp(-2 * d * tf.log(n) + tf.reduce_sum(tf.log(sum_ab)))

    return t1 + t2 + t3


def sum_pairwise_hsic(kernels):
    """Sum of (biased) estimators of pairwise independence.

    kernels should be shape (n, n, d) when testing
    among d variables with n paired samples.
    """
    shp = kernels.get_shape()
    shp.assert_has_rank(3)
    shp[0].assert_is_compatible_with(shp[1])

    n = tf.cast(tf.shape(kernels)[0], kernels.dtype)

    # Centered kernel matrix is given by:
    # (I - 1/n 1 1^T) K (I - 1/n 1 1^T)
    #  = K - 1/n 1 1^T K - 1/n K 1 1^T + 1/n^2 1 1^T K 1 1^T
    row_means = tf.reduce_mean(kernels, axis=0, keepdims=True)
    col_means = tf.reduce_mean(kernels, axis=1, keepdims=True)
    grand_mean = tf.reduce_mean(row_means, axis=1, keepdims=True)
    centered_kernels = kernels - row_means - col_means + grand_mean

    # HSIC on dims (i, j) is  1/n^2 sum_{a, b} K[a, b, i] K[a, b, j]
    # sum over all dimensions is 1/n^2 sum_{i, j, a, b} K[a, b, i] K[a, b, j]
    return tf.einsum('abi,abj->', centered_kernels, centered_kernels) / n**2
