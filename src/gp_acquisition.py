
import torch
import matlab


def init_pes(is_cuda, eng, obj, n_init_samples, n_opt_samples, x_min, x_max):
    eng.addpath('pes_matlab/sourceFiles', eng.path())
    x_samples = eng.lhsu(x_min, x_max, n_init_samples, nargout=1) # (n_init_samples, nvars)
    y_samples = []
    for i in range(n_init_samples):
        y_samples += [obj(np.asarray(x_samples[i]))]
    y_samples = matlab.double(y_samples)

    if is_cuda:
        x_samples = eng.gpuArray(x_samples, nargout=1)
        y_samples = eng.gpuArray(y_samples, nargout=1)

    l, sigma, sigma0 = eng.sampleHypers(x_samples, y_samples, n_opt_samples, nargout=3)

    return x_samples, y_samples, l, sigma, sigma0

def init_pes(is_cuda, eng, obj, x_init_samples, n_opt_samples):
    eng.addpath('pes_matlab/sourceFiles', eng.path())
    x_samples = matlab.double(x_init_samples)
    y_samples = []
    for i in range(n_init_samples):
        y_samples += [obj(np.asarray(x_samples[i]))]
    y_samples = matlab.double(y_samples)

    if is_cuda:
        x_samples = eng.gpuArray(x_samples, nargout=1)
        y_samples = eng.gpuArray(y_samples, nargout=1)

    l, sigma, sigma0 = eng.sampleHypers(x_samples, y_samples, n_opt_samples, nargout=3)

    return x_samples, y_samples, l, sigma, sigma0

def pes(eng, obj, n_opt_samples, n_features, x_min, x_max, x_samples, guesses, y_samples, l, sigma, sigma0):
    # sample from global minimum
    m, hessian = eng.sampleMinimum(n_opt_samples, x_samples, y_samples, l, sigma, sigma0, x_min, x_max, n_features, nargout=2)

    # call the ep method
    ret = eng.initializeEPapproximation(x_samples, y_samples, m, l, sigma, sigma0, hessians, nargout=1)

    # optimize cost function globally

    optimum = eng.globalOptimizationOneArgumentWrapper(x_min, x_max, guesses, ret)

    x_samples = eng.cat(1, x_samples, optimum, nargout=1)
    y_samples = eng.cat(1, x_samples, obj(np.asarray(optimum)), nargout=1)
    l, sigma, sigma0 = eng.sampleHypers(x_samples, y_samples, n_opt_samples, nargout=3)
    
    KernelMatrixInv = []
    for j in range(n_opt_samples):
        l_tr = eng.transpose(l[j, :], nargout=1)
        KernelMatrix = eng.computeKmm(x_samples, l_tr, sigma[j], sigma0[j], nargout=1)
        KernelMatrixInv += [eng.chol2invchol(KernelMatrix, nargout=1)]

    optimum = eng.globalOptimizationWrapper(x_min, x_max, guesses, x_samples, y_samples, KernelMatrixInv, l, sigma, nargout=1)

    guesses = eng.cat(1, guesses, optimum, nargout=1)

    return x_samples, y_samples, guesses, l, sigma, sigma0

def init_mes(is_cuda, eng, obj, x_min, x_max, init_x, init_y):
    eng.addpath('mes_matlab', eng.path())

    xx = init_x
    yy = init_y
    guesses = xx
    guessvals = yy

    options = {
            'savefilenm': matlab.double(),
            'bo_method': 'Add-MES-G',
            'n_max_samples': 5,
            'n_hyper_samples': 10,
            'n_features': 1000
            }

    n_hyper_samples = options['n_hyper_samples']
    l, sigma, sigma0 = eng.sampleHypers(xx, yy, n_hyper_samples, options, nargout=3)
    KernelMatrixInv = []
    for i in range(n_hyper_samples):
        l_tr = eng.transpose(l[j, :], nargout=1)
        KernelMatrix = eng.computeKmm(xx, l_tr, sigma[j], sigma0[j], nargout=1);
        KernelMatrixInv += [eng.chol2invchol(KernelMatrix, nargout=1)]

    return xx, yy, guesses, guessvals, options, KernelMatrixInv, l, sigma, sigma0

def mes(args, eng, obj, options, xx, yy, KernelMatrixInv, guesses, l, sigma, sigma0, x_min, x_max):
    n_hyper_samples = options['n_hyper_samples']
    n_max_samples = options['n_max_samples']
    n_features = options['n_features']

    optimum = eng.mesr_choose(n_hyper_samples, n_max_samples, xx, yy, KernelMatrixInv, guesses, sigma0, sigma, l, x_min, x_max, n_features, nargout=1)

    xx = eng.cat(1, xx, optimum)
    yy = eng.cat(1, xx, obj(np.asarray(optimum)))

    l, sigma, sigma0 = eng.sampleHypers(xx, yy, n_hyper_samples, options, nargout=3)

    KernelMatrixInv = []
    for i in range(n_hyper_samples):
        l_tr = eng.transpose(l[j, :], nargout=1)
        KernelMatrix = eng.computeKmm(xx, l_tr, sigma[j], sigma0[j], nargout=1);
        KernelMatrixInv += [eng.chol2invchol(KernelMatrix, nargout=1)]

    optimum = eng.globalOptimizationWrapper(xx, yy, KernelMatrixInv, l, sigma, xmin, xmax, guesses)

    guesses = eng.cat(1, guesses, optimum, nargout=1)
    guessvals = eng.cat(1, guessvals, obj(np.asarray(optimum)), nargout=1)

    return xx, yy, guesses, guessvals, KernelMatrixInv, l, sigma, sigma0
