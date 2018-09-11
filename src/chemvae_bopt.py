
import torch
import torch.nn as nn
import matlab.engine
import acquisition as ack

def gpopt_pes(n_init_samples, n_opt_samples, n_features, x_min, x_max, predictor_model, num_batches):
    eng = matlab.engine.start_matlab()

    x_samples, y_samples, l, sigma, sigma0 = ack.init_pes(eng, predictor_model, n_init_samples, n_opt_samples, x_min, x_max)

    for i in range(num_batches):
        x_samples, y_samples, guesses, l, sigma, sigma0 = ack.pes(eng, obj, n_opt_samples, n_features, x_min, x_max, x_samples, guesses, y_samples, l, sigma, sigma0)

    eng.quit()

    return np.asarray(x_samples), np.asarray(y_samples), np.asarray(guesses)
