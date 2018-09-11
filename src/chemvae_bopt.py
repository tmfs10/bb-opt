
import torch
import torch.nn as nn
import matlab.engine
import acquisition as ack

def gpopt_pes(x_min, x_max, predictor_model):
    eng = matlab.engine.start_matlab()

    f

    eng.quit()
