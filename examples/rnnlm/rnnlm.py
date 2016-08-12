import os
import numpy as np
import math

rtf_threshold=0.1
log_rtf_threshold = math.log(rtf_threshold)

def train_rnnlm(num_n_best, hidden_layer_size, num_hidden_layers):
    cdir = os.getcwd()

    # TODO: Bing: Add train and scoring script here
    # Assumption: The above section has filled in the values for
    # wer and rtf. wer SHOULD NOT be normalized at this point (% value)
    
    
    os.chdir(cdir)
    log_rtf = math.log(rtf)
    thresholded_rtf = log_rtf_threshold - log_rtf
    
    wer = wer / 100
    
    print("Params_Outs: {} {} {} {} {}".format(num_n_best,
                                               hidden_layer_size,
                                               num_hidden_layers,
                                               wer,
                                               rtf))
    return {
        "rescore_wer" : wer,
        "thresholded_rtf": thresholded_rtf
    }


ns = [50,60,70,80,90,100,110,120,130,140,150]
hidden_layer_sizes = [64, 128, 192, 256, 320, 384, 448, 512]
num_hidden_layers_list = [1,2,3]

def main(job_id, params):
    num_n_best = ns[int(params['num_n_best'])]
    hidden_layer_size = hidden_layer_sizes[int(params['hidden_layer_sizes'])]
    num_hidden_layers = num_hidden_layers_list[int(params['num_hidden_layers'])]
    return train_rnnlm(num_n_best, hidden_layer_size, num_hidden_layers)
