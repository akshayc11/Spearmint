import os
import numpy as np
import math
import subprocess
import time
data_dir = "/data-local/akshayc/Workspace/Software/speech3/egs/wsj/a6/data_25_20_25"
model_dir = "/data-local/akshayc/Workspace/Software/speech3/egs/wsj/a6/exp/dnn_fbank_25_20_25_lc4_rc4_l3_h1536"
graph_dir = "/data-local/akshayc/Workspace/Software/kaldi/egs/wsj/s5/exp_25_20/tri4b/graph_bd_tgpr"
stable_dir = "/data-local/akshayc/Workspace/Software/speech3/egs/wsj/a6/data_25_20_25"

rtf_threshold = 0.2
logRtf_threshold = math.log(rtf_threshold)
device = 1
base_cmd = "./decode-tx1.sh --acoustic-scale {} --beam {} --lattice-beam {} --min-active {} --max-active {} --prune-interval {} --device {} {} {} {} {} {}"

def get_wer_rtf(out_dir):
    with open(os.path.join(out_dir, "s3_wer"),'r') as f:
        wer = float(f.readline().strip())/100.0
    with open(os.path.join(out_dir, "rtf_tx1"), 'r') as f:
        rtf = float(f.readline().strip())
    return wer, rtf

def decode(ac, b,lb, mina, maxa, pr):
    out_dir = os.path.join(model_dir, 
                           'decode_ac{}_b{}_lb{}_min{}_max{}_pr{}'.format(
                               ac,
                               b,
                               lb,
                               mina,
                               maxa,
                               pr))
    cmd = base_cmd.format(ac, b, lb, mina, maxa, pr, device, data_dir, stable_dir, graph_dir, model_dir, out_dir)
    subprocess.call(cmd, shell=True)
    wer, rtf = get_wer_rtf(out_dir)
    print "Result: wer:{} rtf:{}".format(wer, rtf)
    log_rtf = math.log(rtf)
    rtf_constrained = logRtf_threshold - log_rtf
    
    return {"wer": wer,
            "rtf": rtf_constrained}

min_actives = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
max_actives = [2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000]
prune_intervals = [10, 15, 20, 25, 30, 35, 40, 45, 50]

def main(job_id, params):
    ac = float(params['ac'])
    beam = float(params['b'])
    lat_beam = float(params['lb'])
    mina = min_actives[int(params['mina'])]
    maxa = max_actives[int(params['maxa'])]
    pr = prune_intervals[int(params['pr'])]
    print "Params:{} {} {} {} {} {}".format(ac, beam, lat_beam, mina, maxa, pr)
    return decode(ac, beam, lat_beam, mina, maxa, pr)
