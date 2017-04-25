import os
import numpy as np
import math
import subprocess
import time
import sys

data_dir = "/data-local/akshayc/Workspace/Software/speech3/egs/wsj/a3/data_25_20_25"
model_dir = "/data-local/akshayc/Workspace/Software/speech3/egs/wsj/a3/exp/dnn_fbank_25_20_25_lc4_rc4_l3_h1536"
graph_dir = "/data-local/akshayc/Workspace/Software/kaldi/egs/wsj/s5/exp_25_20/tri4b/graph_bd_tgpr"
stable_dir = "/data-local/akshayc/Workspace/Software/speech3/egs/wsj/a3/data_25_20_25"

rtf_threshold = 0.2
logRtf_threshold = math.log(rtf_threshold)
device = 1
base_cmd = "./decode.sh --acoustic-scale {} --beam {} --lattice-beam {} --min-active {} --max-active {} --prune-interval {} --device {} {} {} {} {} {}"

def get_wer_rtf(out_dir):
    with open(os.path.join(out_dir, "s3_wer"),'r') as f:
        wer = float(f.readline().strip())/100.0
    with open(os.path.join(out_dir, "rtf"), 'r') as f:
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
    op  = "Result: wer:{} rtf:{}".format(wer, rtf)
    print op 
    fp = open("{}/best_wer".format(out_dir), 'w')
    fp.write(op)
    fp.close()


import sys

if __name__ == '__main__':
    ac = float(sys.argv[1])
    b = float(sys.argv[2])
    lb = float(sys.argv[3])
    mina = int(sys.argv[4])
    maxa = int(sys.argv[5])
    pr = int(sys.argv[6])    
    decode(ac, b, lb, mina, maxa, pr)
