import sys
import os
import subprocess
import math
import time
frame_size_list = [5,10,15,20,25,30,35,40,45,50]
frame_shift_list = [5,10,15,20,25,30,35,40]
num_mel_bins_list = [5,10,15,20,25,30,35,40,45,50]
hidden_layer_size_list = [512 + (x*256) for x in range(10)]
min_actives = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
max_actives = [2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000]
prune_intervals = [10, 15, 20, 25, 30, 35, 40, 45, 50]

rtf_threshold = 0.2
log_rtf_threshold = math.log(rtf_threshold)
s5_dir="/data-local/akshayc/Workspace/Software/kaldi/egs/wsj/s5"
a3_dir="/data-local/akshayc/Workspace/Software/speech3/egs/wsj/a3"
cmd="./train_and_decode_all.sh {} {} {} {} {} {} {} {} {} {} {} {} {} {}"

dev_dir = os.path.join(a3_dir, "devices")

def get_device():
    fs = os.listdir(dev_dir)
    while len(fs) == 0:
        time.sleep(10)
        fs = os.listdir(dev_dir)
    f = fs[0]
    subprocess.call("rm {}/{}".format(dev_dir, f), shell=True)
    return f

def release_device(f):
    subprocess.call("touch {}/{}".format(dev_dir, f), shell=True)


def get_wer_rtf(frame_size,
                frame_shift,
                num_mel_bins,
                context,
                num_layers,
                hidden_layer_size,
                ac,
                b,
                lb,
                mina,
                maxa,
                pr):
    feat_prefix='{}_{}_{}'.format(frame_size,
                                  frame_shift,
                                  num_mel_bins)
    dnn_fbank='dnn_fbank_{}_lc{}_rc{}_l{}_h{}'.format(feat_prefix,
                                                      context,
                                                      context,
                                                      num_layers,
                                                      hidden_layer_size)
    decode_dir='decode_ac{}_b{}_lb{}_min{}_max{}_pr{}'.format(ac,
                                                              b,
                                                              lb,
                                                              mina,
                                                              maxa,
                                                              pr)
    out_dir='{}/exp/{}/{}/'.format(a3_dir,
                                   dnn_fbank,
                                   decode_dir)
    with open(os.path.join(out_dir, 'best_wer'),'r') as f:
        res = f.readline()
    print res
    comps = res.split(' ')
    wer = float(comps[1].split(':')[1])
    rtf = float(comps[2].split(':')[1])
    return wer, rtf

def train_and_decode(frame_size,
                     frame_shift,
                     num_mel_bins,
                     context,
                     num_layers,
                     hidden_layer_size,
                     ac, 
                     b,
                     lb,
                     mina, 
                     maxa,
                     pr):
    os.chdir(a3_dir)
    device = 0
    act_cmd = cmd.format(frame_size,
                         frame_shift,
                         num_mel_bins,
                         context,
                         num_layers,
                         hidden_layer_size,
                         rtf_threshold,
                         device,
                         ac,
                         b,
                         lb,
                         mina,
                         maxa,
                         pr)
    subprocess.call(act_cmd, shell=True)
    wer, rtf = get_wer_rtf(frame_size,
                           frame_shift,
                           num_mel_bins,
                           context,
                           num_layers,
                           hidden_layer_size,
                           ac,
                           b,
                           lb,
                           mina,
                           maxa,
                           pr)
    log_rtf = math.log(rtf)
    rtf_constraint = log_rtf_threshold - log_rtf
    # release_device(device)
    return {"wer": wer,
            "rtf_constraint": rtf_constraint}

def main(job_id, params):
    frame_size = frame_size_list[int(params['frame_size'])]
    frame_shift = frame_shift_list[int(params['frame_shift'])]
    num_mel_bins = num_mel_bins_list[int(params['num_mel_bins'])]
    context = int(params['context'])
    num_layers = int(params['num_layers'])
    hidden_layer_size = hidden_layer_size_list[int(params['hidden_layer_size'])]
    ac = float(params['ac'])
    beam = float(params['b'])
    lat_beam = float(params['lb'])
    mina = min_actives[int(params['mina'])]
    maxa = max_actives[int(params['maxa'])]
    pr = prune_intervals[int(params['pr'])]
    return train_and_decode(frame_size, 
                            frame_shift, 
                            num_mel_bins, 
                            context, 
                            num_layers, 
                            hidden_layer_size, 
                            ac, 
                            beam, 
                            lat_beam, 
                            mina, 
                            maxa, 
                            pr)
