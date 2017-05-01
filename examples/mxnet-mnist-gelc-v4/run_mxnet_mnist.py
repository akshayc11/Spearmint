import subprocess
import os
import numpy as np

num_neurons = [32*x for x in range(1,101)]
lr_factors = [x * 0.1 for x in range(5,10)]
max_epochs = 20
lr_step_epochs = [','.join([str(step_size + step_size*x) for x in range(max_epochs/step_size - 1)]) for step_size in [4,5,10]]
train_cmd = 'python train_mnist_custom.py --gpus 1 --num-layers {} --num-hidden {} --lr {} --lr-factor {} --lr-step-epoch {} --kill-time {}'
out_dir='log'
# mxnet_dir='/home/akshayc/mxnet/example/image-classification'

def get_best_validation_accuracy(out_file_name):
    out_file = open(out_file_name, 'r')
    out_lines = out_file.readlines()
    validation_accs = [line for line in out_lines if 'Validation-accuracy' in line]
    v_accs = [float(line.strip().split('=')[1]) for line in validation_accs]
    best_acc = max(v_accs)
    out_file.close()
    return best_acc

def run(num_layers, num_hidden, lr, lr_factor, lr_step_epoch, out_file_name, k_time, parameters):
    out_file = open(out_file_name, 'w')
    out_file.write(parameters + '\n')
    run_cmd = train_cmd.format(num_layers, num_hidden, lr, lr_factor, lr_step_epoch, k_time)
    subprocess.call(run_cmd, shell=True, stdout=out_file, stderr=out_file)
    out_file.close()
    
def main(job_id, params):
    # os.chdir(mxnet_dir)
    num_layers = int(params['num_hidden_layers'])
    num_hidden = 32 * int(params['num_neurons'])
    lr = float(params['lr'])
    lr_factor = lr_factors[int(params['lr_factor_idx'])]
    lr_step_epoch = lr_step_epochs[int(params['lr_step_epoch_idx'])]
    parameters='num_layers:{} num_hidden:{} lr:{} lr_factor:{} lr_step_epoch:{}'.format(num_layers, num_hidden, lr, lr_factor, lr_step_epoch)
    print parameters
    if job_id < 5:
        k_time = 1
    else:
        k_time = 30
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file_name='{}/data_{}.log'.format(out_dir, job_id)
    run(num_layers, num_hidden, lr, lr_factor, lr_step_epoch, out_file_name, k_time, parameters)
    best_acc = get_best_validation_accuracy(out_file_name)
    err = 1.0 - best_acc
    return err

def get_validation_accuracies(job_id, params):
    log_file = '{}/data_{}.log'.format(out_dir, job_id)
    f = open(log_file, 'r')
    validation_accs = [line for line in f.readlines() if 'Validation-accuracy' in line]
    v_accs = []
    max_v_acc = 0.0
    for line in validation_accs:
        v_acc = float(line.strip().split('=')[1])
        if max_v_acc < v_acc:
            max_v_acc = v_acc
        v_accs.append(max_v_acc)
    print 'v_accs size:', len(v_accs)
    return np.array(v_accs)
