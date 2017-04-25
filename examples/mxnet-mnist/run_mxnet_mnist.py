import subprocess
import os

num_neurons = [32*x for x in range(1,101)]
lr_factors = [x * 0.1 for x in range(5,10)]
max_epochs = 20
lr_step_epochs = [','.join([str(step_size + step_size*x) for x in range(max_epochs/step_size - 1)]) for step_size in [4,5,10]]
train_cmd = 'python train_mnist_custom.py --gpus 0 --num-layers {} --num-hidden {} --lr {} --lr-factor {} --lr-step-epoch {}'
out_dir='/data-local/akshayc/Workspace/Software/Spearmint-global/examples/mxnet-mnist/log'

def get_best_validation_accuracy(out_file_name):
    out_file = open(out_file_name, 'r')
    out_lines = out_file.readlines()
    validation_accs = [line for line in out_lines if 'Validation-accuracy' in line]
    v_accs = [float(line.strip().split('=')[1]) for line in validation_accs]
    best_acc = max(v_accs)
    out_file.close()
    return best_acc

def run(num_layers, num_hidden, lr, lr_factor, lr_step_epoch, out_file_name, parameters):
    out_file = open(out_file_name, 'w')
    out_file.write(parameters + '\n')
    run_cmd = train_cmd.format(num_layers, num_hidden, lr, lr_factor, lr_step_epoch)
    subprocess.call(run_cmd, shell=True, stdout=out_file, stderr=out_file)
    out_file.close()
    
def main(job_id, params):
    num_layers = int(params['num_hidden_layers'])
    num_hidden = 32 * int(params['num_neurons'])
    lr = float(params['lr'])
    lr_factor = lr_factors[int(params['lr_factor_idx'])]
    lr_step_epoch = lr_step_epochs[int(params['lr_step_epoch_idx'])]
    parameters='num_layers:{} num_hidden:{} lr:{} lr_factor:{} lr_step_epoch:{}'.format(num_layers, num_hidden, lr, lr_factor, lr_step_epoch)
    print parameters
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file_name='{}/data_{}.log'.format(out_dir, job_id)
    run(num_layers, num_hidden, lr, lr_factor, lr_step_epoch, out_file_name, parameters)
    best_acc = get_best_validation_accuracy(out_file_name)
    err = 1.0 - best_acc
    return err
