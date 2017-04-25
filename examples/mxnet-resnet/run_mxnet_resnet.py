import subprocess
import os

num_neurons = [32*x for x in range(1,101)]
lr_factors = [x * 0.1 for x in range(5,10)]
max_epochs = 300
lr_step_epochs = [','.join([str(step_size + step_size*x) for x in range(max_epochs/step_size - 1)]) for step_size in [30, 60, 90, 120, 150]]
lr_step_epochs.append('200,250')

train_cmd ='python train_cifar10_custom.py --gpus 1 --units {} --filter-list {} --lr {} --lr-factor {} --lr-step-epoch {} --kill-time 0'
out_dir='/data-local/akshayc/Workspace/Software/Spearmint-global/examples/mxnet-resnet/log'
#mxnet_dir='/home/akshayc/mxnet/example/image-classification'

def get_best_validation_accuracy(out_file_name):
    out_file = open(out_file_name, 'r')
    out_lines = out_file.readlines()
    validation_accs = [line for line in out_lines if 'Validation-accuracy' in line]
    v_accs = [float(line.strip().split('=')[1]) for line in validation_accs]
    best_acc = max(v_accs)
    out_file.close()
    return best_acc

def run(units, filter_list, lr, lr_factor, lr_step_epoch, out_file_name, parameters):
    out_file = open(out_file_name, 'w')
    out_file.write(parameters + '\n')
    run_cmd = train_cmd.format(' '.join([str(j) for j in units]),
                               ' '.join([str(j) for j in filter_list]),
                               lr,
                               lr_factor,
                               lr_step_epoch)
    subprocess.call(run_cmd, shell=True, stdout=out_file, stderr=out_file)
    out_file.close()
    
def main(job_id, params):
    #os.chdir(mxnet_dir)
    print params
    per_unit = int(params['per_unit'])
    num_segments = int(params['num_segments'])
    lr = float(params['lr'])
    lr_factor = lr_factors[int(params['lr_factor_idx'])]
    lr_step_epoch = lr_step_epochs[int(params['lr_step_epoch_idx'])]
    units = [per_unit for j in range(num_segments)]
    filter_list = [16*(2**j) for j in range(num_segments+1)]

    parameters='num_segments:{} units:{} filter_list:{} lr:{} lr_factor:{} lr_step_epoch:{}'.format(num_segments, units, filter_list, lr, lr_factor, lr_step_epoch)
    print parameters
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file_name='{}/data_{}.log'.format(out_dir, job_id)
    run(units, filter_list, lr, lr_factor, lr_step_epoch, out_file_name, parameters)
    best_acc = get_best_validation_accuracy(out_file_name)
    err = 1.0 - best_acc
    return err
