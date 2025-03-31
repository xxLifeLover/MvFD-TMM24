import io
import os
import time
import numpy as np
import json
import yaml
import torch
import datetime
from collections import defaultdict, deque
import math

class Scheduler:
    def __call__(self, **kwargs):
        raise NotImplemented()


class LinearScheduler(Scheduler):
    def __init__(self, start_value, end_value, n_iterations, start_iteration=0):
        self.start_value = start_value
        self.end_value = end_value
        self.n_iterations = n_iterations
        self.start_iteration = start_iteration
        self.m = (end_value - start_value) / n_iterations

    def __call__(self, iteration):
        if iteration > self.start_iteration + self.n_iterations:
            return self.end_value
        elif iteration <= self.start_iteration:
            return self.start_value
        else:
            return (iteration - self.start_iteration) * self.m + self.start_value



class ExponentialScheduler(LinearScheduler):
    def __init__(self, start_value, end_value, n_iterations, start_iteration=0, base=10):
        self.base = base

        super(ExponentialScheduler, self).__init__(start_value=math.log(start_value, base),
                                                   end_value=math.log(end_value, base),
                                                   n_iterations=n_iterations,
                                                   start_iteration=start_iteration)

    def __call__(self, iteration):
        linear_value = super(ExponentialScheduler, self).__call__(iteration)
        return self.base ** linear_value


class AverageStdMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.vals = []
        self.avg = 0
        self.std = 0
        self.sum = 0

    def update(self, val, n=1):
        self.vals.append(val)
        self.avg = np.mean(self.vals)
        self.std = np.std(self.vals, ddof=1)
        self.sum = np.sum(self.vals)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_joint(x_out, x_tf_out):
    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1) 
    p_i_j = p_i_j.sum(dim=0)  
    p_i_j = (p_i_j + p_i_j.t()) / 2. 
    p_i_j = p_i_j / p_i_j.sum() 

    return p_i_j


import sys
def instance_contrastive_Loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

    loss = - p_i_j * (torch.log(p_i_j) \
                      - lamb * torch.log(p_j) \
                      - lamb * torch.log(p_i))

    loss = loss.sum()

    return loss


def get_config(args):
    if args.configs=='':
        return
    config = yaml.load(open('configs/'+args.configs, 'r'))
    args.data_set = config['dataset']['data_set']
    args.dataset_location = config['dataset']['dataset_location']

    args.model = config['model']['model']
    args.backbone = config['model']['backbone']

    args.training_mode = config['training']['training_mode']
    args.batch_size = config['training']['batch_size']
    args.epochs = config['training']['epochs']
    args.validate_every = config['training']['validate_every']
    args.output_str = config['training']['output_str']

    args.opt = config['optimizer']['opt']
    args.lr = config['optimizer']['lr']
    args.weight_decay = config['optimizer']['weight_decay']
    args.sched = config['optimizer']['sched']

    if 'resume' in config.keys() and config['resume'] != '':
        args.resume = config['resume']

    if args.backbone == 'ViT':
        args.patch_num = config['net']['patch_num']
        args.embed_dim = config['net']['embed_dim']
        args.depth = config['net']['depth']
        args.num_heads = config['net']['num_heads']
    elif args.backbone == 'CNN':
        args.channels = list(map(int, config['net']['channels'].split(' ')))
        args.kernels = list(map(int, config['net']['kernels'].split(' ')))
    else:
        args.channels = list(map(int, config['net']['channels'].split(' ')))

    if args.training_mode == 'pretrain':
        args.output_dir = 'checkpoints/' + args.model + args.data_set + '-' + args.backbone + '-E' + str(args.epochs) + '-V' + str(args.validate_every) + args.output_str
    else:
        args.output_dir = 'checkpoints/' + args.model + args.data_set + '-' + args.backbone + '-E' + str(args.epochs) + '-V' + str(
            args.validate_every) + args.output_str
    return args


def write_log(path, logs):
    with open(path, 'a') as f:
        f.write(json.dumps(logs) + "\n")


import json


def dict2json(file_name, res, flag=True):
    if flag:
        print(res)
    with open(file_name, 'a+') as json_file:
        json.dump(res, json_file)
        json_file.write('\n')


class SmoothedValue(object):

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()*100

    @property
    def var(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.var().item()*100

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def save_on_master(*args, **kwargs):
    torch.save(*args, **kwargs)


def print_args(args):
    print('********************** Hyper-parameters Start *************************')
    for arg in vars(args):
        print(arg, ':', getattr(args, arg)) 
    print('********************** Hyper-parameters End *************************')
    # print('Model: ', args.model)
    # print('Dataset: ', args.data_set)
    # print('Dataset Location: ', args.dataset_location)
    # if args.resume:
    #     print('From ', args.resume_path, 'to ', args.output_dir)
    # else:
    #     print('To ', args.output_dir)
    # print('---------- Network ----------')
    # print('Channels: ', args.channels)
    # if 'Meta' in args.model:
    #     print('Meta Channels: ', args.meta_channels)
    #     print('Meta Kernels: ', args.meta_kernels)
    #     print('Inner Lr: ', args.inner_lr)
    #     print('Outer Lr: ', args.outer_lr)
    #     print('S/Q: ', args.rate_support * 10, ':', (1 - args.rate_support) * 10)
    # print('---------- Training ----------')
    # print('Batchsize: ', args.batch_size)
    # print('Epochs: ', args.epochs)


def deal(list_ori,p):
    list_new=[]				
    list_short=[]			
    for k,v in enumerate(list_ori):
        if v==p and k!=0:
            list_new.append(list_short)
            list_short=[]
        list_short.append(v)
    list_new.append(list_short)
    return list_new



