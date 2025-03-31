import os
import metrics
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import utils
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from get_data import GetData
from model import MvFD
import copy
import scipy.sparse as sparse


chunk_list = {'HW6': 2000, 'CAL20': 2386, 'CAL101': 9144, 'CCV': 6773, 'SUN': 3445, 'NUS': 5000, 'Youtube': 4413}


def get_args_parser():
    parser = argparse.ArgumentParser('MvFD', add_help=False)

    parser.add_argument('--model', default='MvFD', type=str)
    parser.add_argument('--channels', nargs='+', type=int, help='--channels -1 200')
    parser.add_argument('--data_set', default='Caltech101_20', type=str)
    parser.add_argument('--dataset_location', default='datasets/', type=str)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_false', default=True)
    parser.add_argument('--views_use', nargs='+', type=int, help='--views_use -1')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epoch_start', default=0, type=int)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--validate_every', default=1, type=int)
    parser.add_argument('--device', default='1')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--resume_path', default='')
    parser.add_argument('--premodel_epoch', default='')
    parser.add_argument('--testing', action='store_true', default=False)
    parser.add_argument('--testing_path', default='')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--output_str', default='')
    parser.add_argument('--dim_feature', default=60, type=int)
    parser.add_argument('--channels_deg', nargs='+', type=int, help='--channels_deg -1 200')
    parser.add_argument('--epochs_pre', default=1000, type=int)
    parser.add_argument('--epochs_h', default=10, type=int)
    parser.add_argument('--epochs2', default=200, type=int)
    parser.add_argument('--lr_pre', default=1.0e-3, type=float)
    parser.add_argument('--lr_ft', default=1.0e-4, type=float)
    parser.add_argument('--w_rec', default=1., type=float)
    parser.add_argument('--w_coe', default=1., type=float)
    parser.add_argument('--w_ort', default=0., type=float)
    parser.add_argument('--w_ali', default=0., type=float)
    parser.add_argument('--w_mcr', default=0., type=float)
    parser.add_argument('--w_mi', default=0., type=float)
    parser.add_argument('--d', default=8, type=int)
    parser.add_argument('--ro', default=15, type=int)
    return parser

def main(args):
    utils.print_args(args)
    args.device = 'cuda:' + args.device
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    print("Loading dataset ....")
    dataset_train = GetData(data_dir=args.dataset_location, data_name=args.data_set, mode='all', views_use=args.views_use, seed=args.seed)
    args.class_num = dataset_train.get_num_class()
    args.view_num = dataset_train.get_num_view()
    args.sample_num = len(dataset_train)
    # args.batch_size = len(dataset_train)
    args.batch_size = chunk_list[args.data_set]

    args.view_shape = dataset_train.get_view_list()
    print('view_shpae: ', args.view_shape, ', batch_size: ', args.batch_size)
    data_loader_train = torch.utils.data.DataLoader(dataset_train, shuffle=False,
                                                    batch_size=len(dataset_train), num_workers=args.num_workers,
                                                    pin_memory=args.pin_mem, drop_last=False)
    print(f"Creating model: {args.model} ...")
    model_dic = {"MvFD": MvFD(args=args)}
    model = model_dic[args.model]
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.resume:
        resume_path = args.output_dir + '/last_premodel.pth' if args.resume_path == '' else args.resume_path
        print('Resuming From: ', resume_path)
        checkpoint = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(checkpoint['state'], strict=True)
        args.epoch_start = checkpoint['epoch_pre']


    train_x, train_y = dataset_train.get_full_data()
    train_x = [torch.Tensor(train_x[v]).to(device) for v in range(len(train_x))]
    if np.min(train_y) == 0:
        train_y += 1
    train_y = np.squeeze(train_y)

    if args.testing:
        tsting(args, n_parameters)
    else:
        print('Training...........')
        acc_max = 0.
        utils.dict2json(os.path.join(args.output_dir, 'log_args.json'), {'args': str(args)}, False)
        model.to(device)

        print('--------------------- Step1: Pretraining enc, dec, and con with L_rec L_ort ---------------------')
        params1_ae = [p for n, p in model.named_parameters() if "enc" in n or "dec" in n]
        params1_fea = [p for n, p in model.named_parameters() if "con" in n]
        optimizer1_ae = torch.optim.Adam(params=params1_ae, lr=args.lr_pre)
        optimizer1_fea = torch.optim.Adam(params=params1_fea, lr=args.lr_pre)

        n_iter_per_epoch = args.sample_num // args.batch_size

        premodel_path = args.output_dir + '/premodel'+str(args.epochs_pre)+'.pth' if args.premodel_epoch == '' else args.output_dir + '/premodel'+str(args.premodel_epoch)+'.pth'
        for epoch in range(args.epoch_start, args.epochs_pre):
            model.train()
            if os.path.exists(premodel_path):
                print('There is already a pretraining model and load success!', epoch)
                checkpoint = torch.load(premodel_path, map_location='cpu')
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint['state'].items() if "enc" in k or "dec" or "con" in k}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict, strict=False)
                break
            log_dict = {'epoch_pre': epoch, 'lr_ae': optimizer1_ae.state_dict()['param_groups'][0]['lr'], 'lr_fea': optimizer1_fea.state_dict()['param_groups'][0]['lr']}
            for idx, (data, targets) in enumerate(data_loader_train):


                randidx = torch.randperm(args.sample_num)
                for i in range(n_iter_per_epoch):
                    batch_idx = randidx[i * args.batch_size : (i + 1) * args.batch_size]
                    batch = [data[v][batch_idx].to(device).type(torch.cuda.FloatTensor) for v in range(args.view_num)]

                    logits = model(batch, batch_idx)
                    loss1_ae = model.loss_rec_ort(batch, logits, batch_idx)
                    optimizer1_ae.zero_grad()
                    loss1_ae.backward()
                    optimizer1_ae.step()
                    log_dict['loss1_ae'] = loss1_ae.item()

                    for ep in range(args.epochs_h):
                        logits = model(batch, batch_idx)
                        loss1_fea = model.loss_rec_ort(batch, logits, batch_idx)
                        optimizer1_fea.zero_grad()
                        loss1_fea.backward()
                        optimizer1_fea.step()
                        log_dict['loss1_fea'] = loss1_fea.item()

            utils.dict2json(os.path.join(args.output_dir, 'log_pretrain.json'), log_dict, True)
            log_dict['state'] = model.state_dict()
            torch.save(log_dict, os.path.join(args.output_dir, "last_premodel.pth"))
            save_epoch_list = [1999]
            if epoch in save_epoch_list:
                torch.save(log_dict, os.path.join(args.output_dir, "premodel"+str(epoch+1)+".pth"))

        model.feature_uni.data = model.feature_con.data
        print('--------------------------Step2: Finetuning ae+con , Training uni--------------------------')
        params2_ft = [p for n, p in model.named_parameters() if "enc" in n or "dec" in n or "con" in n]
        params2_other = [p for n, p in model.named_parameters() if "enc" not in n and "dec" not in n and "con" not in n]
        optimizer2_ft = torch.optim.Adam(params=params2_ft, lr=args.lr_ft)
        optimizer2_other = torch.optim.Adam(params=params2_other, lr=args.lr_pre)
        for epoch in range(args.epochs2):
            log_dict = {'epoch': epoch, 'lr2_ft': optimizer2_ft.state_dict()['param_groups'][0]['lr'], 'lr2_other': optimizer2_other.state_dict()['param_groups'][0]['lr']}
            for idx, (data, targets) in enumerate(data_loader_train):

                randidx = torch.randperm(args.sample_num)
                for i in range(n_iter_per_epoch):
                    batch_idx = randidx[i * args.batch_size : (i + 1) * args.batch_size]
                    batch = [data[v][batch_idx].to(device).type(torch.cuda.FloatTensor) for v in range(args.view_num)]
                    logits = model(batch, batch_idx)
                    loss2 = model.loss_rec_ort_ali_mib_cr2_coe(batch, logits, batch_idx)
                    optimizer2_ft.zero_grad()
                    optimizer2_other.zero_grad()
                    loss2.backward()
                    optimizer2_ft.step()
                    optimizer2_other.step()
                    log_dict['loss2'] = loss2.item()
            utils.dict2json(os.path.join(args.output_dir, 'log_train_2step' + args.output_dir2 + '.json'), log_dict, False)

            if epoch % args.validate_every == 0:
                model.eval()
                test_dict = {'epoch': epoch}
                train_con = model.feature_con
                train_uni = model.feature_uni
                coef_uni_sp = rep2coef(train_uni, batch_size=chunk_list[args.data_set])
                test_dict.update(metrics.culster_subspace(torch.matmul(train_uni, torch.transpose(train_uni, 0, 1)), train_y, args.class_num, args.d, args.ro))
                test_dict.update(metrics.culster_subspace(torch.from_numpy(coef_uni_sp.todense()), train_y, args.class_num, args.d, args.ro, post_str='_sp'))
                utils.dict2json(os.path.join(args.output_dir, 'log_val2' + args.output_dir2 + '.json'), test_dict, True)
                acc_current = test_dict['SC_ACC'] if 'SC_ACC' in test_dict else test_dict['SP_ACC']
                if float(acc_current) >= acc_max:
                    log_dict['train_con'] = train_con
                    log_dict['train_uni'] = train_uni
                    log_dict['train_y'] = train_y
                    log_dict['class_num'] = args.class_num
                    torch.save(log_dict, os.path.join(args.output_dir, "best_model.pth"))
                    acc_max = float(acc_current)

            log_dict['state'] = model.state_dict()
            torch.save(log_dict, os.path.join(args.output_dir, "last_model.pth"))

        tsting(args, n_parameters)


def rep2coef(data, batch_size=10, non_zeros=1000):
    N, D = data.shape
    non_zeros = min(N, non_zeros)
    if (N % batch_size != 0):
        raise Exception("batch_size should be a factor of dataset size.")
    val = []
    indicies = []
    for i in range(data.shape[0] // batch_size):
        chunk = data[i * batch_size:(i + 1) * batch_size].to(args.device)
        C = torch.mm(chunk, torch.transpose(data, 0, 1)).cpu()
        rows = list(range(batch_size))
        cols = [j + i * batch_size for j in rows]
        C[rows, cols] = 0.0
        _, index = torch.topk(torch.abs(C), dim=1, k=non_zeros)
        val.append(C.gather(1, index).reshape([-1]).cpu().data.numpy())
        index = index.reshape([-1]).cpu().data.numpy()
        indicies.append(index)

    val = np.concatenate(val, axis=0)
    indicies = np.concatenate(indicies, axis=0)
    indptr = [non_zeros * i for i in range(N + 1)]

    C_sparse = sparse.csr_matrix((val, indicies, indptr), shape=[N, N])
    return C_sparse

def tsting(args, n_parameters):
    print('Testing............')
    testing_path = args.output_dir + '/best_model.pth' if args.testing_path == '' else args.testing_path
    print('Testing From: ', testing_path)
    checkpoint = torch.load(testing_path, map_location='cpu')
    test_dict = {'Method': args.model, 'Data': args.data_set}
    train_uni = checkpoint['train_uni']
    train_y = checkpoint['train_y']
    coef_uni_sp = rep2coef(train_uni.to(args.device), batch_size=chunk_list[args.data_set])
    test_dict.update(metrics.culster_subspace(torch.matmul(train_uni, torch.transpose(train_uni, 0, 1)), train_y, args.class_num))
    test_dict.update(metrics.culster_subspace(torch.from_numpy(coef_uni_sp.todense()), train_y, args.class_num, post_str='_sp'))

    test_dict.update({'Epoch': checkpoint['epoch'], 'views': str(args.views_use), 'n_params': n_parameters})
    utils.dict2json(os.path.join('RELUSTS.json'), test_dict, False)
    utils.dict2json(os.path.join(args.output_dir, 'log_test'+args.output_dir2+'.json'), test_dict, True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MetaViewer', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.views_use is None: args.views_use = [-1]
    args.output_dir = 'checkpoints/' if args.output_dir == '' else args.output_dir
    args.output_dir = args.output_dir + args.model + '-' + args.data_set + '-CL' + str(args.channels) + '-D' + str(args.dim_feature) \
                      + '-CD' + str(args.channels_deg) + '-V' + str(args.views_use) + '-E[' + str(args.epochs_pre) \
                      + ', ' + str(args.epochs_h) + ']-L[' + str(args.lr_pre) + ']' + '-W[' + str(args.w_rec) + ', ' + str(args.w_ort) + ']' + args.output_str

    args.output_dir2 = '_E[' + str(args.epochs) + ']_L[' + str(args.lr_ft) + ']_W[' + str(args.w_rec) + ', ' + str(args.w_ort) \
                       + ',' + str(args.w_coe) + ', ' + str(args.w_ali) + ', ' + str(args.w_mcr) + ',' + str(args.w_mi) + ']'

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)