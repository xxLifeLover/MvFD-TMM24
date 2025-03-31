import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softplus
import copy
import numpy as np
from sklearn.cluster import DBSCAN


class MIEstimator(nn.Module):
    def __init__(self, size1, size2):
        super(MIEstimator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(size1 + size2, 256),
            nn.ReLU(True),
            nn.Linear(256, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1),
        )
    def forward(self, x1, x2):
        pos = self.net(torch.cat([x1, x2], 1)) 
        neg = self.net(torch.cat([torch.roll(x1, 1, 0), x2], 1))
        return -softplus(-pos).mean() - softplus(neg).mean(), pos.mean() - neg.exp().mean() + 1


class AdaptiveSoftThreshold(nn.Module):
    def __init__(self, dim):
        super(AdaptiveSoftThreshold, self).__init__()
        self.dim = dim
        self.register_parameter("bias", nn.Parameter(torch.from_numpy(np.zeros(shape=[self.dim])).float()))

    def forward(self, c):
        return torch.sign(c) * torch.relu(torch.abs(c) - self.bias)


class AssignNet(nn.Module):
    def __init__(self, in_dim, class_num):
        super(AssignNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, class_num),
        )

    def forward(self, x):
        return self.net(x)



def get_pseudo_label(features):
    features = features.to('cpu').detach().numpy()
    clustering = DBSCAN(eps=3, min_samples=2).fit(features)
    return clustering.labels_


class MaximalCodingRateReduction(torch.nn.Module):
    def __init__(self, gam1=1.0, gam2=1.0, eps=0.01, device = 'cuda:0'):
        super(MaximalCodingRateReduction, self).__init__()
        self.gam1 = gam1
        self.gam2 = gam2
        self.eps = eps
        self.device = device

    def compute_discrimn_loss_empirical(self, W):
        p, m = W.shape
        I = torch.eye(p).to(self.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + self.gam1 * scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss_empirical(self, W, Pi):
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p).to(self.device)
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += log_det * trPi / m
        return compress_loss / 2.

    def forward(self, X, Y, num_classes=None):
        if num_classes is None:
            num_classes = Y.max() + 1
        W = X.T
        Pi = label_to_membership(Y.numpy(), num_classes)
        Pi = torch.tensor(Pi, dtype=torch.float32).to(self.device)


        discrimn_loss_empi = self.compute_discrimn_loss_empirical(W)
        compress_loss_empi = self.compute_compress_loss_empirical(W, Pi)

        total_loss_empi = self.gam2 * -discrimn_loss_empi + compress_loss_empi
        return total_loss_empi


def label_to_membership(targets, num_classes=None):
    targets = one_hot(targets, num_classes)
    num_samples, num_classes = targets.shape
    Pi = np.zeros(shape=(num_classes, num_samples, num_samples))
    for j in range(len(targets)):
        k = np.argmax(targets[j])
        Pi[k, j, j] = 1.
    return Pi


def one_hot(labels_int, n_classes):
    labels_onehot = torch.zeros(size=(len(labels_int), n_classes)).float()
    for i, y in enumerate(labels_int):
        labels_onehot[i, y] = 1.
    return labels_onehot


class MvFD(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.sample_num = args.sample_num
        self.view_num = args.view_num
        self.view_shape = args.view_shape
        self.dim_feature = args.dim_feature
        self.class_num = args.class_num
        self.device = args.device

        self.w_rec = args.w_rec
        self.w_coe = args.w_coe
        self.w_ort = args.w_ort
        self.w_ali = args.w_ali
        self.w_mcr = args.w_mcr
        self.w_mi = args.w_mi

        channels_enc = copy.deepcopy(args.channels)
        channels_dec = copy.deepcopy(args.channels)
        encoders = []
        heads_red = []
        heads_com = []
        decoders = []
        for v in range(self.view_num):
            encoder = []
            channels_enc[0] = args.view_shape[v]
            for i in range(0, len(channels_enc) - 1):
                encoder.append(nn.Sequential(nn.Linear(channels_enc[i], channels_enc[i + 1]), nn.Sigmoid()))
            encoders.append(nn.Sequential(*encoder))
            heads_red.append(nn.Sequential(nn.Linear(channels_enc[-1], self.dim_feature), nn.Sigmoid()))
            heads_com.append(nn.Sequential(nn.Linear(channels_enc[-1], self.dim_feature), nn.Sigmoid()))
            decoder = []
            channels_dec[0] = args.view_shape[v]
            channels_dec[-1] = self.dim_feature * 3
            for i in range(0, len(channels_dec) - 1):
                decoder.append(nn.Sequential(nn.Linear(channels_dec[len(channels_dec) - i - 1], channels_dec[len(channels_dec) - i - 2]), nn.Sigmoid()))   # Note: Relu最后一层加不加？
            decoders.append(nn.Sequential(*decoder))
        self.encoders = nn.ModuleList(encoders)
        self.heads_red = nn.ModuleList(heads_red)
        self.heads_com = nn.ModuleList(heads_com)
        self.decoders = nn.ModuleList(decoders)
        self.feature_con = nn.Parameter( torch.from_numpy(np.random.uniform(0, 1, [self.sample_num, self.dim_feature])).float().to(self.device), requires_grad=True)
        self.assign_net = AssignNet(self.view_num * self.dim_feature, self.class_num)
        self.lossF_mcr = MaximalCodingRateReduction(gam1=self.class_num, gam2=self.class_num, eps=0.5, device=self.device)
        self.mi_estimator = MIEstimator(self.dim_feature, self.dim_feature)
        self.feature_uni = nn.Parameter(
            torch.from_numpy(np.random.uniform(0, 1, [self.sample_num, self.dim_feature])).float().to(self.device),
            requires_grad=True)
        self.thres = AdaptiveSoftThreshold(1)
        self.shrink = 1.0 / args.dim_feature
        self.lossF_mse = torch.nn.MSELoss(reduction='mean')
        self.lossF_l1 = torch.nn.L1Loss()
        self._init(self)

    def get_coeff(self, q_emb, k_emb):
        c = self.thres(q_emb.mm(k_emb.t()))
        return self.shrink * c


    def forward(self, data, batch_idx):
        feature_red = []
        feature_com = []
        recs = []
        current_con = self.feature_con[batch_idx]
        for v in range(self.view_num):
            feature_temp = self.encoders[v](data[v])
            feature_red.append(self.heads_red[v](feature_temp))
            feature_com.append(self.heads_com[v](feature_temp))
            enc_cat = torch.cat([feature_red[v], feature_com[v]], dim=1)
            recs.append(self.decoders[v](torch.cat([enc_cat, current_con], dim=1)))
        return [feature_red, feature_com, recs]


    def loss_rec_ort(self, data, logits, batch_idx):
        [feature_red, feature_com, recs] = logits
        loss_rec = 0.0
        loss_ort = 0.0
        current_con = self.feature_con[batch_idx]
        for v in range(self.view_num):
            loss_rec += 0.5 * self.lossF_mse(recs[v], data[v])

            red_con = feature_red[v] * current_con
            com_con = feature_com[v] * current_con
            com_red = feature_com[v] * feature_red[v]
            loss_ort += self.lossF_l1(com_con, torch.zeros_like(com_con))
            loss_ort += self.lossF_l1(com_red, torch.zeros_like(com_red))
            loss_ort += self.lossF_l1(red_con, torch.zeros_like(red_con))
        return loss_rec + self.w_ort * loss_ort

    def loss_rec_ort_ali_mib_cr2_coe(self, data, logits, batch_idx):
        [feature_red, feature_com, recs] = logits
        loss_rec = 0.0
        loss_ort = 0.0
        loss_ali = 0.0
        current_con = self.feature_con[batch_idx]
        current_uni = self.feature_uni[batch_idx]
        for v in range(self.view_num):
            loss_rec += 0.5 * self.lossF_mse(recs[v], data[v])

            red_con = feature_red[v] * current_con
            com_con = feature_com[v] * current_con
            com_red = feature_com[v] * feature_red[v]
            loss_ort += self.lossF_l1(com_con, torch.zeros_like(com_con))
            loss_ort += self.lossF_l1(com_red, torch.zeros_like(com_red))
            loss_ort += self.lossF_l1(red_con, torch.zeros_like(red_con))

        feature_com_cat = torch.cat([feature_com[v] for v in range(self.view_num)], dim=1)
        feature_red_cat = torch.cat([feature_red[v] for v in range(self.view_num)], dim=1)
        feature_con_cat = torch.cat([current_con for v in range(self.view_num)], dim=1)
        coe_com = F.softmax(self.assign_net(feature_com_cat))
        coe_red = F.softmax(self.assign_net(feature_red_cat))
        coe_con = F.softmax(self.assign_net(feature_con_cat))
        loss_ali += self.lossF_mse(coe_con, coe_com)
        loss_ali += self.lossF_mse((1-coe_con), coe_red)

        mi_gradient, mi_estimation = self.mi_estimator(current_uni, current_con)
        loss_mi_uni_co2 = mi_gradient.mean()
        loss_mi_uni_red = 0.0
        for v in range(self.view_num):
            mi_gradient, mi_estimation = self.mi_estimator(current_uni, feature_com[v])
            loss_mi_uni_co2 += mi_gradient.mean()
            mi_gradient, mi_estimation = self.mi_estimator(current_uni, feature_red[v])
            loss_mi_uni_red += mi_gradient.mean()

        pseudo_label = torch.from_numpy(get_pseudo_label(coe_con))
        loss_mcr = self.lossF_mcr(current_uni, pseudo_label)

        coef = self.get_coeff(current_uni, current_uni)
        rec_coe = coef.mm(current_uni)
        loss_coe = self.lossF_mse(rec_coe, current_uni) / len(batch_idx)
        loss_coe += self.lossF_l1(coef, torch.ones_like(coef).to(self.device))

        return self.w_rec * loss_rec + self.w_ort * loss_ort + self.w_ali * loss_ali + self.w_coe * loss_coe + self.w_mcr * loss_mcr + self.w_mi * loss_mi_uni_red - self.w_mi * loss_mi_uni_co2


    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)