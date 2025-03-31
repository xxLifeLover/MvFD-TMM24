import torch
import numpy as np

from scipy import sparse as sp
from scipy.special import comb
from scipy.sparse.linalg import svds

from sklearn import cluster
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, spectral_clustering
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster._supervised import check_clusterings




def cal_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind_row, ind_col = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) * 1.0 / y_pred.size


def contingency_matrix(labels_true, labels_pred, eps=None, sparse=False):
    """Build a contingency matrix describing the relationship between labels.

    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference

    labels_pred : array, shape = [n_samples]
        Cluster labels to evaluate

    eps : None or float, optional.
        If a float, that value is added to all values in the contingency
        matrix. This helps to stop NaN propagation.
        If ``None``, nothing is adjusted.

    sparse : boolean, optional.
        If True, return a sparse CSR continency matrix. If ``eps is not None``,
        and ``sparse is True``, will throw ValueError.

        .. versionadded:: 0.18

    Returns
    -------
    contingency : {array-like, sparse}, shape=[n_classes_true, n_classes_pred]
        Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
        true class :math:`i` and in predicted class :math:`j`. If
        ``eps is None``, the dtype of this array will be integer. If ``eps`` is
        given, the dtype will be float.
        Will be a ``scipy.sparse.csr_matrix`` if ``sparse=True``.
    """

    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")

    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    contingency = sp.coo_matrix((np.ones(class_idx.shape[0]),
                                 (class_idx, cluster_idx)),
                                shape=(n_classes, n_clusters),
                                dtype=np.int_)
    if sparse:
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
    else:
        contingency = contingency.toarray()
        if eps is not None:
            contingency = contingency + eps
    return contingency


def rand_index_score(labels_true, labels_pred):
    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)
    n_samples = labels_true.shape[0]
    n_classes = np.unique(labels_true).shape[0]
    n_clusters = np.unique(labels_pred).shape[0]
    if (n_classes == n_clusters == 1 or
            n_classes == n_clusters == 0 or
            n_classes == n_clusters == n_samples):
        return 1.0
    contingency = contingency_matrix(labels_true, labels_pred)

    n = np.sum(np.sum(contingency))
    t1 = comb(n, 2)
    t2 = np.sum(np.sum(np.power(contingency, 2)))
    nis = np.sum(np.power(np.sum(contingency, 0), 2))
    njs = np.sum(np.power(np.sum(contingency, 1), 2))
    t3 = 0.5 * (nis + njs)

    A = t1 + t2 - t3
    nc = (n * (n ** 2 + 1) - (n + 1) * nis - (n + 1) * njs + 2 * (nis * njs) / n) / (2 * (n - 1))
    AR = (A - nc) / (t1 - nc)
    return A / t1


def b3_precision_recall_fscore(labels_true, labels_pred):
    """Compute the B^3 variant of precision, recall and F-score.
    Parameters
    ----------
    :param labels_true: 1d array containing the ground truth cluster labels.
    :param labels_pred: 1d array containing the predicted cluster labels.
    Returns
    -------
    :return float precision: calculated precision
    :return float recall: calculated recall
    :return float f_score: calculated f_score
    Reference
    ---------
    Amigo, Enrique, et al. "A comparison of extrinsic clustering evaluation
    metrics based on formal constraints." Information retrieval 12.4
    (2009): 461-486.
    """
    # Check that labels_* are 1d arrays and have the same size

    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)

    # Check that input given is not the empty set
    if labels_true.shape == (0,):
        raise ValueError(
            "input labels must not be empty.")

    # Compute P/R/F scores
    n_samples = len(labels_true)
    true_clusters = {}  # true cluster_id => set of sample indices
    pred_clusters = {}  # pred cluster_id => set of sample indices

    for i in range(n_samples):
        true_cluster_id = labels_true[i]
        pred_cluster_id = labels_pred[i]

        if true_cluster_id not in true_clusters:
            true_clusters[true_cluster_id] = set()
        if pred_cluster_id not in pred_clusters:
            pred_clusters[pred_cluster_id] = set()

        true_clusters[true_cluster_id].add(i)
        pred_clusters[pred_cluster_id].add(i)

    for cluster_id, cluster in true_clusters.items():
        true_clusters[cluster_id] = frozenset(cluster)
    for cluster_id, cluster in pred_clusters.items():
        pred_clusters[cluster_id] = frozenset(cluster)

    precision = 0.0
    recall = 0.0

    intersections = {}

    for i in range(n_samples):
        pred_cluster_i = pred_clusters[labels_pred[i]]
        true_cluster_i = true_clusters[labels_true[i]]

        if (pred_cluster_i, true_cluster_i) in intersections:
            intersection = intersections[(pred_cluster_i, true_cluster_i)]
        else:
            intersection = pred_cluster_i.intersection(true_cluster_i)
            intersections[(pred_cluster_i, true_cluster_i)] = intersection

        precision += len(intersection) / len(pred_cluster_i)
        recall += len(intersection) / len(true_cluster_i)

    precision /= n_samples
    recall /= n_samples

    f_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f_score


def cluster_Kmeans(features, labels, n_clusters, count=1):
    """
    :param n_clusters: number of categories
    :param features: input to be clustered
    :param labels: ground truth of input
    :param count:  times of clustering
    :return: average acc and its standard deviation,
             average nmi and its standard deviation
    """
    features = features.to('cpu').detach().numpy()
    labels = labels if isinstance(labels, np.ndarray) else labels.to('cpu').detach().numpy()
    acc = np.zeros(count)
    nmi = np.zeros(count)
    ar = np.zeros(count)
    ri =  np.zeros(count)
    f =  np.zeros(count)
    p = np.zeros(count)
    r = np.zeros(count)
    pred_all = []
    for i in range(count):
        km = KMeans(n_clusters=n_clusters, n_init=10)
        pred = km.fit_predict(features)
        pred_all.append(pred)
    gt = labels.copy()
    gt = np.reshape(gt, np.shape(pred))
    if np.min(gt) == 1:
        gt -= 1
    for i in range(count):
        acc[i] = cal_acc(gt, pred_all[i])
        nmi[i] = normalized_mutual_info_score(gt, pred_all[i])
        ar[i] = adjusted_rand_score(gt, pred_all[i])
        ri[i] = rand_index_score(gt, pred_all[i])
        p[i], r[i], f[i] = b3_precision_recall_fscore(gt, pred_all[i])
    return {"K_ACC": '%05.2f' % (acc.mean() * 100),
            "K_NMI": '%05.2f' % (nmi.mean() * 100),
            "K_AR":  '%05.2f' % (ar.mean()  * 100),
            "K_RI":  '%05.2f' % (ri.mean()  * 100),
            "K_P":   '%05.2f' % (p.mean()   * 100),
            "K_R":   '%05.2f' % (r.mean()   * 100),
            "K_F":   '%05.2f' % (f.mean()   * 100)}

def thrC(C, ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N,N))
        S = np.abs(np.sort(-np.abs(C),axis=0))
        Ind = np.argsort(-np.abs(C),axis=0)
        for i in range(N):
            cL1 = np.sum(S[:,i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while(stop == False and t < N):
                csum = csum + S[t,i]
                if csum > ro*cL1:
                    stop = True
                    Cp[Ind[0:t+1,i],i] = C[Ind[0:t+1,i],i]
                t = t + 1
    else:
        Cp = C

    return Cp


def post_proC(C, K, d=6, alpha=8):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5*(C + C.T)
    r = d*K + 1
    U, S, _ = svds(C,r,v0 = np.ones(C.shape[0]))
    U = U[:,::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis = 1)
    Z = U.dot(U.T)
    Z = Z * (Z>0)
    L = np.abs(Z ** alpha)
    L = L/L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L


def culster_subspace(coef, labels, class_num=10, d=8, ro=15, post_str=""):
    coef = coef - torch.diag(torch.diag(coef))          #?
    commonZ = coef.cpu().detach().numpy()
    alpha = max(0.4 - (class_num - 1) / 10 * 0.1, 0.1)
    commonZ = thrC(commonZ, alpha)
    preds, _ = post_proC(commonZ, class_num, d=int(d), alpha=int(ro))
    acc = cal_acc(labels, preds)
    pur = purity_score(labels, preds)
    nmi = normalized_mutual_info_score(labels, preds)
    _, _, f = b3_precision_recall_fscore(labels, preds)
    return {"SC_ACC"+post_str: '%05.2f' % (acc * 100),
            "SC_NMI"+post_str: '%05.2f' % (nmi * 100),
            "SC_PUR"+post_str: '%05.2f' % (pur * 100),
            "SC_F"+post_str:   '%05.2f' % (f * 100)
            }



from sklearn import metrics
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

import scipy.sparse as sparse
from sklearn.utils import check_symmetric

def spectral_clustering_senet(affinity_matrix_, n_clusters, k, seed=1, n_init=20):
    affinity_matrix_ = check_symmetric(affinity_matrix_)
    laplacian = sparse.csgraph.laplacian(affinity_matrix_, normed=True)
    _, vec = sparse.linalg.eigsh(sparse.identity(laplacian.shape[0]) - laplacian,
                                 k=k, sigma=None, which='LA')
    embedding = normalize(vec)
    _, labels_, _ = cluster.k_means(embedding, n_clusters,
                                         random_state=seed, n_init=n_init)
    return labels_

def culster_subspace_sparse(coef, labels, n_clusters, spectral_dim=15, post_str=''):
    C_sparse_normalized = normalize(coef).astype(np.float32)
    Aff = 0.5 * (np.abs(C_sparse_normalized) + np.abs(C_sparse_normalized).T)
    preds = spectral_clustering_senet(Aff, n_clusters, spectral_dim)

    acc = cal_acc(labels, preds)
    nmi = normalized_mutual_info_score(labels, preds)
    pur = purity_score(labels, preds)
    _, _, f = b3_precision_recall_fscore(labels, preds)
    return {"SP_ACC"+post_str: '%05.2f' % (acc * 100),
            "SP_NMI"+post_str: '%05.2f' % (nmi * 100),
            "SP_PUR"+post_str: '%05.2f' % (pur * 100),
            "SP_F"+post_str:   '%05.2f' % (f * 100)
            }
