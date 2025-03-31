import torch
import numpy as np
import random
import scipy.io as scio
torch.multiprocessing.set_sharing_strategy('file_system')
from sklearn.preprocessing import MinMaxScaler

name_map = {'HW6': 'Handwritten.mat',
            'CAL20': 'Caltech101-20.mat'}

def normalize(x, min=0):
    if min == 0:
        scaler = MinMaxScaler((0, 1))
    else:
        scaler = MinMaxScaler((-1, 1))
    norm_x = scaler.fit_transform(x)
    return norm_x

class GetData(object):
    def __init__(self, data_dir='datasets/', data_name='HW2', mode='train', views_use=-1, seed=2023):
        super(GetData, self).__init__()
        mat_file = scio.loadmat(data_dir + name_map[data_name])
        views = [mat_file['x'][0][vv] for vv in range(len(mat_file['x'][0]))]
        self.label = mat_file['y']
        num_pre_class = mat_file['num_pre_class'][0].tolist()
        self.views = views if -1 in views_use else [views[idx] for idx in views_use]
        self.views = [normalize(view, 0) for view in self.views]
        self.num_views = len(self.views)
        self.num_classes = len(num_pre_class)
        self.view_list = [v.shape[1] for v in self.views]
        if not isinstance(self.views[0], np.ndarray):
            self.views = [self.views[i].todense().getA() for i in range(self.num_views)]
        print('view_shape: ', [v.shape for v in self.views], 'label_shape: ', self.label.shape, 'num_pre_class: ', num_pre_class, 'num_view: ', self.num_views, 'num_class: ', self.num_classes)


        idx_data = []
        class_index_start = 0
        class_index_end = 0
        for iter, class_num in enumerate(num_pre_class):
            class_index_end += class_num
            sample_index = range(class_index_start, class_index_end)
            target = self.label[class_index_start][0]
            class_samples = []
            for i in range(len(sample_index)):
                sample_view_all = np.tile(sample_index[i], self.num_views)
                class_samples.append((sample_view_all, target))
            random.seed(int(seed))
            train_index = random.sample(range(0, class_num), int(0.6*class_num))
            rem_index = [rem for rem in range(0, class_num) if rem not in train_index]
            val_index = random.sample(rem_index, int(0.5*len(rem_index)))
            test_index = [rem for rem in rem_index if rem not in val_index]

            train_part = [class_samples[i] for i in train_index]
            val_part = [class_samples[i] for i in val_index]
            test_part = [class_samples[i] for i in test_index]
            class_index_start = class_index_end
            if mode == 'train':
                idx_data.extend(train_part)
            elif mode == 'val':
                idx_data.extend(val_part)
            elif mode == 'test':
                idx_data.extend(test_part)
            else:
                idx_data.extend(train_part)
                idx_data.extend(val_part)
                idx_data.extend(test_part)
        self.idx_data = idx_data


    def __len__(self):
        return len(self.idx_data)

    def get_num_class(self):
        return self.num_classes

    def get_num_view(self):
        return self.num_views

    def get_view_list(self):
        return self.view_list

    def get_idx_data(self):
        return self.idx_data

    def get_full_data(self, flag_tensor = False):
        idx_list = [ii[0][0] for ii in self.idx_data]
        target = [ii[1] for ii in self.idx_data]
        if flag_tensor:
            full_data = [torch.from_numpy(np.array(self.views[iii][idx_list]).astype(float)).type(torch.FloatTensor) for iii in range(self.num_views)]
            return full_data,torch.tensor(target)
        else:
            full_data = [np.array(self.views[iii][idx_list]).astype(float) for iii in range(self.num_views)]
            return full_data, target

    def get_mean_std(self):
        idx_list = [ii[0][0] for ii in self.idx_data]
        mean = [np.mean(np.array(self.views[iii][idx_list]).astype(float)) for iii in range(self.num_views)]
        std = [np.std(np.array(self.views[iii][idx_list]).astype(float)) for iii in range(self.num_views)]
        return mean, std


    def __getitem__(self, index):
        data = [torch.from_numpy(np.array(self.views[iii][self.idx_data[index][0][0]]).astype(float)).type(torch.FloatTensor) for iii
                     in range(self.num_views)]
        target = torch.tensor(self.idx_data[index][1])
        return data, target