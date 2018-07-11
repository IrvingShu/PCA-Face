import os
import sys
import os.path as osp
import struct
import numpy as np
from sklearn.externals import joblib
from src import matio
import argparse

from multiprocessing import Process
import multiprocessing
import time


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-list', type=str, help='image list file')
    parser.add_argument('--feature-dir', type=str, help='feature dir')
    parser.add_argument('--feature-dims', type=int, help='feature dims', default=512)
    parser.add_argument('--ipca-model-path', type=str, help='ipca model path')
    parser.add_argument('--save-format', type=str, help='feature format')
    parser.add_argument('--out-dir', type=str, help='where to save the feature after pca')
    parser.add_argument('--nProcess', type=int, default=1, help='number of process')
    return parser.parse_args(argv)


def read_txtlist(path, save_format):
    pathList = []
    with open(path, 'r') as f:
        aPath = f.readline().strip()
        while aPath:
            aPath += save_format
            pathList.append(aPath)
            aPath = f.readline().strip()
    return pathList


def write_bin(path, feature):
  feature = list(feature)
  #print(len(feature))
  with open(path, 'wb') as f:
    f.write(struct.pack('4i', len(feature),1,4,5))
    f.write(struct.pack("%df"%len(feature), *feature))


def multiprocessing_pca(n_process, ipca,fea_dir, fea_list, save_dir, feat_len, save_format,succ_num, failed_num):
    step = len(fea_list) / n_process
    pos = 0
    process_list = []
    for i in range(n_process):
        if i== n_process -1:
            para_pathList = fea_list[pos:]
        else:
            para_pathList = fea_list[pos:pos+step]
        pos += step
        process_list.append(Process(target=cal_pca_fea, args=(ipca,fea_dir, para_pathList, save_dir, feat_len, save_format, failed_num, succ_num)))
        process_list[i].start()

    for i in range(n_process):
        process_list[i].join()
    return


def cal_pca_fea(ipca, fea_dir,fea_list, save_dir, feat_len, save_format,failed_num, succ_num):

    for feature_path in fea_list:

        sub_dir_list = feature_path.split('/')
        sub_dir = ''
        for i in range(len(sub_dir_list) - 1):
            sub_dir = osp.join(sub_dir, sub_dir_list[i])
        final_dir = os.path.join(save_dir, sub_dir)

        if not osp.exists(final_dir):
            os.makedirs(final_dir)
        feature_name = feature_path.strip() + save_format
        feature_path = osp.join(fea_dir, feature_name)
        new_feature_path = osp.join(final_dir, feature_path.strip().split('/')[-1] + save_format)
        if not osp.exists(feature_path):
            print('Not existed: %s' % feature_path)
            failed_num = failed_num + 1
        else:
            feat = np.ravel(matio.load_mat(feature_path))
            x_vec = feat[:feat_len].reshape((1, -1))
            x_vec_2 = None
            if feat.size > feat_len:
                x_vec_2 = feat[feat_len:].reshape((1, -1))
            x_vec_by_pca = ipca.transform(x_vec)

            norm_x_vec = 1.0 * x_vec_by_pca / (np.linalg.norm(x_vec_by_pca, ord=2) + 0.000001)

            # feature is NAN
            if np.isnan(norm_x_vec).sum() > 0:
                print('%s is Nan' % feature_path)
            else:
                if x_vec_2 is not None:
                    norm_x_vec = np.hstack((norm_x_vec, x_vec_2))
                norm_x_vec = norm_x_vec.T
                write_bin(new_feature_path, norm_x_vec)
                succ_num = succ_num + 1
                if succ_num % 10000 == 0:
                    print('####### Save index %d feature  ######' % succ_num)


def main(args):
    print('===> args:\n', args)

    nProcess = args.nProcess
    image_list = args.image_list
    fea_dir = args.feature_dir
    feat_len = args.feature_dims

    ipca_model_path = args.ipca_model_path
    save_format = args.save_format

    save_dir = args.out_dir
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    #load pca model
    ipca = joblib.load(ipca_model_path)

    print('components num: %d' %ipca.n_components)
    print('explained_variance_ratio: %s' % str(ipca.explained_variance_ratio_.shape))
    sum_variance_ratio = 0
    for i in range(ipca.n_components):
        sum_variance_ratio += ipca.explained_variance_ratio_[i]
    print('sum_variance_ratio: %f' %sum_variance_ratio)

    succ_num = multiprocessing.Value("d", 0)
    failed_num = multiprocessing.Value("d", 0)

    multiprocessing_pca(nProcess, ipca,fea_dir, image_list, save_dir, feat_len,save_format, succ_num.value, failed_num.value)
    print('###### Finished Feature Reduce Dims, %d ######' %succ_num.value)
    print('###### Failed Feature Reduce Dims, %d ######' %failed_num.value)


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
