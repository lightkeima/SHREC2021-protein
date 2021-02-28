from utils.metrics import *
import os.path as osp
from glob import glob
import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_labeled', action='store_true')

    args = parser.parse_args()

    emb_dir = '/home/nhhnghia/SHREC-protein-domains/saved_test_labeled_embeddings/edge_conv-off-512-256-10'
    # int(osp.basename(f).split('.')[0]), 
    file_list = [f for f in glob(emb_dir + '/*', recursive=True) if not osp.isdir(f)]
    file_list = sorted(file_list)
    label_arr, dist_arr = label_distance_matrix_from_file(file_list, file_list)

    print(dist_arr)

    if args.is_labeled:
        # label_df = pd.read_csv('/home/nhtduy/SHREC21/protein-physicochemical/trainingClass.csv')
        # label_df = label_df.sort_values(by=['off_file'])
        # label_dict = {key: value for key, value in label_df.to_records(index=False)}
        pf = retrieval_precision(dist_arr, label_arr, 1)
        ps = retrieval_precision(dist_arr, label_arr, 2)

        print(pf)
        print(ps)
        
