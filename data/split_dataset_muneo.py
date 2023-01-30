import csv
import os
import random
import json
import collections
import math
import shutil
import argparse

from kiwipiepy import Kiwi
import kss
from matplotlib import pyplot as plt
from matplotlib import rc
from tqdm import tqdm

plt.figure(figsize=(13, 8))

def data_cleansing(text):
    text = ' '.join(text.split())
    return text

def plot_bar(x, y, name: str):
    # plt.clf
    rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False

    plt.bar(x, y, label=name) # width=
    plt.xticks(rotation=20, fontsize=12)
    plt.xlabel('category', fontsize=12)
    plt.ylabel('# of books', fontsize=12)
    plt.title('# of books per category', fontsize=17)
    plt.legend()
    for i in range(len(y)):
        x_offset = 0.2*['original', 'train', 'dev', 'test'].index(name)
        plt.text(i-0.43+x_offset, y[i]+10, y[i], fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join('../', f'data_dist_{name}.png'))

def split_and_copy(args):
    file_dir_dict = {}
    print('-'*5 + 'Read files and Sort by categories.')
    for path in tqdm(os.listdir(args.input)):
        fpath = os.path.join(args.input, path)
        if os.path.isfile(fpath) and os.path.basename(fpath).split('.')[-1] == 'json': # exclude pdf description file
            with open(fpath, 'r') as f:
                category = json.load(f)['metadata']['category']
                file_dir_dict[category] = file_dir_dict.get(category, list()) + [fpath]
    file_dir_dict = collections.OrderedDict(sorted(file_dir_dict.items()))

    # generate figure for visualizing the distribution of data
    print('-'*5 + 'Plot distribution.')
    values = [len(l) for l in file_dir_dict.values()]
    plot_bar(x=file_dir_dict.keys(), y=values, name='original')

    # split given files
    print('-'*5 + 'Split given dataset.')
    split_dir_dict = {}
    for split in ['train', 'dev', 'test']:
        split_dir_dict[split] = {}

    for k, v in tqdm(file_dir_dict.items()):
        dev_num = math.ceil(len(v)*args.split_ratio[1]/sum(args.split_ratio))
        test_num = math.ceil(len(v)*args.split_ratio[2]/sum(args.split_ratio))
        ids_list = {}
        ids_list['dev'] = random.sample(v, dev_num)
        ids_list['test'] = random.sample(list(set(v) - set(ids_list['dev'])), test_num)
        ids_list['train'] = list(set(v) - set(ids_list['dev']) - set(ids_list['test']))
        for split in ['train', 'dev', 'test']:
            split_dir_dict[split][k] = ids_list[split]
    for split in ['train', 'dev', 'test']:
        split_dir_dict[split] = collections.OrderedDict(sorted(split_dir_dict[split].items()))

    # copy files
    print('-'*5 + 'Copy files.')
    for split in tqdm(['train', 'dev', 'test']):
        os.makedirs(os.path.join(args.input, split), exist_ok=True)
        for _, v in tqdm(split_dir_dict[split].items()):
            for file in v:
                shutil.copy2(file, os.path.join(args.input, split, os.path.basename(file)))

    # generate figure for visualizing the distribution of data
    print('-'*5 + 'Plot distributions.')
    for split in ['train', 'dev', 'test']:  
        values = [len(l) for l in split_dir_dict[split].values()]
        plot_bar(x=split_dir_dict[split].keys(), y=values, name=split)

def merge_files(args):
    kiwi = Kiwi()
    for split in ['train', 'dev', 'test']:
        if split == 'train':
            split_dir = args.output_train
        elif split == 'dev':
            split_dir = args.output_dev
        elif split == 'test':
            split_dir = args.output_test
        
        file_list = os.listdir(split_dir)
        print('-'*5 + 'Merge Files.')
        with open(os.path.join(args.input, split+'_final_mecab.txt'), 'w') as wf:
            for idx, path in tqdm(enumerate(file_list)):
                fpath = os.path.join(args.input, path)
                if os.path.isfile(fpath) and os.path.basename(fpath).split('.')[-1] == 'json': # exclude pdf description file
                    with open(fpath, 'r') as jsonf:
                        book = json.load(jsonf)['document']
                        assert len(book) == 1 , 'More than one document in a single file!'
                        for paragraph in book[0]['paragraph']:
                            if not isinstance(paragraph['form'], str):
                                print(f"paragraph['form'] not a string but {type(paragraph['form'])}")
                            sentences = kss.split_sentences(paragraph['form'], backend='mecab', num_workers='auto') # kiwi.split_into_sents(paragraph['form'])
                            # sentences = list(map(lambda s: s.text, sentences)) # for kiwi
                            wf.write('\n'.join(sentences))
                            wf.write('\n') if idx < len(file_list)-1 else None


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='../../NIKL_WRITTEN(v1.1)/', # required=True, 
                        help='input (file) directory')
    parser.add_argument('--output_train', type=str, default='../../NIKL_WRITTEN(v1.1)/train/', # required=True, 
                        help='train dataset output directory')
    parser.add_argument('--output_dev', type=str, default='../../NIKL_WRITTEN(v1.1)/dev/', # required=True, 
                        help='dev dataset output directory')
    parser.add_argument('--output_test', type=str, default='../../NIKL_WRITTEN(v1.1)/test/', # required=True, 
                        help='test dataset output directory')
    parser.add_argument('--split_ratio', type=list, default=[7, 2, 1], # required=True, 
                        help='dataset split ratio. sum should be 10')
    parser.add_argument('--seed', type=int, default=1234)
    
    args = parser.parse_args()

    if len(args.split_ratio) != 3 or sum(args.split_ratio) != 10:
        raise ValueError('args.split_ratio: the sum must be 10.')

    random.seed(args.seed)

    # split_and_copy(args)
    merge_files(args)