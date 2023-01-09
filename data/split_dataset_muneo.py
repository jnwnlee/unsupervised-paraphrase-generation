import csv
import os
import random
import json
import collections
import math
import shutil

from matplotlib import pyplot as plt
from tqdm import tqdm

def data_cleansing(text):
    text = ' '.join(text.split())
    return text

def main(args):
    file_dir_dict = collections.OrderedDict()
    for path in os.listdir(args.input):
        fpath = os.path.join(args.input, path)
        if os.path.isfile(fpath):
            with open(fpath, 'rb') as f:
                category = json.load(fpath)['metadata']['category']
                file_dir_dict[category] = file_dir_dict.get(category, []).append(fpath)

    # generate figure for visualizing the distribution of data
    print('-'*5 + 'Extract distribution.')
    values = [len(l) for l in file_dir_dict.values()]
    plt.bar(values)
    plt.xticks(file_dir_dict.keys())
    plt.xlabel('category')
    plt.ylabel('#')
    plt.title('# of books per category')
    for i in range(len(values)):
        plt.text(i, values[i]+5, values[i])
    plt.save('../data_dist_original.png')

    # split given files
    print('-'*5 + 'Split given dataset.')
    split_dir_dict = {}
    for split in ['train', 'dev', 'test']:
        split_dir_dict[split] = collections.OrderedDict()

    for k, v in tqdm(file_dir_dict.items()):
        dev_num = math.ceil(len(v)*args.split_ratio[1])
        test_num = math.ceil(len(v)*args.split_ratio[2])
        ids_list = []
        ids_list['dev'] = random.sample(v, dev_num)
        ids_list['test'] = random.sample(v, test_num)
        ids_list['train'] = list(set(v) - set(ids_list['dev']) - set(ids_list['test']))
        for split in ['train', 'dev', 'test']:
            split_dir_dict[split][k] = ids_list[split]

    # copy files
    print('-'*5 + 'Copy files.')
    for split in tqdm(['train', 'dev', 'test']):
        os.makedirs(os.path.join(args.input, split), exist_ok=True)
        for _, v in tqdm(split_dir_dict[split]):
            for file in v:
                shutil.copy2(file, os.path.join(args.input, split, os.path.basename(file)))

    # generate figure for visualizing the distribution of data
    print('-'*5 + 'Extract distributions.')
    for split in ['train', 'dev', 'test']:  
        values = [len(l) for l in split_dir_dict[split].values()]
        plt.bar(values)
        plt.xticks(split_dir_dict[split].keys())
        plt.xlabel('category')
        plt.ylabel('#')
        plt.title('# of books per category')
        for i in range(len(values)):
            plt.text(i, values[i]+5, values[i])
        plt.save(f'../data_dist_{split}.png')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, default=''
                        help='input (file) directory')
    parser.add_argument('--output_train', type=str, required=True,
                        help='train dataset output directory')
    parser.add_argument('--output_dev', type=str, required=True,
                        help='dev dataset output directory')
    parser.add_argument('--output_test', type=str, required=True,
                        help='test dataset output directory')
    parser.add_argument('--split_ratio', type=list, required=True,
                        help='dataset split ratio. sum should be 10')

    if len(args.split_ratio) != 3 or sum(args.split_ratio) != 10:
        raise ValueError('args.split_ratio: the sum must be 10.')
    
    # Read file
    paraphrase_labeled_file = 'QQP/train.csv'
    paraphrase_unlabeled_file = 'QQP/test.csv'
    # Write files
    train_file = 'QQP_split/train.txt'
    dev_file = 'QQP_split/dev.txt'
    test_input_file = 'QQP_split/test_input.txt'
    test_target_file = 'QQP_split/test_target.txt'

    test_pair_num = 30000
    dev_num = 60000
    unlabeled_used = 300000

    random.seed(1234)
    os.mkdir("QQP_split")
    463010 60000 30000 83.7 10.8 5.42 7 2 1
    #################################################
    with open(paraphrase_labeled_file) as f:
        reader = csv.reader(f)
        questions_1 = []
        questions_2 = []
        paraphrases = []
        header = next(reader)
        for idx, row in enumerate(reader):
            _, _, _, question1, question2, is_duplicate = row
            questions_1.append(data_cleansing(question1))
            questions_2.append(data_cleansing(question2))
            if is_duplicate == '1':
                paraphrases.append(idx)

    with open(paraphrase_unlabeled_file) as f:
        reader = csv.reader(f)
        header = next(reader)
        unlabeled_questions = []
        for idx, row in enumerate(reader):
            _, question1, _ = row
            unlabeled_questions.append(data_cleansing(question1))
            if idx >= unlabeled_used:
                break

    test_indices = random.sample(paraphrases, test_pair_num)
    test_questions = [questions_1[idx] for idx in test_indices] \
        + [questions_2[idx] for idx in test_indices]
    test_questions = set(test_questions)

    questions = set(questions_1 + unlabeled_questions)
    training_questions = list(questions - set(test_questions))

    print("# questions: {}".format(len(questions)))
    print("# training questions: {}".format(len(training_questions)))

    with open(test_input_file, 'w', newline='') as f_i, \
            open(test_target_file, 'w', newline='') as f_t:
        for idx in test_indices:
            q1 = questions_1[idx]
            q2 = questions_2[idx]
            f_i.write(q1 + '\n')
            f_t.write(q2 + '\n')

    # Converting set to list could be shuffled the order,
    # but we want the same result with the same random seed
    training_questions.sort()

    random.shuffle(training_questions)
    with open(dev_file, 'w') as f:
        for question in training_questions[:dev_num]:
            f.write(question + '\n')
    with open(train_file, 'w') as f:
        for question in training_questions[dev_num:]:
            f.write(question + '\n')

