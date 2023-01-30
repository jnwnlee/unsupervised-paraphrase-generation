import argparse
import csv
import random
import os
import json
import random

from tqdm import tqdm
# from nltk.corpus import stopwords
# from nltk.tokenize.treebank import TreebankWordTokenizer
# from nltk.tokenize.treebank import TreebankWordDetokenizer
from transformers import AutoTokenizer
# from transformers import GPT2Tokenizer

from eda import synonym_replacement

# english_stopwords = stopwords.words('english') # ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', ...]

# # Stopwords from case study of the paper
# # 1. From case study
# english_stopwords += ['someone', 'something', 'make', 'see']
# # 2. From possible candidates
# english_stopwords += ['everything']
# # 3. Similar words from those of case study
# english_stopwords += ['anyone', 'anything', 'everyone']


# tokenizer = TreebankWordTokenizer()
# detokenizer = TreebankWordDetokenizer()
# list(map(tokenizer.decode, tokenizer.encode(prompt)))


def remove_stopwords(sentence):
    # sentence = tokenizer.tokenize(sentence)
    # sentence = [word for word in sentence
    #             if word.lower() not in english_stopwords]
    # sentence = ' '.join(sentence)
    # regularization
    # sentence = sentence.replace("''", '"').replace('``', '"')
    # TODO: 특수문자 제거?? re /^[ㄱ-ㅎ|가-힣|a-z|A-Z|0-9|]+$/ -> 문장부호는 필요함

    # sentence = detokenizer.detokenize(sentence.split())
    
    return sentence

# https://huggingface.co/beomi/kcbert-base
# import re
# import emoji
# from soynlp.normalizer import repeat_normalize

# emojis = list({y for x in emoji.UNICODE_EMOJI.values() for y in x.keys()})
# emojis = ''.join(emojis)
# pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣{emojis}]+')
# url_pattern = re.compile(
#     r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

# def clean(x):
#     x = pattern.sub(' ', x)
#     x = url_pattern.sub('', x)
#     x = x.strip()
#     x = repeat_normalize(x, num_repeats=2)
#     return x


def sentence_noising(sentence, shuffle_ratio=0.3, replace_ratio=0.2):
    # 1. Synonym replacement
    words = sentence.split()
    n_sr = max(1, int(len(words)*replace_ratio))
    # words = synonym_replacement(words, n_sr)
    for idx in random.sample(range(len(words)), n_sr):
        words[idx] = '[MASK]'

    # 2. Random shuffling
    if random.random() < shuffle_ratio:
        random.shuffle(words)

    return ' '.join(words)


def data_preparation(args):
    # gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium') 
    gpt_tokenizer = AutoTokenizer.from_pretrained(
                        'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
                        bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]', sep_token='[SEP]'
                    )
    data = []

    print('-'*5, 'preprocess sentences.')
    with open(args.input) as f:
        skipped = 0
        for line in tqdm(f):
            sentence = line.strip()
            corrupted_sentence = remove_stopwords(sentence)
            write_line = corrupted_sentence + '\n' + sentence
            if len(gpt_tokenizer.encode(corrupted_sentence)) >= args.min_length \
                and len(gpt_tokenizer.encode(write_line)) < args.max_length:
                data.append([corrupted_sentence, sentence])
            else:
                skipped += 1
    print("Skipped: {}".format(skipped))

    print('-'*5, 'write in files.')
    with open(args.output, 'w') as wf:
        writer = csv.writer(wf)
        for corrupted, sentence in tqdm(data):
            writer.writerow([corrupted, sentence])

    if args.save_noised_output is True:
        with open(args.noised_output, 'w') as wf:
            writer = csv.writer(wf)
            for corrupted, sentence in tqdm(data):
                corrupted = sentence_noising(corrupted, args.shuffle_ratio, args.replace_ratio)
                writer.writerow([corrupted, sentence])
    # else:
    #     raise AssertionError('please save noised output.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='input (file) directory')
    parser.add_argument('--output', type=str, required=True,
                        help='output sentence after removing stop words')

    parser.add_argument('--save_noised_output', action="store_true",
                        help='add noise: synonym replacement and shuffling')
    parser.add_argument('--noised_output', type=str, default=None,
                        help='output sentences with additional noise')

    parser.add_argument('--max_length', type=int, default=1024) # 1024
    parser.add_argument('--min_length', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--shuffle_ratio', type=float, default=0.3)
    parser.add_argument('--replace_ratio', type=float, default=0.2)

    args = parser.parse_args()

    random.seed(args.seed)

    if args.noised_output is None:
        # args.noised_output = args.output + '.0'
        filename = os.path.basename(args.output).split('.')
        filename[-2] = filename[-2]+f'_shuffle_{args.shuffle_ratio}_replace_{args.replace_ratio}'
        filename = '.'.join(filename)
        args.noised_output = os.path.join(os.path.dirname(args.output), filename)
    
    data_preparation(args)
