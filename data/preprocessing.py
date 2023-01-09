import argparse
import csv
import random
import os
import json
import random

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

tokenizer = AutoTokenizer.from_pretrained(
  'kakaobrain/kogpt', revision='KoGPT6B-ryan1.5b-float16',  # or float32 version: revision=KoGPT6B-ryan1.5b
  bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
)

# tokens = tokenizer.encode(prompt, return_tensors='pt').to(device='cuda', non_blocking=True)
# generated = tokenizer.batch_decode(gen_tokens)[0]
# tokenizer = TreebankWordTokenizer()
# detokenizer = TreebankWordDetokenizer()


def remove_stopwords(sentence):
    # sentence = tokenizer.tokenize(sentence)
    sentence = tokenizer.encode(sentence, return_tensors='pt').to(device='cuda', non_blocking=True)
    # sentence = [word for word in sentence
    #             if word.lower() not in english_stopwords]
    sentence = ' '.join(sentence)
    # regularization
    sentence = sentence.replace("''", '"').replace('``', '"')
    # TODO: 특수문자 제거?? re /^[ㄱ-ㅎ|가-힣|a-z|A-Z|0-9|]+$/ -> 문장부호는 필요함

    # sentence = detokenizer.detokenize(sentence.split())
    sentence = tokenizer.batch_decode(sentence.split())[0]

    return sentence


def sentence_noising(sentence, shuffle_ratio=0.2, replace_ratio=0.2):
    # 1. Synonym replacement
    words = sentence.split()
    n_sr = max(1, int(len(words)*shuffle_ratio))
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
                        bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
                    )
    data = []

    if args.data_name == 'muneo':
        file_list = []
        for path in os.listdir(args.input):
            if os.path.isfile(os.path.join(args.input, path)):
                file_list.append(os.path.join(args.input, path))
        
        skipped = 0
        for fpath in file_list:
            with open(fpath, 'rb') as f:
                book_file = json.load(f)
                for document in book_file['document']:
                    for paragraph in document['paragraph']:
                        sentence = paragraph['form'].strip()
                        corrupted_sentence = remove_stopwords(sentence)
                        write_line = corrupted_sentence + '\n' + sentence
                        if len(gpt_tokenizer.encode(write_line)) < args.max_length:
                            data.append([corrupted_sentence, sentence])
                        else:
                            skipped += 1
                # book_corpus['document'][0]['paragraph'][0]['form'] 
                # id, metadata, document #index #id, metadata, paragraph #index #id, form
    else:
        raise AttributeError('wrong data name args.data_name.')

    # with open(args.input) as f:
    #     skipped = 0
    #     for line in f:
    #         sentence = line.strip()
    #         corrupted_sentence = remove_stopwords(sentence)
    #         write_line = corrupted_sentence + '\n' + sentence
    #         if len(gpt_tokenizer.encode(write_line)) < args.max_length:
    #             data.append([corrupted_sentence, sentence])
    #         else:
    #             skipped += 1
    print("Skipped: {}".format(skipped))

    with open(args.output, 'w') as wf:
        writer = csv.writer(wf)
        for corrupted, sentence in data:
            writer.writerow([corrupted, sentence])

    if args.save_noised_output is True:
        with open(args.noised_output, 'w') as wf:
            writer = csv.writer(wf)
            for corrupted, sentence in data:
                corrupted = sentence_noising(corrupted)
                writer.writerow([corrupted, sentence])


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

    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--data_name', type=str, required=True, default='muneo',
                        help='name of dataset')

    args = parser.parse_args()

    random.seed(args.seed)

    if args.noised_output is None:
        args.noised_output = args.output + '.0'

    data_preparation(args)
