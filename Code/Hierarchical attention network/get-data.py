import os
import re
import numpy as np
from collections import defaultdict
from spacy.lang.en import English 

nlp = English()
nlp.add_pipe('sentencizer')


PATH = '../datasets/20news-bydate'

def gather_20newsgroups_data():
    dirs = [f'{PATH}/{d}/' for d in os.listdir(PATH)
            if os.path.isdir(f'{PATH}/{d}')]
    train_dir, test_dir = (
        dirs[0], dirs[1]) if 'train' in dirs[0] else (dirs[1], dirs[0])
    list_newsgroups = [newsgroup for newsgroup in os.listdir(train_dir)]
    list_newsgroups.sort()
    with open('../datasets/20news-bydate/stop_word.txt') as f:
        stop_words = f.read().splitlines()
    def collect_data_from(parent_dir, newsgroup_list):
        data = []
        for group_id, newsgroup in enumerate(newsgroup_list):
            label = group_id
            dir_path = f'{parent_dir}/{newsgroup}'
            files = [(filename, f'{dir_path}/{filename}') 
                        for filename in os.listdir(dir_path) 
                            if os.path.isfile(f'{dir_path}/{filename}')]
            files.sort()
            for filename, filepath in files:
                with open(filepath) as f:
                    contents = f.read().lower()
                    contents = nlp(contents)
                    list_sentences = [str(sent).strip() for sent in contents.sents]
                    words = [word for word in [re.split(r'\W+', sent) for sent in list_sentences] if word not in stop_words]
                    content = '<fff>'.join([' '.join(sent) for sent in words])
                    assert len(content.splitlines()) == 1
                    data.append(str(label) + "<fff>" + filename + "<fff>" + content)
        return data
        
    train_data = collect_data_from(parent_dir = train_dir, newsgroup_list= list_newsgroups)
    test_data = collect_data_from(parent_dir=test_dir, newsgroup_list=list_newsgroups)
    with open('../datasets/processed-attention/20news-train-processed.txt','w') as f:
        f.write('\n'.join(train_data))
    with open('../datasets/processed-attention/20news-test-processed.txt','w') as f:
        f.write('\n'.join(test_data))

def encode_data(data_path, vocab_path):
    unknown_ID = 1
    padding_ID = 0
    MAX_WORDS = 50
    MAX_SENTENCES = 30
    with open(vocab_path) as f:
        vocab = dict([(word, word_ID + 2)
                      for word_ID, word in enumerate(f.read().splitlines())])
    with open(data_path) as f:
        documents = [(line.split('<fff>')[0], line.split('<fff>')[1], line.split('<fff>')[2:2+MAX_SENTENCES])
                     for line in f.read().splitlines()]
    
    encoded_data = []
    for document in documents:
        label, doc_id, contents = document 
        encoded_data = []
        for sentence in contents:
            words = sentence.split()[:MAX_WORDS] 
            sentence_length = len(words)
            encoded_sentence = []
            for word in words:
                if word in vocab:
                    encoded_sentence.append(str(vocab[word]))
                else:
                    encoded_sentence.append(str(unknown_ID))
        
            if sentence_length < MAX_WORDS:
                for i in range(MAX_WORDS - sentence_length):
                    encoded_sentence.append(str(padding_ID))
                    
            encoded_data.append(' '.join(encoded_sentence))
            
        if len(contents) < MAX_SENTENCES:
            for i in range(MAX_SENTENCES - len(contents)):
                encoded_data.append(' '.join([str(padding_ID)] * MAX_WORDS))

        encoded_data.append(str(label) + '<fff>' + str(doc_id) + '<fff>' + '<fff>'.join(encoded_data))
    
    dir_name = '/'.join(data_path.split('/')[:-1])
    filename = '-'.join(data_path.split('/')[-1].split('-')[:-1]) + '-encoded.txt'
    with open(dir_name + '/' + filename, 'w') as f:
        f.write('\n'.join(encoded_data))

if __name__ == '__main__':

    encode_data('../datasets/processed-attention/20news-train-processed.txt', '../datasets/w2v/vocab-raw.txt')
    encode_data('../datasets/processed-attention/20news-test-processed.txt', '../datasets/w2v/vocab-raw.txt')
    