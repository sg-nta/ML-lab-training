import os
import re
import numpy as np
from collections import defaultdict
from nltk.stem.porter import PorterStemmer

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
    stemmer = PorterStemmer()
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
                    text = f.read().lower()
                    words = [stemmer.stem(word) for word in re.split("\W+", text) if word not in stop_words]
                    content = ' '.join(words)
                    assert len(content.splitlines()) == 1
                    data.append(str(label) + "<fff>" + filename + "<fff>" + content)
        return data
        
    train_data = collect_data_from(parent_dir = train_dir, newsgroup_list= list_newsgroups)
    test_data = collect_data_from(parent_dir=test_dir, newsgroup_list=list_newsgroups)
    full_data = train_data + test_data
    with open('../datasets/20news-bydate/20news-train-processed.txt','w') as f:
        f.write('\n'.join(train_data))
    with open('../datasets/20news-bydate/20news-test-processed.txt','w') as f:
        f.write('\n'.join(test_data))
    with open('../datasets/20news-bydate/20news-full-processed.txt','w') as f:
        f.write('\n'.join(full_data))

def generate_vocabulary(data_path):
    def compute_idf(df, corpus_size):
        assert df > 0
        return np.log10(corpus_size/df)
    with open(data_path) as f:
        lines = f.read().splitlines()
    doc_count = defaultdict(int)
    corpus_size = len(lines)
    for line in lines:
        features = line.split('<fff>')
        text = features[-1]
        words = list(set(text.split()))
        for word in words:
            doc_count[word] += 1
    words_idfs  = [(word, compute_idf(document_freq, corpus_size)) 
                        for word, document_freq in zip(doc_count.keys(), doc_count.values()) if document_freq > 10 and not word.isdigit()]
    words_idfs.sort(key=lambda word_idf: word_idf[1], reverse=True)
    print("Vocabulary size: {}".format(len(words_idfs)))
    with open("../datasets/20news-bydate/words_idfs.txt",'w') as f:
        f.write('\n'.join([word + "<fff>" + str(idf) for word, idf in words_idfs]))

def get_tf_idf(data_path):
    with open(f'../datasets/20news-bydate/words-idfs.txt') as f:
        words_idfs = [(word, float(idf)) for line in f.read().splitlines() 
                        for word, idf in [line.split('<fff>')]]
        word_IDs = {word: idx for idx, (word, idf) in enumerate(words_idfs)}
        idfs = dict(words_idfs)
    
    with open(data_path) as f:
        documents = [(int(line.split('<fff>')[0]), int(line.split('<fff>')[1]), line.split('<fff>')[2]) 
                        for line in f.read().splitlines()]    

        data_tf_idf = []
        for document in documents:
            label,doc_id,text = document
            words = [word for word in text.split() if word in idfs]
            word_set = list(set(words))
            max_term_freq = max([words.count(word) for word in word_set])
            words_tfidfs = []
            sum_squares = 0
            for word in word_set:
                tf = words.count(word) / max_term_freq
                idf = idfs[word]
                tfidf = tf * idf
                words_tfidfs.append((word_IDs[word], tfidf))
                sum_squares += tfidf ** 2

            words_tfidfs_normalized = [f'{idx}:{tfidf / np.sqrt(sum_squares)}' 
                                        for idx, tfidf in words_tfidfs]
            sparse_rep = ' '.join(words_tfidfs_normalized)
            data_tf_idf.append(f'{label}<fff>{doc_id}<fff>{sparse_rep}')
    
    dataset_name = '-'.join(data_path.split('/')[-1].split('.')[0].split('-')[:-1])
    with open(f'{PATH}/{dataset_name}-tf-idf.txt', 'w') as f:
        f.write('\n'.join(data_tf_idf))
if __name__ == "__main__":
    #gather_20newsgroups_data()
    generate_vocabulary(f'{PATH}/20news-train-processed.txt')
    get_tf_idf(f'{PATH}/20news-train-processed.txt')