import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC, SVC

def load_data(data_path):
    def sparse_to_dense(sparse_r_d, vocab_size):
        r_d = [0.0 for _ in range(vocab_size)]
        indices_tfidfs = sparse_r_d.split()
        for index_tfidf in indices_tfidfs:
            index = int(index_tfidf.split(":")[0])
            tfidf = float(index_tfidf.split(":")[1])
            r_d[index] = tfidf
        return np.array(r_d)
    
    with open(data_path) as f:
        d_lines = f.read().splitlines()
    with open('../datasets/20news-bydate/words_idfs.txt') as f:
        vocab_size = len(f.read().splitlines())
        
    data = np.empty((len(d_lines), vocab_size))
    labels = np.empty(len(d_lines))
    
    for data_id, d in enumerate(d_lines):
        features = d.split("<fff>")
        label = int(features[0])
        r_d = sparse_to_dense(sparse_r_d = features[2], vocab_size = vocab_size)
        data[data_id] = r_d
        labels[data_id] = label
    
    return data, labels

def clustering_with_KMeans():
    X, _ = load_data(data_path='../datasets/20news-bydate/20news-full-tf-idf.txt')
    kmeans = KMeans(
        n_clusters=20,
        init='random',
        n_init=5,
        tol=1e-3,
        verbose = 2,
        random_state=2022,
    )
    kmeans.fit(X)
    print("Stop after ", kmeans.n_iter_," iterations, total distance from members to their clusters: ", kmeans.inertia_ )

def classifying_with_linear_SVMs():
    X_train, y_train = load_data(data_path='../datasets/20news-bydate/20news-train-tf-idf.txt')
    classifier = LinearSVC(
        C=10.0,
        tol=1e-3,
        verbose=False,
    )
    classifier.fit(X_train, y_train)
    
    X_test, y_test = load_data(data_path='../datasets/20news-bydate/20news-test-tf-idf.txt')
    y_predicted  = classifier.predict(X_test)
    accuracy = np.mean(y_predicted == y_test)
    print("Linear SVM accuracy: ", accuracy)

def classifying_with_kernel_SVMs():
    X_train, y_train = load_data(data_path='../datasets/20news-bydate/20news-train-tf-idf.txt')
    classifier = SVC(
        C = 50.0,
        kernel = 'rbf',
        gamma = 0.1,
        tol = 1e-3,
        verbose = True,
    )
    classifier.fit(X_train, y_train)
    
    X_test, y_test = load_data(data_path='../datasets/20news-bydate/20news-test-tf-idf.txt')
    y_predicted  = classifier.predict(X_test)
    accuracy = np.mean(y_predicted == y_test)
    print("Kernel SVM accuracy: ", accuracy)
    
if __name__ == '__main__':
    #classifying_with_linear_SVMs()
    clustering_with_KMeans()