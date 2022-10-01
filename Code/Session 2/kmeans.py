import numpy as np
from collections import defaultdict

class Member:
    def __init__(self, r_d, label=None, doc_id=None):
        self._r_d = r_d
        self._label = label
        self._doc_id = doc_id


class Cluster:
    def __init__(self):
        self._centroid = None
        self._members = []
    
    def reset_members(self):
        self.members = []
    
    def add_member(self, member):
        self._members.append(member)
    
class Kmeans:
    def __init__(self, num_clusters):
        self._num_clusters = num_clusters
        self._clusters = [Cluster() for _ in range(self._num_clusters)]
        self._E = [] #list of centroid
        self._S = 0 #overall similarity
        
    def load_data(self, data_path):
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
        self._data = []
        self._label_count = defaultdict(int)
        for data_id, d in enumerate(d_lines):
            features = d.split("<fff>")
            label, doc_id = int(features[0]), int(features[1])
            self._label_count[label] += 1
            r_d = sparse_to_dense(sparse_r_d = features[2], vocab_size = vocab_size)
            self._data.append(Member(r_d = r_d, label = label, doc_id = doc_id))
        
        
    def random_init(self, seed_value):
        np.random.seed(seed_value)
        centroid_index = np.random.choice(len(self._data),
                                        size=self._num_clusters, 
                                        replace=False)
        for i in range(self._num_clusters):
            self._clusters[i]._centroid = self._data[centroid_index[i]]._r_d
        
    
    def compute_similarity(self, member, centroid):
        cosine_similarity = (member._r_d @ centroid) / ((np.sqrt(np.sum(member._r_d ** 2))) * ((np.sqrt(np.sum(centroid ** 2)))))
        return cosine_similarity
    
    def select_cluster_for(self, member):
        best_fit_cluster = None
        max_similarity = -1
        for cluster in self._clusters:
            similarity = self.compute_similarity(member, cluster._centroid)
            if similarity > max_similarity:
                best_fit_cluster = cluster
                max_similarity = similarity
        best_fit_cluster.add_member(member)
        return max_similarity
    
    def update_centroid_of(self, cluster):
        member_r_ds = [member._r_d for member in cluster._members]
        aver_r_d = np.mean(member_r_ds, axis = 0)
        sqrt_sum_sqr = np.sqrt(np.sum(aver_r_d ** 2))
        new_centroid = np.array([value / sqrt_sum_sqr for value in aver_r_d])
        cluster._centroid = new_centroid
    
    def stopping_condition(self, criterion, threshold):
        criteria = ['centroid', 'similarity', 'max_iters']
        assert criterion in criteria
        if criterion == 'max_iters':
            return self._iteration >= threshold
        
        elif criterion == 'centroid':
            E_new = [list(cluster._centroid) for cluster in self._clusters]
            E_new_minus_E = [centroid for centroid in E_new
                             if centroid not in self._E]
            self._E = E_new
            return len(E_new_minus_E) <= threshold
        else:
            new_S_minus_S = self._new_S - self._S
            self._S = self._new_S
            return new_S_minus_S <= threshold
            
    def run(self, seed_value, criterion, threshold):
        self.random_init(seed_value)
        
        self._iteration = 0
        while True:
            for cluster in self._clusters:
                cluster.reset_members()
            self._new_S = 0
            for member in self._data:
                max_s = self.select_cluster_for(member)
                self._new_S += max_s
            for cluster in self._clusters:
                self.update_centroid_of(cluster)
            
            self._iteration += 1
            print("Iteration", self._iteration, ", Overall similarity: ", self._S)
            if self.stopping_condition(criterion, threshold):
                break
                
                
    def compute_purity(self):
        majority_sum = 0
        for cluster in self._clusters:
            member_labels = [member._label for member in cluster.members]
            max_count = max([member_labels.count(label) for label in range(20)])
            majority_sum += max_count
        return majority_sum / len(self._data)
    
    def compute_NMI(self):
        I_value, H_omega, H_c, N = 0, 0, 0, len(self._data)
        for cluster in self._clusters:
            member_labels = [member._label for member in cluster._members]
            for label in set(member_labels):
                wk_cj = member_labels.count(label)
                wk = len(cluster._members)
                cj = self._label_count[label]
                I_value += (wk_cj / N) * np.log10(N * wk_cj / (wk * cj))
        
        for cluster in self._clusters:
            wk = len(cluster._members)
            H_omega += (wk / N) * np.log10(N / wk)
        
        for label in set(self._label_count):
            cj = self._label_count[label]
            H_c += (cj / N) * np.log10(N / cj)

        return I_value * 2 / (H_omega + H_c)
    

if __name__ == '__main__':
    kmeans = Kmeans(num_clusters=20)
    kmeans.load_data('../datasets/20news-bydate/20news-full-tf-idf.txt')
    kmeans.run(seed_value=1, criterion='similarity', threshold=1e-4)
    print(kmeans.compute_purity())
    print(kmeans.compute_NMI())