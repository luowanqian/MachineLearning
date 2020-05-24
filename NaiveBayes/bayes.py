import numpy as np


class NaiveBayes:
    def _create_vocabulary(self, data):
        vocab_set = set()
        for record in data:
            vocab_set = vocab_set | set(record)

        return list(vocab_set)

    def _words_to_vec(self, vocabulary, record):
        vec = np.zeros(len(vocabulary))
        for word in record:
            if word in vocabulary:
                vec[vocabulary.index(word)] = 1
            else:
                print('The word {} is not in vocabulary.'.format(word))

        return vec

    def _create_matrix(self, vocabulary, data):
        mat = np.zeros((len(data), len(vocabulary)))
        for i, record in enumerate(data):
            mat[i, :] = self._words_to_vec(vocabulary, record)

        return mat

    def fit(self, data, label):
        unique_label = list(set(label))
        num_classes = len(unique_label)
        vocabulary = self._create_vocabulary(data)

        train_matrix = self._create_matrix(vocabulary, data)
        num_records, num_words = train_matrix.shape
        prob_matrix = np.ones((num_classes, num_words))
        prob_class = np.zeros(num_classes)
        words_num = np.ones(num_classes) + 1.0
        for i in range(num_records):
            label_idx = unique_label.index(label[i])
            prob_class[label_idx] += 1
            prob_matrix[label_idx, :] += train_matrix[i]
            words_num[label_idx] += np.sum(train_matrix[i])

        for i in range(num_classes):
            prob_matrix[i, :] /= words_num[i]
        prob_class /= num_records
        prob_matrix = np.log(prob_matrix)

        self.unique_label = unique_label
        self.prob_class = prob_class
        self.prob_matrix = prob_matrix
        self.vocabulary = vocabulary

    def predict(self, data):
        num_classes = len(self.unique_label)
        prob_vec = np.zeros(num_classes)
        test_matrix = self._create_matrix(self.vocabulary, data)

        for record in data:
            for i in range(num_classes):
                pass