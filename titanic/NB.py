import collections
class NAIVE_BAYES(object):

    def __init__(self, dataset, labels, b=1e-8) -> None:
        super().__init__()
        self.dataset = dataset
        self.labels = labels
        self.bias = b
        self.labels_size = len(labels)
        self.x_dim = len(dataset[0]) - 1
        self.prior_proba = {}
        self.con_proba = {}
        self.labels_dict = collections.Counter(self.labels)
        self.prior_proba = self.prior_probability()
        self.con_proba = self.conditional_proability()

    def prior_probability(self):
        for label in self.labels_dict:
          self.prior_proba[label] = float((self.labels_dict[label] + self.bias) / (self.labels_size + self.bias * len(self.labels_dict)))
        return self.prior_proba

    def conditional_proability(self):
        dataset_c = self.dataset[:]
        for label_i in self.labels_dict:
            label_cp = []
            data_label_i = []
            index = 0
            while index < len(dataset_c):
                if dataset_c[index][-1] == label_i:
                    data_label_i.append(dataset_c[index])
                    dataset_c.pop(index)
                else:
                    index += 1
            for i in range(self.x_dim):
                x_i = [data[i] for data in data_label_i]
                x_i_dict = collections.Counter(x_i)
                for x in x_i_dict:
                    x_i_dict[x] = float((x_i_dict[x] + self.bias) / (self.labels_dict[label_i] + self.bias * len(x_i_dict)))
                label_cp.append(x_i_dict)
            self.con_proba[label_i] = label_cp
        print(self.con_proba)
        return self.con_proba

    def predict(self, x):
        pre_proba = {}
        for label in self.labels_dict:
            proba = self.prior_proba[label]
            for i in range(self.x_dim):
                proba *= self.con_proba[label][i][x[i]]
            pre_proba[label] = proba
        pre_proba_s = sorted(pre_proba.items(), key=lambda d: d[1], reverse=True)
        return pre_proba_s[0]


def load_data(filename):
    dataset = []
    labels = []
    with open(file=filename, mode='r') as f:
        for line in f:
            data = [i for i in line.strip().split()]
            dataset.append(data)
            labels.append(data[-1])
    return dataset, labels

if __name__ == '__main__':
    dataset, labels = load_data('')
    NB = NAIVE_BAYES(dataset, labels)
    print(NB.predict(['2', 'S']))

