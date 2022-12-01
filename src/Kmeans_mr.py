"""
task:Kmeans on MapReduce

input files:
    datas.txt       containment: index, category, 3-dimension data
    Centroids.txt   containment: 3-dimension centroids in a line, initialized at random

"""

import numpy as np
import mrjob
from mrjob.job import MRJob
from mrjob.step import MRStep

class KMean(MRJob):
    OUTPUT_PROTOCOL = mrjob.protocol.RawProtocol

    def configure_options(self):
        super(KMean, self).configure_options()

        #self.add_file_option('--infile')
        #self.add_file_option('--outfile')
        #self.add_passthrough_option('--iterations', dest='iterations', default=10, type='int')


    def read_centroids(self):
        centroids = np.loadtxt('./Centroid.txt', delimiter=',')
        return centroids

    def write_centroids(self, centroids):
        np.savetxt('./output.txt', centroids, fmt='%.5f', delimiter=',')

    def update_category(self, _, line):
        """
        Mapper function
            calculates distances from items to centroids

            Input: txt, in the format of "ID|category|features(dim=3, delimiter=',')"
            Output: new categories of items and their indexes and features
        """
        data_index, category, features = line.split('|')
        features = features.strip('\r\n')
        features = np.array(features.split(','), dtype=float)
        global centroids
        centroids = self.read_centroids()
        centroids = np.reshape(centroids, (-1, len(features)))
        global category_num
        category_num = centroids.shape[0]
        global feature_dim
        feature_dim = centroids.shape[1]
        dist = ((centroids - features)**2).sum(axis=1)
        new_category = str(dist.argmin() + 1)
        feature_list = features.tolist()
        yield new_category, (data_index, feature_list)

    def get_new_centroids(self, category, items):
        """
        Combiner function

            calculates the sum of features of the items in the same category
        """
        indexes = []
        features = []
        feature_sum = np.zeros(feature_dim)
        for index, feature in items:
            features.append(','.join(str(e) for e in feature))
            feature = np.array(feature, dtype=float)
            indexes.append(index)
            feature_sum += feature
            feature_sum = feature_sum.tolist()
        yield category, (indexes, feature_sum, features)

    def update_centroids(self, category, items):
        """
        Reducer function

            calculates features of new centroids, which equal to the averages of features in the same category
            updates centroids
            writes into Centroids.txt
        """
        indexes = []
        features = []
        feature_sum = np.zeros(feature_dim)
        global centroids
        for index, f_sum, fs in items:
            features += fs
            f_sum = np.array(f_sum, dtype=float)
            indexes += index
            feature_sum += f_sum
        
        curr_centroids = feature_sum / len(indexes)
        centroids[feature_dim * (int(category) - 1) : feature_dim * (int(category))] = curr_centroids
        if int(category) == category_num:
            #print(centroids)
            centroids = np.reshape(centroids,(1,-1))
            self.write_centroids(centroids)
        
        for index in indexes:
            idx = indexes.index(index)
            yield None, (index + '|' + category + '|' + features[idx])

    def steps(self):
        #return [MRStep(mapper=self.update_category, combiner=self.get_new_centroids, reducer=self.update_centroids)] * self.options.iterations
	    return [MRStep(mapper=self.update_category, combiner=self.get_new_centroids, reducer=self.update_centroids)] * 5

def main():
    KMean.run()

if __name__ == '__main__':
    main()
