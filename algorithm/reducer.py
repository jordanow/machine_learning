# import modules
from sklearn.decomposition import IncrementalPCA as IPCA

# We need to perform dimensional reduction on the given matrix
# For the current use case, we'd use Principal Component Analysis
# We'll use the IncrementalPCA from sklearn.decomposition considering the
# size of the data we're dealing with. It is a much more memory efficient
# method of dimensional reduction than PCA from the same module.

# Only want to keep the top 200 features which have the maximum variance
number_of_features = 200
batch_size = 500


# Dimensionally reduce the given matrix
def dimensional_reduction(matrix, num_features=number_of_features):
    ipca = IPCA(n_components=num_features, batch_size=batch_size)
    ipca.fit(matrix)
    return ipca.transform(matrix)
