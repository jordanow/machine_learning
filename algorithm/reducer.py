# import modules
from sklearn.decomposition import IncrementalPCA as IPCA

# We need to perform dimensional reduction on the given matrix
# For the current use case, we'd use Principal Component Analysis
# We'll use the IncrementalPCA from sklearn.decomposition considering the
# size of the data we're dealing with. It is a much more memory efficient
# method of dimensional reduction than PCA from the same module.


# dimensionally reduce the given matrix
def dimensional_reduction(matrix):
  ipca = IPCA()
  ipca.fit(matrix)
  return ipca.transform(matrix);
