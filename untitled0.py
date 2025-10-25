# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix

# Load the Iris dataset from Scikit-learn
iris = datasets.load_iris()
iris_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

# Map numerical targets to species names
iris_df['species'] = iris_df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
iris_df.drop('target', axis=1, inplace=True)

# 1. Load and Inspect the Iris Dataset
print(f"Shape of the data: {iris_df.shape}")
print(f"Type of the data: {type(iris_df)}")
print("First 3 rows:\n", iris_df.head(3))

# 2. Explore Dataset Structure
print("Keys of the iris dataset:", iris.keys())
print("Number of rows-columns:", iris_df.shape)
print("Feature names:", iris.feature_names)
print("Description of the Iris data:\n", iris.DESCR)

# 3. Data Cleaning
print("Number of observations:", len(iris_df))
print("Missing values:", iris_df.isnull().sum().sum())
print("NaN values:", iris_df.isna().sum().sum())

# 4. Create a Sparse Matrix
identity_matrix = np.eye(4)  # Using 4 because the Iris dataset features are 4-dimensional
sparse_matrix = csr_matrix(identity_matrix)
print("Sparse Matrix in CSR format:\n", sparse_matrix)

# 5. Statistical Analysis
print("Statistical details of iris data:\n", iris_df.describe())

# 6. Observations by Species
print("Observations of each species:\n", iris_df['species'].value_counts())

# 7. Modify the DataFrame
# For this example, no 'Id' column to drop as we're working directly with the dataset from sklearn

# 8. Access Specific Cells
print("First four cells (excluding species column):\n", iris_df.iloc[0:4, 0:4])

# Visualization 1: General Statistics Plot
sns.pairplot(iris_df, hue='species')
plt.suptitle("General Statistics of Iris Data", y=1.02)
plt.show()
