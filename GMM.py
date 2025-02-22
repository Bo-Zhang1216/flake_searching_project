import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your data assuming a CSV file with the columns:
# "flake_R", "flake_G", "flake_B", "background_R", "background_G", "background_B"
data = pd.read_csv('flake_data.csv')

# Create the feature array
features = data[['flake_R', 'flake_G', 'flake_B', 'background_R', 'background_G', 'background_B']].values

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Choose the number of components (clusters) you expect; for example, 3
n_components = 4
gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
gmm.fit(features_scaled)

# Predict cluster labels
labels = gmm.predict(features_scaled)
data['cluster'] = labels

# Print out the cluster means in the scaled space
print("Cluster means (scaled):")
print(gmm.means_)

# Inverse transform the means to get back to the original RGB scale
cluster_means = scaler.inverse_transform(gmm.means_)
print("Cluster means (original scale):")
print(cluster_means)

# Optionally, evaluate the model using BIC to find the optimal number of clusters
bics = []
components_range = range(1, 10)
for n in components_range:
    gmm_test = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
    gmm_test.fit(features_scaled)
    bics.append(gmm_test.bic(features_scaled))

plt.plot(components_range, bics, marker='o')
plt.xlabel("Number of Components")
plt.ylabel("BIC")
plt.title("BIC Scores for GMM")
plt.show()
