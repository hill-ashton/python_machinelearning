#This is a mtSNE and UMAP Dimension Reduction lab, originally performed in an online IDE for a coursera program
#For the purposes of archiving and documenting my work, I have rewritten all the code into a format which will work in this codespace
#Most of this code does not function as intended in this environment

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import umap.umap_ as UMAP 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import plotly.express as px
from sklearn.datasets import make_blobs

#generate synthetic data with four clusters in a 3D space
#cluster centers:
centers = [ [ 2, -6, -6],
            [-1,  9,  4],
            [-8,  7,  2],
            [ 4,  7,  9] ]

#cluster standard deviations:
cluster_std=[1,1,2,3.5]

#make the blobs and return the data and the blob labels
X, labels_ = make_blobs(n_samples=500, centers=centers, n_features=3, cluster_std=cluster_std, random_state=42)