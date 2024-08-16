import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from osgeo import gdal
from os import listdir

label = listdir("tif/")
label.sort()
img_path = []
X = []
Y = []
for i in range(len(label)):
    images = listdir("tif/"+label[i]+"/")
    images.sort()
    flag = 0
    for j in images :
        if flag >= 100:
            break
        dataset = gdal.Open("tif/"+label[i]+"/"+j)
        im_width = dataset.RasterXSize
        im_height = dataset.RasterYSize
        im_data = dataset.ReadAsArray(0,0,im_width,im_height)
        im_bands = dataset.RasterCount
        temp = np.reshape(im_data,(im_bands*im_height*im_width,))
        img_path.append("2750/"+label[i]+'/'+j)
        X.append(temp)
        Y.append(i)
        flag +=1

n_samples,n_features = X.shape
X = np.asarray(X)
X.shape
Y = np.asarray(Y)
Y

from matplotlib import offsetbox
def plot_embedding(X, Y, label, n_samples, img_path, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)#正则化
    plt.figure(figsize=(30, 30))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(label[Y[i]]),
                 color=plt.cm.Set1(Y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 8})
    #print a colorful font
    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(n_samples):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 10e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            temp = plt.imread(img_path[i].replace('.tif','.jpg'))
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(temp), X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

# PCA
from sklearn.decomposition import PCA
start_time = time.time()
X_pca = PCA(n_components=2).fit_transform(X)
X_pca.shape
end_time = time.time()
plot_embedding(X_pca,Y,label,n_samples,img_path,"Principal Components projection of the digits (time: %.3fs)" % (end_time - start_time))
plt.savefig("pca.jpg")
plt.show()
# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
start_time = time.time()
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X,Y)
X_lda = lda.transform(X)
end_time = time.time()
plot_embedding(X_lda,Y,label,n_samples,img_path,"Linear discriminant analysis of the digits (time: %.3fs)" % (end_time - start_time))
plt.savefig("lda.png")
# kernel PCA
from sklearn.decomposition import KernelPCA
start_time = time.time()
rbf_pca=KernelPCA(n_components=2,kernel='rbf',gamma=0.04)
X_reduced=rbf_pca.fit_transform(X)
end_time = time.time()
plot_embedding(X_reduced,Y,label,n_samples,img_path,"Kernel Principal Components projection of the digits (time: %.3fs)" % (end_time - start_time))
# LLE
from sklearn.manifold import LocallyLinearEmbedding
start_time = time.time()
lle=LocallyLinearEmbedding(n_components=2,n_neighbors=10)
X_reduced=lle.fit_transform(X)
end_time = time.time()
plot_embedding(X_reduced,Y,label,n_samples,img_path,"Locally Linear Embedding of the digits (time: %.3fs)" % (end_time - start_time))
# TSNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, init='pca', random_state=0)
start_time = time.time()
X_reduced=tsne.fit_transform(X)
end_time = time.time()
plot_embedding(X_reduced,Y,label,n_samples,img_path,"TSNE Embedding of the digits (time: %.3fs)" % (end_time - start_time))
# isomap
from sklearn.manifold import Isomap
isomap = Isomap(n_components=2)
start_time = time.time()
X_reduced=isomap.fit_transform(X)
end_time = time.time()
plot_embedding(X_reduced,Y,label,n_samples,img_path,"Isomap Embedding of the digits (time: %.3fs)" % (end_time - start_time))
# LTSA
from sklearn.manifold import LocallyLinearEmbedding
start_time = time.time()
lle=LocallyLinearEmbedding(n_components=2,n_neighbors=3,method='ltsa')
X_reduced=lle.fit_transform(X)
end_time = time.time()
plot_embedding(X_reduced,Y,label,n_samples,img_path,"LTSA Embedding of the digits (time: %.3fs)" % (end_time - start_time))
# Hessian LLE
from sklearn.manifold import LocallyLinearEmbedding
start_time = time.time()
lle=LocallyLinearEmbedding(n_components=2,n_neighbors=10,method='hessian')
X_reduced=lle.fit_transform(X)
end_time = time.time()
plot_embedding(X_reduced,Y,label,n_samples,img_path,"LTSA Embedding of the digits (time: %.3fs)" % (end_time - start_time))
# MDS
from sklearn.manifold import MDS
mds = MDS(n_components=2)
start_time = time.time()
X_reduced=isomap.fit_transform(X)
end_time = time.time()
plot_embedding(X_reduced,Y,label,n_samples,img_path,"MDS Embedding of the digits (time: %.3fs)" % (end_time - start_time))
