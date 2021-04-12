# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 15:46:30 2016

@author: Alonyan
"""
import numpy as np
import scipy.io as sio

def num2str(num, precision): 
    return "%0.*f" % (precision, num)
    
def colorcode(datax, datay):
    from scipy import interpolate
    import numpy as np
    H, xedges, yedges = np.histogram2d(datax,datay, bins=30)
    xedges = (xedges[:-1]+xedges[1:])/2
    yedges = (yedges[:-1]+yedges[1:])/2
    f = interpolate.RectBivariateSpline(xedges,yedges , H)
    
    z = np.array([])
    for i in datax.index:
        z = np.append(z,f(datax[i],datay[i]))
    #z=(z-min(z))/(max(z)-min(z))   
    z[z<0] = 0
    idx = z.argsort()
    return z, idx


class kmeans:        
    def __init__(self, X, K):
        # Initialize to K random centers
        oldmu = X.sample(K).values#np.random.sample(X, K)
        mu = X.sample(K).values#np.random.sample(X, K)
        while not _has_converged(mu, oldmu):
            oldmu = mu
            # Assign all points in X to clusters
            clusters = _cluster_points(X, mu)
            # Reevaluate centers
            mu = _reevaluate_centers(oldmu, clusters)
        self.mu = mu
        self.clusters = clusters
        #return(mu, clusters)
        
    def _cluster_points(X, mu):
        clusters  = {}
        for x in X:
            bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                        for i in enumerate(mu)], key=lambda t:t[1])[0]
            try:
                clusters[bestmukey].append(x)
            except KeyError:
                clusters[bestmukey] = [x]
        return clusters

    def _reevaluate_centers(mu, clusters):
        newmu = []
        keys = sorted(clusters.keys())
        for k in keys:
            newmu.append(np.mean(clusters[k], axis = 0))
        return newmu

    def _has_converged(mu, oldmu):
        return (set(mu) == set(oldmu))


      
def regionprops_to_df(im_props):
    """
    Read content of all attributes for every item in a list
    output by skimage.measure.regionprops
    """
    import pandas as pd

    def scalar_attributes_list(im_props):
        """
        Makes list of all scalar, non-dunder, non-hidden
        attributes of skimage.measure.regionprops object
        """
        attributes_list = []

        for i, test_attribute in enumerate(dir(im_props[0])):

            #Attribute should not start with _ and cannot return an array
            #does not yet return tuples
            if test_attribute[:1] != '_' and not\
                    isinstance(getattr(im_props[0], test_attribute), np.ndarray):                
                attributes_list += [test_attribute]

        return attributes_list


    attributes_list = scalar_attributes_list(im_props)

    # Initialise list of lists for parsed data
    parsed_data = []

    # Put data from im_props into list of lists
    for i, _ in enumerate(im_props):
        parsed_data += [[]]
        
        for j in range(len(attributes_list)):
            parsed_data[i] += [getattr(im_props[i], attributes_list[j])]

    # Return as a Pandas DataFrame
    return pd.DataFrame(parsed_data, columns=attributes_list)


