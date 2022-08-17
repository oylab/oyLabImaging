# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 15:46:30 2016

@author: Alonyan
"""
import numpy as np


def num2str(num, precision):
    return "%0.*f" % (precision, num)


def colorcode(datax, datay):
    import numpy as np
    from scipy import interpolate

    H, xedges, yedges = np.histogram2d(datax, datay, bins=30)
    xedges = (xedges[:-1] + xedges[1:]) / 2
    yedges = (yedges[:-1] + yedges[1:]) / 2
    f = interpolate.RectBivariateSpline(xedges, yedges, H)

    z = np.array([])
    for ix, iy in zip(datax, datay):
        z = np.append(z, f(ix, iy))
    # z=(z-min(z))/(max(z)-min(z))
    z[z < 0] = 0
    idx = z.argsort()
    return z, idx


class kmeans:
    def __init__(self, X, K):
        # Initialize to K random centers
        oldmu = X.sample(K).values  # np.random.sample(X, K)
        mu = X.sample(K).values  # np.random.sample(X, K)
        while not kmeans._has_converged(mu, oldmu):
            oldmu = mu
            # Assign all points in X to clusters
            clusters = kmeans._cluster_points(X, mu)
            # Reevaluate centers
            mu = kmeans._reevaluate_centers(oldmu, clusters)
        self.mu = mu
        self.clusters = clusters
        # return(mu, clusters)

    @classmethod
    def _cluster_points(cls, X, mu):
        clusters = {}
        for x in X:
            bestmukey = min(
                [(i[0], np.linalg.norm(x - mu[i[0]])) for i in enumerate(mu)],
                key=lambda t: t[1],
            )[0]
            try:
                clusters[bestmukey].append(x)
            except KeyError:
                clusters[bestmukey] = [x]
        return clusters

    @classmethod
    def _reevaluate_centers(cls, mu, clusters):
        newmu = []
        keys = sorted(clusters.keys())
        for k in keys:
            newmu.append(np.mean(clusters[k], axis=0))
        return newmu

    @classmethod
    def _has_converged(cls, mu, oldmu):
        return set(mu) == set(oldmu)


def InAxes(ax=None):
    # find indexes of plot points which are inside axes rectangle
    # by default works on the current axes, otherwise give an axes handle
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    h = ax.get_children()

    Xlim = ax.get_xlim()
    Ylim = ax.get_ylim()
    for hi in h:
        try:
            offs = hi.get_offsets().data
            J = np.where(
                (offs[:, 0] > Xlim[0])
                * (offs[:, 0] < Xlim[1])
                * (offs[:, 1] > Ylim[0])
                * (offs[:, 1] < Ylim[1])
            )
            J = J[0]
        except:
            continue
    return J


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

            # Attribute should not start with _ and cannot return an array
            # does not yet return tuples
            if test_attribute[:1] != "_" and not isinstance(
                getattr(im_props[0], test_attribute), np.ndarray
            ):
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


# Python 3 program to find the stem
# of given list of words

# function to find the stem (longest
# common substring) from the string array


def findstem(arr):
    # Determine size of the array
    n = len(arr)
    # Take first word from array
    # as reference
    s = arr[0]
    l = len(s)
    res = ""
    for i in range(l):
        for j in range(i + 2, l + 1):  # lenth at least 2
            # generating all possible substrings
            # of our reference string arr[0] i.e s
            stem = s[i:j]
            k = 1
            for k in range(1, n):
                # Check if the generated stem is
                # common to all words
                if stem not in arr[k]:
                    break
            # If current substring is present in
            # all strings and its length is greater
            # than current result
                if k + 1 == n and len(res) < len(stem):
                    res = stem
    return res


def findregexp(fnames):
    # use first name as a template
    baseStr = [fnames[0]]
    arr = baseStr + fnames

    # Function call
    stem = findstem(arr)
    stems = []
    while stem:
        # keep finding stems and replacing them in the template with stars. Can probably add an *ignore stars* to the stem finder
        baseStr = [baseStr[0].replace(stem, "*")]
        arr = baseStr + [s.replace(stem, "") for s in arr[1:]]
        # arr = baseStr+fnames
        stems.append(stem)
        stem = findstem(arr)

    # make a list of stars as lont as the OG template
    stars = ["*"] * len(fnames[0])
    # replace all stems in the list
    for s in stems:
        stars[fnames[0].find(s) : fnames[0].find(s) + len(s)] = s
    # remove repeating *s
    superStars = []
    superStars.append(stars[0])
    for s in stars[1:]:
        if not s == "*":
            superStars.append(s)
        elif not superStars[-1] == s:
            superStars.append(s)
    globExp = "".join(superStars)

    return globExp


def extractFieldsByRegex(globExp, fnames):
    import re

    matches = []
    for f in fnames:
        match = re.findall(
            globExp.replace("\\", "/").replace("*", "(.*)"), f.replace("\\", "/")
        )
        if not len(match) == 1:
            print("Non unique matches from regexp")
            break
        if type(match[0]) is not tuple:
            match[0] = (match[0],)
        matches.append(match[0])
    return matches


def alias(aliases):
    import functools

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for alias, name in aliases.items():
                if name not in kwargs and alias in kwargs:
                    kwargs[name] = kwargs[alias]
                    del kwargs[alias]
            return func(*args, **kwargs)

        return wrapper

    return decorator



# import functools
# from typing import Callable, Dict, TypeVar

# T = TypeVar("T", bound=Callable)

# def alias(aliases: Dict[str, str]) -> Callable[[T], T]:
#     print('1')
#     def decorator(func: T) -> T:

#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#             for alias, name in aliases.items():
#                 if val := kwargs.pop(alias, None):
#                     kwargs.setdefault(name, val)
#             return func(*args, **kwargs)

#         return wrapper

#     return decorator
