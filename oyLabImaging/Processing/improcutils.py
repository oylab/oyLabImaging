# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 15:46:30 2016

@author: Alonyan
"""

import numpy as np
try:
    from numba import jit
except ImportError:
    def jit(obj, **k):
        return obj


def periodic_smooth_decomp(I: np.ndarray) -> (np.ndarray, np.ndarray):
    """Performs periodic-smooth image decomposition
    Parameters
    ----------
    I : np.ndarray
        [M, N] image. will be coerced to a float.
    Returns
    -------
    P : np.ndarray
        [M, N] image, float. periodic portion.
    S : np.ndarray
        [M, N] image, float. smooth portion.

        Code from: https://github.com/jacobkimmel/ps_decomp
    """
    from pyfftw.interfaces.numpy_fft import fft2, ifft2

    def u2v(u: np.ndarray) -> np.ndarray:
        """Converts the image `u` into the image `v`
        Parameters
        ----------
        u : np.ndarray
            [(L), M, N] image
        Returns
            -------
        v : np.ndarray
            [(L), M, N] image, zeroed expect for the outermost rows and cols
        """

        if u.ndim == 2:
            np.expand_dims(u, 0)

        v = np.zeros(u.shape, dtype=np.float64)

        v[:, 0, :] = np.subtract(u[:, -1, :], u[:, 0, :], dtype=np.float64)
        v[:, -1, :] = np.subtract(u[:, 0, :], u[:, -1, :], dtype=np.float64)

        v[:, :, 0] += np.subtract(u[:, :, -1], u[:, :, 0], dtype=np.float64)
        v[:, :, -1] += np.subtract(u[:, :, 0], u[:, :, -1], dtype=np.float64)
        return v

    def v2s(v_hat: np.ndarray) -> np.ndarray:
        """Computes the maximally smooth component of `u`, `s` from `v`
        s[q, r] = v[q, r] / (2*np.cos( (2*np.pi*q)/M )
            + 2*np.cos( (2*np.pi*r)/N ) - 4)
        Parameters
        ----------
        v_hat : np.ndarray
            [M, N] DFT of v
        """
        L, M, N = v_hat.shape

        q = np.arange(M).reshape(M, 1).astype(v_hat.dtype)
        r = np.arange(N).reshape(1, N).astype(v_hat.dtype)

        den = (
            2 * np.cos(np.divide((2 * np.pi * q), M))
            + 2 * np.cos(np.divide((2 * np.pi * r), N))
            - 4
        )
        s = np.divide(v_hat, den, out=np.zeros_like(v_hat), where=den != 0)
        s[:, 0, 0] = 0
        return s

    u = I.astype(np.float64)
    v = u2v(u)
    v_fft = fft2(v, threads=8)
    s = v2s(v_fft)
    s_i = ifft2(s, threads=8)
    s_f = np.real(s_i)
    p = u - s_f  # u = p + s
    return np.squeeze(p), np.squeeze(s_f)


def perdecomp_3D(u):
    """
    3D version of periodic plus smooth decomposition
    author: AOY
    """
    u = u.astype(np.float64)
    s = np.zeros_like(u)
    nx = s.shape[0]
    ny = s.shape[1]
    nz = s.shape[2]

    b1 = u[-1, :, :] - u[0, :, :]
    b2 = u[:, -1, :] - u[:, 0, :]
    b3 = u[:, :, -1] - u[:, :, 0]

    s[0, :, :] = -b1
    s[-1, :, :] = b1

    s[:, 0, :] = s[:, 0, :] - b2
    s[:, -1, :] = s[:, -1, :] + b2

    s[:, :, 0] = s[:, :, 0] - b3
    s[:, :, -1] = s[:, :, -1] + b3

    fft3_s = np.fft.fftn(s)

    cx = 2.0 * np.pi / nx
    cy = 2.0 * np.pi / ny
    cz = 2.0 * np.pi / nz

    mat_x = np.expand_dims(
        np.concatenate(
            (np.arange(np.round(nx / 2)), np.arange(np.round(nx / 2), 0, -1))
        ),
        (1, 2),
    )
    mat_y = np.expand_dims(
        np.concatenate(
            (np.arange(np.round(ny / 2)), np.arange(np.round(ny / 2), 0, -1))
        ),
        (0, 2),
    )
    mat_z = np.expand_dims(
        np.concatenate(
            (np.arange(np.round(nz / 2)), np.arange(np.round(nz / 2), 0, -1))
        ),
        (0, 1),
    )

    b = 0.5 / (3.0 - np.cos(cx * mat_x) - np.cos(cy * mat_y) - np.cos(cz * mat_z))
    fft3_s = fft3_s * b
    fft3_s[0, 0, 0] = 0
    s = np.real(np.fft.ifftn(fft3_s))
    return u - s, s


def trithresh(pix, nbins=256):
    imhist, edges = np.histogram(pix[:], nbins)
    centers = (edges[1:] + edges[:-1]) / 2

    a = centers[np.argmax(np.cumsum(imhist) / np.sum(imhist) > 0.9999)]  # brightest
    b = centers[np.argmax(imhist)]  # most probable
    h = np.max(imhist)  # response at most probable

    m = h / (b - a)

    x1 = np.arange(0, a - b, 0.1)
    y1 = np.interp(x1 + b, centers, imhist)

    L = (m**2 + 1) * (
        (y1 - h) * (1 / (m**2 - 1)) - x1 * m / (m**2 - 1)
    ) ** 2  # Distance between line m*x+b and curve y(x) maths!

    triThresh = b + x1[np.argmax(L)]
    return triThresh


def awt(I, nBands=None):
    """
    A description of the algorithm can be found in:
    J.-L. Starck, F. Murtagh, A. Bijaoui, "Image Processing and Data
    Analysis: The Multiscale Approach", Cambridge Press, Cambridge, 2000.

    W = AWT(I, nBands) computes the A Trou Wavelet decomposition of the
    image I up to nBands scale (inclusive). The default value is nBands =
    ceil(max(log2(N), log2(M))), where [N M] = size(I).

    Output:
    W contains the wavelet coefficients, an array of size N x M x nBands+1.
    The coefficients are organized as follows:
    W(:, :, 1:nBands) corresponds to the wavelet coefficients (also called
    detail images) at scale k = 1...nBands
    W(:, :, nBands+1) corresponds to the last approximation image A_K.


    Sylvain Berlemont, 2009
    Vectorized version - Alon Oyler-Yaniv, 2018
    python version - Alon Oyler-Yaniv, 2020
    """
    if np.ndim(I) == 2:
        I = np.expand_dims(I, 2)

    [N, M, L] = np.shape(I)

    K = np.ceil(max([np.log2(N), np.log2(M), np.log2(L)]))

    if nBands is None:
        nBands = K
    assert nBands <= K, "nBands must be <= %d" % K

    W = np.zeros((N, M, L, nBands + 1))

    lastA = I.astype("float")

    from numba import jit

    @jit(nopython=True)
    def convx(tmp, k1, k2):
        I = (
            6 * tmp[k2:-k2, :, :]
            + 4 * (tmp[k2 + k1 : -k2 + k1, :, :] + tmp[k2 - k1 : -k2 - k1, :, :])
            + tmp[2 * k2 :, :, :]
            + tmp[0 : -2 * k2, :, :]
        )
        return I

    from numba import jit

    @jit(nopython=True)
    def convy(tmp, k1, k2):
        I = (
            6 * tmp[:, k2:-k2, :]
            + 4 * (tmp[:, k2 + k1 : -k2 + k1, :] + tmp[:, k2 - k1 : -k2 - k1, :])
            + tmp[:, 2 * k2 :, :]
            + tmp[:, 0 : -2 * k2, :]
        )
        return I

    def convolve(I, k):
        k1 = 2 ** (k - 1)
        k2 = 2**k

        tmp = np.pad(I, ((k2, k2), (0, 0), (0, 0)), "edge")
        # Convolve the columns
        # I = 6*tmp[k2:-k2, : , :] + 4*(tmp[k2+k1:-k2+k1, :, :] + tmp[k2-k1:-k2-k1, :, :]) + tmp[2*k2:, :, :] + tmp[0:-2*k2, :, :]
        I = convx(tmp, k1, k2)
        tmp = np.pad(I * 0.0625, ((0, 0), (k2, k2), (0, 0)), "edge")
        # I = 6*tmp[:,k2:-k2, :] + 4*(tmp[:,k2+k1:-k2+k1, :] + tmp[:,k2-k1:-k2-k1, :]) + tmp[:,2*k2:, :] + tmp[:,0:-2*k2, :]
        I = convy(tmp, k1, k2)

        return I * 0.0625

    for k in np.arange(1, nBands + 1):
        newA = convolve(lastA, k)
        W[:, :, :, k - 1] = lastA - newA
        lastA = newA

    W[:, :, :, nBands] = lastA

    return np.squeeze(W)


"""
Python implementation of the imimposemin function in MATLAB.
Reference: https://www.mathworks.com/help/images/ref/imimposemin.html
"""


def imimposemin(I, BW, conn=None, max_value=255):
    import math

    import numpy as np
    from skimage.morphology import ball, cube, disk, reconstruction, square

    if I.ndim not in (2, 3):
        raise Exception("'I' must be a 2-D or 3D array.")

    if BW.shape != I.shape:
        raise Exception("'I' and 'BW' must have the same shape.")

    if BW.dtype is not bool:
        BW = BW != 0

    # set default connectivity depending on whether the image is 2-D or 3-D
    if conn is None:
        if I.ndim == 3:
            conn = 26
        else:
            conn = 8
    else:
        if conn in (4, 8) and I.ndim == 3:
            raise Exception("'conn' is invalid for a 3-D image.")
        elif conn in (6, 18, 26) and I.ndim == 2:
            raise Exception("'conn' is invalid for a 2-D image.")

    # create structuring element depending on connectivity
    if conn == 4:
        selem = disk(1)
    elif conn == 8:
        selem = square(3)
    elif conn == 6:
        selem = ball(1)
    elif conn == 18:
        selem = ball(1)
        selem[:, 1, :] = 1
        selem[:, :, 1] = 1
        selem[1] = 1
    elif conn == 26:
        selem = cube(3)

    fm = I.astype(float)

    try:
        fm[BW] = -math.inf
        fm[np.logical_not(BW)] = math.inf
    except:
        fm[BW] = -float("inf")
        fm[np.logical_not(BW)] = float("inf")

    if I.dtype == float:
        I_range = np.amax(I) - np.amin(I)

        if I_range == 0:
            h = 0.1
        else:
            h = I_range * 0.001
    else:
        h = 1

    fp1 = I + h

    g = np.minimum(fp1, fm)

    # perform reconstruction and get the image complement of the result
    if I.dtype == float:
        J = reconstruction(1 - fm, 1 - g, selem=selem)
        J = 1 - J
    else:
        J = reconstruction(255 - fm, 255 - g, method="dilation", selem=selem)
        J = 255 - J

    try:
        J[BW] = -math.inf
    except:
        J[BW] = -float("inf")

    return J


@jit(nopython=True)
def DownScale(imgin):  # use 2x downscaling for scrol speed
    # imgout = trans.downscale_local_mean(imgin,(Sc, Sc))
    imgout = (
        imgin[0::2, 0::2] + imgin[1::2, 0::2] + imgin[0::2, 1::2] + imgin[1::2, 1::2]
    ) / 4
    return imgout


class segmentation(object):
    """
    class for all segmentation functions
    All functions accept an img and any arguments they need and return a labeled matrix.
    """

    import contextlib
    import logging
    logging.getLogger("cellpose").propagate = False    
    from cellpose import models

    def segmentation_types():
        return [
            "watershed",
            "cellpose_nuclei",
            "cellpose_cyto",
            "stardist_nuclei",
        ]  # 'cellpose_nuc_cyto'

    @contextlib.contextmanager
    def nostdout():
        import io
        import sys

        save_stdout = sys.stdout
        sys.stdout = io.BytesIO()
        yield
        sys.stdout = save_stdout

    def segtype_to_segfun(segment_type):
        if segment_type == "watershed":
            seg_fun = segmentation._segment_nuclei_watershed
        elif segment_type == "cellpose_nuclei":
            seg_fun = segmentation._segment_nuclei_cellpose
        elif segment_type == "cellpose_cyto":
            seg_fun = segmentation._segment_cytoplasm_cellpose
        elif segment_type == "stardist_nuclei":
            seg_fun = segmentation._segment_nuclei_stardist
        elif segment_type == "cellpose_nuc_cyto":
            seg_fun = segmentation._segment_nuccyto_cellpose
        return seg_fun

    def _segment_nuclei_watershed(
        img, imgCyto=[], voronoi=None, cellsize=5, hThresh=0.001
    ):
        from scipy.ndimage.morphology import distance_transform_edt
        from skimage import filters, measure
        from skimage.feature import peak_local_max
        from skimage.morphology import (
            closing,
            dilation,
            disk,
            h_maxima,
        )

        # from skimage.morphology import watershed
        from skimage.segmentation import watershed
        from oyLabImaging.Processing.improcutils import awt

        # wavelet transform and SAR
        W = awt(img, 9)
        img = np.sum(W[:, :, 1:8], axis=2)

        if voronoi is None:
            # Smoothen
            voronoi = {}
            imgSmooth = filters.gaussian(img, sigma=cellsize)
            img_hmax = h_maxima(imgSmooth, hThresh)  # threshold
            coordinates = peak_local_max(img_hmax, footprint=np.ones((30, 30)))
            RegionMax = np.zeros_like(img, dtype=np.bool)
            RegionMax[tuple(coordinates.T)] = True
            RegionMax = RegionMax.astype("int")
            se = disk(cellsize)
            RegionMax = closing(RegionMax, se)
            imgBW = dilation(RegionMax, se)
            dt = distance_transform_edt(1 - imgBW)
            DL = watershed(dt, watershed_line=1)
            RegionBounds = DL == 0  # the region bounds are==0 voronoi cells
            voronoi["imgBW"] = imgBW
            voronoi["RegionBounds"] = RegionBounds

        imgBW = voronoi["imgBW"]
        RegionBounds = voronoi["RegionBounds"]

        # gradient magnitude
        GMimg = filters.sobel(filters.gaussian(img, sigma=cellsize))
        GMimg[np.logical_or(imgBW, RegionBounds)] = 0
        L = watershed(GMimg, markers=measure.label(imgBW), watershed_line=1)

        # We use regionprops
        props = measure.regionprops(L)
        Areas = np.array([r.area for r in props])

        # remove BG region and non-cells
        Areas > 10000
        BG = [i for i, val in enumerate(Areas > 10000) if val]

        if BG:
            for i in np.arange(len(BG)):
                L[L == BG[i] + 1] = 0

        L = measure.label(L)
        return L

    def _segment_nuclei_stardist(
        img,
        imgCyto=[],
        model_name=["2D_versatile_fluo"],
        scale=1,
        prob_thresh=0.5,
        nms_thresh=0.3,
        vmin=1,
        vmax=98,
        **kwargs
    ):
        import logging
        from contextlib import contextmanager
        import sys, os
        from pathlib import Path
        
        


        # heavy guns for getting rid of retracing warnings!
        logging.getLogger("stardist").propagate = False
        logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
        logging.getLogger("tensorflow").propagate = False
        logging.disable(logging.WARNING)
        import warnings

        import tensorflow as tf

        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        
        warnings.filterwarnings("ignore")

        @contextmanager
        def suppress_stdout():
            with open(os.devnull, "w") as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                try:
                    yield
                finally:
                    sys.stdout = old_stdout


        # Patch for csbdeep.models.pretrained not being parallel friendly.
        # The problem comes from calling the function get_file (og from keras) at every model run. 
        # This leads to a new attempt at extracting the model files. When multiple workers try this 
        # it the same time this breaks. The patch is to first check if the file is already there, and 
        # to skip extracting if it is.

        from csbdeep.models import pretrained

        def my_get_folder(cls, key_or_alias):
            key, alias, m = pretrained.get_model_details(cls, key_or_alias)
            cache_dir = os.path.join(os.path.expanduser("~"), ".keras")
            target = str(Path('models') / cls.__name__ / key)

            model_path = Path(cache_dir,target,key+'.zip')

            if os.path.isfile(model_path):
                return model_path.parent
            else:
                path = Path(pretrained.get_file(fname=key+'.zip', origin=m['url'], file_hash=m['hash'],
                                                                    cache_subdir=target, extract=True))
                assert path.exists() and path.parent.exists()
                return path.parent

        pretrained.get_model_folder = my_get_folder

        from csbdeep.utils import normalize
        from stardist.models import StarDist2D
        from tensorflow import convert_to_tensor
        import cv2
        from cv2 import INTER_NEAREST, resize
        from skimage.transform import rescale

        cv2.setNumThreads(1)

        with suppress_stdout():
            model = StarDist2D.from_pretrained(model_name[0])

        img = np.squeeze(img)
        assert img.ndim == 2, "_segment_nuclei_stardist accepts 2D images"

        img = normalize(img, vmin, vmax)
        with suppress_stdout():
            masks, _ = model.predict_instances(
                rescale(img, scale),
                nms_thresh=nms_thresh,
                prob_thresh=prob_thresh,verbose=False,
                **kwargs
            )

        dim = (img.shape[1], img.shape[0])
        # resize masks to original image size using nearest neighbor interpolation to preserve masks
        L = resize(masks, dim, interpolation=INTER_NEAREST)

        return L

    def _segment_nuclei_cellpose(
        img, imgCyto=[], diameter=50, scale=0.5, GPU=True, **kwargs
    ):
        import logging
        logging.getLogger("cellpose").propagate = False
        import warnings

        warnings.filterwarnings("ignore")

        import cv2
        from cellpose import models
        from cv2 import INTER_NEAREST, resize
        from skimage.transform import rescale

        cv2.setNumThreads(2)

        model = models.Cellpose(gpu=GPU, model_type='nuclei')
        img = np.squeeze(img)
        assert img.ndim == 2, "_segment_nuclei_cellpose accepts 2D images"

        masks, _, _, _ = model.eval(
            [rescale(img, scale)],
            diameter=diameter * scale,
            channels=[[0, 0]],
            **kwargs
        )

        dim = (img.shape[1], img.shape[0])
        # resize masks to original image size using nearest neighbor interpolation to preserve masks
        L = resize(masks[0], dim, interpolation=INTER_NEAREST)

        return L

    def _segment_cytoplasm_cellpose(
        img, imgCyto=[], diameter=30, scale=0.5, GPU=True, **kwargs
    ):

        import logging

        logging.getLogger("cellpose").propagate = False
        import warnings

        warnings.filterwarnings("ignore")

        import cv2
        from cellpose import models
        from cv2 import INTER_NEAREST, resize
        from skimage.transform import rescale

        cv2.setNumThreads(2)

        model = models.Cellpose(gpu=GPU, model_type="cyto")

        imgNucCyto = rescale(
            np.concatenate(
                (np.expand_dims(img, 2), np.expand_dims(imgCyto, 2)), axis=2
            ),
            (scale, scale, 1),
        )

        masks, _, _, _ = model.eval(
            [imgNucCyto], diameter=diameter, channels=[[2, 1]], **kwargs
        )

        dim = (img.shape[1], img.shape[0])
        # resize masks to original image size using nearest neighbor interpolation to preserve masks

        return resize(masks[0], dim, interpolation=INTER_NEAREST)

    def _segment_nuccyto_cellpose(
        img,
        imgCyto=[],
        diameter_nuc=20,
        diameter_cyto=30,
        scale=0.5,
        GPU=True,
        **kwargs
    ):
        Lnuc = segmentation._segment_nuclei_cellpose(
            img, diameter=diameter_nuc, scale=scale, GPU=GPU, **kwargs
        )
        Lcyto = segmentation._segment_cytoplasm_cellpose(
            img, imgCyto, diameter=diameter_cyto, scale=scale, GPU=GPU, **kwargs
        )

        Lcyto_new = np.zeros_like(Lcyto)

        for i in np.arange(1, np.max(Lnuc) + 1):
            ind_in_cyto = np.median(Lcyto[Lnuc == i])
            if ind_in_cyto:
                Lcyto_new[Lcyto == ind_in_cyto] = i
            else:
                Lnuc[Lnuc == i] = 0

        # Lcyto_new = Lcyto_new-Lnuc;
        return Lnuc, Lcyto_new

    def test_segmentation_params(
        img, imgCyto=[], segfun=None, segment_type="watershed", **kwargs
    ):
        """
        Function to test different segmentation parameters. Will create widget where parameters can be modified and tested.
        Parameters
        ----------
        img : image of nuclear channel
        imgCyto : [None] image of cytoplasm channel, optional. Only used for certain models.
        kwargs : optional. Additional arguments for specific functions.

        Returns
        -------
        function handle for track generator
        """
        import ipywidgets as widgets
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import ListedColormap
        
        if segfun is None:
            segfun = segmentation.segtype_to_segfun(segment_type)
            print("\nusing " + segfun.__name__)

        if img.ndim == 3:
            img = np.squeeze(img[0, :, :])

        nargs = segfun.__code__.co_argcount
        defaults = list(segfun.__defaults__)
        # ndefault = len(defaults)
        nimgs = 2  # nargs-ndefault
        args = [segfun.__code__.co_varnames[i] for i in range(nimgs, nargs)]
        input_dict = {args[i]: defaults[i + 1] for i in range(len(args))}
        input_dict = {**input_dict, **kwargs}
        L = segfun(img, imgCyto, **input_dict)

        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.1)

        ax.imshow(
            img, cmap="gray", vmin=np.percentile(img, 10), vmax=np.percentile(img, 95)
        )
        cmap = ListedColormap(np.random.rand(256, 3))
        cmap.colors[0, :] = [0, 0, 0]

        l = ax.imshow(L, alpha=0.5, cmap=cmap)

        def clean_dict(d):
            result = {}
            for key, value in d.items():
                if not value.isnumeric():
                    value = eval(value)
                else:
                    value = float(value)
                result[key] = value
            return result

        def update_mask(b):
            print("calculating with new parameters")
            # new_input_dict = {args[i]: eval('text_box_' + str(i)+ '.value') for i in range(0, nargs-1)}
            new_input_dict = {
                tblist[i].description: tblist[i].value for i in range(0, len(tblist))
            }
            new_input_dict = clean_dict(new_input_dict)
            box.input_dict = new_input_dict
            L = segfun(img, imgCyto, **new_input_dict)
            l.set_data(L)
            l.set_alpha(0.5)
            cmap = ListedColormap(np.random.rand(256, 3))
            cmap.colors[0, :] = [0, 0, 0]
            l.set_cmap(cmap)
            l.set_clim([0, np.max(L)])
            plt.draw()
            plt.show()
            print("Done!")

        num = 0
        tblist = []
        for arg, de in input_dict.items():
            exec(
                "text_box_"
                + str(num)
                + "=widgets.Text(value=str(de), description=str(arg))"
            )
            tblist.append(eval("text_box_" + str(num)))
            num += 1
        exButton = widgets.Button(description="Segment cells!")
        exButton.on_click(update_mask)

        box = widgets.HBox([widgets.VBox(tblist), exButton])
        box.input_dict = input_dict
        return box


def squarify(M, val=np.nan):
    (a, b) = M.shape
    if a > b:
        padding = ((0, 0), (int(np.floor((a - b) / 2)), int(np.ceil((a - b) / 2))))
    else:
        padding = ((int(np.floor((b - a) / 2)), int(np.ceil((b - a) / 2))), (0, 0))
    return np.pad(M, padding, mode="constant", constant_values=val)


class Zernike(object):
    def coeff(img, n=8):
        from zernike import RZern

        cart = RZern(n)
        L, K = img.shape
        ddx = np.linspace(-1.0, 1.0, K)
        ddy = np.linspace(-1.0, 1.0, L)
        xv, yv = np.meshgrid(ddx, ddy)
        cart.make_cart_grid(xv, yv)

        c1 = cart.fit_cart_grid(img)[0]
        return c1, L, K

    def coeff_fast(
        img, n=8
    ):  # this is a faster implementation of zernike taken from the poppy package
        from poppy.zernike import decompose_opd_nonorthonormal_basis

        from oyLabImaging.Processing.improcutils import squarify

        L, K = img.shape
        img = squarify(img)
        c1 = decompose_opd_nonorthonormal_basis(
            img,
            aperture=~np.isnan(img),
            nterms=int((n + 1) * (n + 2) / 2),
            iterations=1,
        )
        return c1, L, K

    def reconstruct(c1, L, K, n=8):
        from zernike import RZern

        cart = RZern(n)

        ddx = np.linspace(-1.0, 1.0, K)
        ddy = np.linspace(-1.0, 1.0, L)
        xv, yv = np.meshgrid(ddx, ddy)
        cart.make_cart_grid(xv, yv)

        return cart.eval_grid(c1[0 : cart.nk], matrix=True)


##Filters


def _prepare_frequency_map(pix):
    nx = pix.shape[0]
    ny = pix.shape[1]

    cx = 2.0 * np.pi / nx
    cy = 2.0 * np.pi / ny

    mat_x = np.expand_dims(
        np.concatenate(
            (np.arange(np.round(nx / 2), 0, -1), np.arange(np.round(nx / 2)))
        ),
        1,
    )
    mat_y = np.expand_dims(
        np.concatenate(
            (np.arange(np.round(ny / 2), 0, -1), np.arange(np.round(ny / 2)))
        ),
        0,
    )
    f = np.sqrt(cx * mat_x**2 + cy * mat_y**2)
    return f


def log_filter(p, sigma=5):
    # p,s =  periodic_smooth_decomp(p)

    img_fft = np.fft.fft2(p)
    img_fft = np.fft.fftshift(img_fft)

    f = _prepare_frequency_map(p)
    kernel = np.exp(-(sigma * sigma * (f**2)) / (2 * (2 * np.pi**2) ** 2)) * (
        f**2
    )
    img_ridges = np.real(np.fft.ifft2(np.fft.ifftshift(img_fft * kernel)))
    return img_ridges


def gaussian_filter(p, sigma=10):
    # p,s =  periodic_smooth_decomp(p)

    img_fft = np.fft.fft2(p)
    img_fft = np.fft.fftshift(img_fft)

    f = _prepare_frequency_map(p)

    kernel = np.exp(-(sigma * sigma * (f**2)) / (2 * (2 * np.pi**2) ** 2))
    img_smooth = np.real(np.fft.ifft2(np.fft.ifftshift(img_fft * kernel)))
    return img_smooth


def sample_stack(img, N=1000):
    rng = np.random.default_rng()
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    ints = rng.integers([[0]] * N, img.shape)
    return img[ints[:, 0], ints[:, 1], ints[:, 2]]
