"""
Spatial correlation analysis for PosLbl objects.

All functions are computed on-the-fly and return plain dicts — nothing is
persisted to disk.  Pass a specific frame index for fast single-timepoint
analysis, or None to average g(r) across all frames (each frame treated as an
independent replicate, then combined with pair-count weighting).

Usage
-----
from oyLabImaging.Processing.spatial import radial_corr, radial_density, plot_radial

# Cross-correlation between two channels
result = radial_corr(pos, 'GFP', 'DeepBlue', max_r=150, dr=5)

# Autocorrelation of a single channel
result = radial_corr(pos, 'GFP', max_r=150, dr=5)

# Pair correlation function (spatial density g(r))
result = radial_density(pos, frame=0, max_r=200, dr=5)

# Pixel-level autocorrelation from raw image (FFT-based)
result = radial_corr(pos, 'GFP', img=True, max_r=50, dr=0.5)

# Plot any result (or a list of results for comparison)
ax = plot_radial(result)
ax = plot_radial([res1, res2], labels=['ctrl', 'treated'])
"""
from __future__ import annotations

from typing import Optional, Union, List
import warnings

import numpy as np
from scipy.spatial import cKDTree


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def radial_corr(
    pos,
    ch_i: str,
    ch_j: Optional[str] = None,
    frame: Optional[Union[int, List[int]]] = None,
    img: bool = False,
    ffield: bool = True,
    max_r: Optional[float] = None,
    dr: Optional[float] = None,
    intensity: str = 'mean',
    periring: bool = False,
    n_max: int = 200_000,
    seed: int = 42,
) -> dict:
    """Radial cross/auto-correlation g(r) between two channels.

    Two modes:

    ``img=False`` (default) — cell-level
        For each ordered pair of cells (A, B) separated by distance r,
        computes the Pearson correlation between ch_i at A and ch_j at B,
        binned by r.  Intensities are per-cell means (or other metrics).

    ``img=True`` — pixel-level (FFT)
        Uses the full intensity field rather than per-cell means, giving
        sub-cell spatial resolution.  Boundary artefacts are handled via
        periodic-plus-smooth decomposition (Moisan 2011).

    Parameters
    ----------
    pos : PosLbl
    ch_i : str
        Source channel name.
    ch_j : str, optional
        Target channel name.  Defaults to ch_i (autocorrelation).
    frame : int, list of int, or None
        Index into pos.framelabels.  None averages g(r) over all frames.
    img : bool
        False = cell-level correlation; True = pixel-level FFT correlation.
    ffield : bool
        Flat-field correction flag.
        - img=False : checks whether the underlying FrameLbl was segmented
          with ffield=True.  Warns if not (cannot retroactively correct).
        - img=True  : applies flat-field correction when loading images.
          Warns if no flat field has been calculated.
    max_r : float, optional
        Maximum radius in micrometers.  Defaults to 200 (cell) or 250 (img).
    dr : float, optional
        Radial bin width in micrometers.  Defaults to 5 (cell) or 0.5 (img).
    intensity : str
        Per-cell intensity metric: 'mean', 'median', 'max', 'min', 'ninety'.
        Only used when img=False.
    periring : bool
        Use perinuclear-ring intensity.  Only used when img=False.
    n_max : int
        Maximum pairs per bin (random subsampling).  Only used when img=False.
    seed : int
        Random seed for subsampling.  Only used when img=False.

    Returns
    -------
    dict
        r    : (B,) bin centres in µm
        g    : (B,) mean Pearson correlation per bin
        sem  : (B,) standard error of the mean
        n    : (B,) pair count per bin
        ch_i, ch_j, max_r, dr, img
    """
    ch_j = ch_j or ch_i
    frames = _resolve_frames(pos, frame)

    # Mode-dependent defaults
    if max_r is None:
        max_r = 250.0 if img else 200.0
    if dr is None:
        dr = 0.5 if img else 5.0

    if img:
        pixel_size = float(pos.PixelSize)
        per_frame = []
        for t in frames:
            frame_num = pos.frames[t]
            img_i = np.squeeze(pos.img(Channel=ch_i, frame=[frame_num],
                                       verbose=False, ffield=ffield)).astype(np.float64)
            img_j = (np.squeeze(pos.img(Channel=ch_j, frame=[frame_num],
                                        verbose=False, ffield=ffield)).astype(np.float64)
                     if ch_j != ch_i else img_i)
            per_frame.append(_crosscorr_img_core(img_i, img_j, pixel_size, max_r, dr))
    else:
        # Cell-level: cross-check ffield flag against how FrameLbls were segmented
        if ffield:
            uncorrected = [t for t in frames
                           if not getattr(pos.framelabels[t], '_ffield', False)]
            if uncorrected:
                warnings.warn(
                    f"ffield=True requested but the FrameLbl data for "
                    f"{len(uncorrected)} frame(s) was segmented without flat-field "
                    f"correction.  Cell intensities may contain illumination bias.  "
                    f"Re-segment with ffield=True to correct this.",
                    stacklevel=2,
                )
        else:
            corrected = [t for t in frames
                         if getattr(pos.framelabels[t], '_ffield', False)]
            if corrected:
                warnings.warn(
                    f"ffield=False requested but the FrameLbl data for "
                    f"{len(corrected)} frame(s) was segmented with flat-field "
                    f"correction.  Cell intensities are already ffield-corrected.",
                    stacklevel=2,
                )
        per_frame = []
        for t in frames:
            fl = pos.framelabels[t]
            coords, vals_i, vals_j = _frame_data(fl, ch_i, ch_j, intensity, periring)
            if coords is None or len(coords) < 2:
                continue
            per_frame.append(
                _crosscorr_core(coords, vals_i, vals_j, max_r, dr, n_max, seed)
            )

    if not per_frame:
        raise ValueError("No frames with enough data to compute correlation.")

    result = _combine(per_frame, max_r, dr)
    result.update({'ch_i': ch_i, 'ch_j': ch_j, 'max_r': max_r, 'dr': dr, 'img': img})
    return result


def radial_density(
    pos,
    frame: Optional[Union[int, List[int]]] = None,
    max_r: float = 200.0,
    dr: float = 5.0,
) -> dict:
    """Pair correlation function g(r) for cell positions.

    g(r) = 1 for a completely random (Poisson) spatial distribution.
    g(r) > 1 indicates clustering; g(r) < 1 indicates repulsion at that scale.

    Normalization uses the bounding-box area of each frame as a proxy for the
    field-of-view area.  For non-rectangular or partially filled FOVs this is
    approximate.

    Parameters
    ----------
    pos : PosLbl
    frame : int, list of int, or None
        Frame index into pos.framelabels.  None averages over all frames.
    max_r : float
        Maximum radius in micrometers.
    dr : float
        Bin width in micrometers.

    Returns
    -------
    dict
        r    : (B,) bin centres in µm
        g    : (B,) pair correlation function
        n    : (B,) raw pair counts per bin
        max_r, dr
    """
    frames = _resolve_frames(pos, frame)

    per_frame = []
    for t in frames:
        fl = pos.framelabels[t]
        coords = np.asarray(fl.centroid_um, dtype=np.float64)
        if len(coords) < 2:
            continue
        per_frame.append(_density_core(coords, max_r, dr))

    if not per_frame:
        raise ValueError("No frames with enough cells to compute pair correlation.")

    result = _combine(per_frame, max_r, dr)
    result.update({'max_r': max_r, 'dr': dr})
    return result



# ─────────────────────────────────────────────────────────────────────────────
# Data extraction
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_frames(pos, frame) -> List[int]:
    n = len(pos.framelabels)
    if frame is None:
        return list(range(n))
    if isinstance(frame, (int, np.integer)):
        return [int(frame)]
    return [int(f) for f in frame]


_INTENSITY_METHODS = {
    'mean':   'mean',
    'median': 'median',
    'max':    'maxint',
    'min':    'minint',
    'ninety': 'ninetyint',
}


def _frame_data(fl, ch_i, ch_j, intensity, periring):
    """Return (coords, vals_i, vals_j) for one FrameLbl, or (None, None, None)."""
    if fl.num == 0:
        return None, None, None

    method_name = _INTENSITY_METHODS.get(intensity)
    if method_name is None:
        raise ValueError(
            f"intensity must be one of {list(_INTENSITY_METHODS)}, got '{intensity}'"
        )

    coords = np.asarray(fl.centroid_um, dtype=np.float64)
    if len(coords) == 0:
        return None, None, None

    get = lambda ch: np.asarray(getattr(fl, method_name)(ch, periring=periring),
                                dtype=np.float64)
    vals_i = get(ch_i)
    vals_j = get(ch_j) if ch_j != ch_i else vals_i

    # Guard: lengths must match coords
    if len(vals_i) != len(coords) or len(vals_j) != len(coords):
        return None, None, None

    return coords, vals_i, vals_j


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def _radial_bins(max_r: float, dr: float):
    edges = np.arange(0.0, max_r + dr, dr)
    centres = edges[:-1] + dr / 2.0
    return edges, centres, len(centres)


def _crosscorr_core(coords, vals_i, vals_j, max_r, dr, n_max, seed) -> dict:
    mu_i, sig_i = vals_i.mean(), vals_i.std()
    mu_j, sig_j = vals_j.mean(), vals_j.std()

    if sig_i < 1e-12 or sig_j < 1e-12:
        warnings.warn("Near-zero variance in one channel — g(r) will be NaN.")

    edges, centres, n_bins = _radial_bins(max_r, dr)

    tree = cKDTree(coords)
    dist_mat = tree.sparse_distance_matrix(tree, max_distance=max_r,
                                           output_type='coo_matrix')

    rows = np.asarray(dist_mat.row, dtype=int)
    cols = np.asarray(dist_mat.col, dtype=int)
    dists = np.asarray(dist_mat.data)

    sig_ij = max(sig_i * sig_j, 1e-12)
    contribs = (vals_i[rows] - mu_i) * (vals_j[cols] - mu_j) / sig_ij

    bin_idx = np.searchsorted(edges[1:], dists, side='right')
    valid = bin_idx < n_bins
    bin_idx, contribs = bin_idx[valid], contribs[valid]

    rng = np.random.default_rng(seed)
    bin_counts  = np.bincount(bin_idx, minlength=n_bins).astype(int)
    bin_sums    = np.bincount(bin_idx, weights=contribs,      minlength=n_bins)
    bin_sq_sums = np.bincount(bin_idx, weights=contribs ** 2, minlength=n_bins)

    for b in np.where(bin_counts > n_max)[0]:
        idx_b = rng.choice(np.where(bin_idx == b)[0], size=n_max, replace=False)
        c = contribs[idx_b]
        bin_sums[b]    = c.sum()
        bin_sq_sums[b] = (c ** 2).sum()
        bin_counts[b]  = n_max

    with np.errstate(invalid='ignore'):
        g = np.where(bin_counts > 0, bin_sums / bin_counts, np.nan)
        variance = np.where(
            bin_counts > 1,
            (bin_sq_sums / bin_counts - g ** 2) / (bin_counts - 1),
            np.nan,
        )
        sem = np.sqrt(np.where(variance > 0, variance, 0.0))

    return {'r': centres, 'g': g, 'sem': sem, 'n': bin_counts}


def _density_core(coords: np.ndarray, max_r: float, dr: float) -> dict:
    N = len(coords)
    edges, centres, n_bins = _radial_bins(max_r, dr)

    tree = cKDTree(coords)
    dist_mat = tree.sparse_distance_matrix(tree, max_distance=max_r,
                                           output_type='coo_matrix')

    rows = np.asarray(dist_mat.row, dtype=int)
    cols = np.asarray(dist_mat.col, dtype=int)
    dists = np.asarray(dist_mat.data)

    keep = rows != cols
    dists = dists[keep]

    bin_idx = np.searchsorted(edges[1:], dists, side='right')
    valid = bin_idx < n_bins
    bin_counts = np.bincount(bin_idx[valid], minlength=n_bins).astype(int)

    # Normalize: bounding-box area as proxy for FOV
    mins, maxs = coords.min(axis=0), coords.max(axis=0)
    area = float(np.prod(maxs - mins))
    if area < 1e-6:
        area = 1.0

    # Expected pairs per bin for a homogeneous Poisson process
    expected = N * (N - 1) / area * 2.0 * np.pi * centres * dr

    with np.errstate(invalid='ignore', divide='ignore'):
        g = np.where(expected > 0, bin_counts / expected, np.nan)

    return {'r': centres, 'g': g, 'n': bin_counts}




def _crosscorr_img_core(img_i: np.ndarray, img_j: np.ndarray,
                         pixel_size: float, max_r: float, dr: float) -> dict:
    """FFT-based Pearson spatial cross-correlation, radially averaged.

    Uses periodic-plus-smooth decomposition (Moisan 2011) instead of
    zero-padding.  The periodic component tiles without edge discontinuities,
    so the FFT sees no artificial boundary jumps and the correlation is clean
    at all lags without inflating memory or compute.
    """
    from oyLabImaging.Processing.improcutils import periodic_smooth_decomp

    H, W = img_i.shape
    si, sj = img_i.std(), img_j.std()
    if si < 1e-12 or sj < 1e-12:
        warnings.warn("Near-zero variance in image channel — g(r) will be NaN.")

    # Decompose each image; use only the periodic component for the FFT
    pi, _ = periodic_smooth_decomp(img_i - img_i.mean())
    pj, _ = periodic_smooth_decomp(img_j - img_j.mean()) if img_j is not img_i else (pi, None)

    Fi = np.fft.rfft2(pi)
    Fj = np.fft.rfft2(pj)
    C = np.fft.irfft2(Fi * np.conj(Fj))

    # Normalise to Pearson: C[0,0] = 1 for autocorrelation
    C /= H * W * max(si * sj, 1e-12)

    # Shift so zero-lag is at centre
    C = np.fft.fftshift(C)
    cy, cx = H // 2, W // 2

    # Crop to max_r
    max_pix = int(np.ceil(max_r / pixel_size)) + 1
    y0, y1 = max(0, cy - max_pix), min(H, cy + max_pix + 1)
    x0, x1 = max(0, cx - max_pix), min(W, cx + max_pix + 1)
    C_crop = C[y0:y1, x0:x1]

    # Pixel distances in µm
    ys = (np.arange(y0, y1) - cy) * pixel_size
    xs = (np.arange(x0, x1) - cx) * pixel_size
    XX, YY = np.meshgrid(xs, ys)
    dists = np.sqrt(XX**2 + YY**2).ravel()
    vals  = C_crop.ravel()

    edges, centres, n_bins = _radial_bins(max_r, dr)
    bin_idx = np.searchsorted(edges[1:], dists, side='right')
    valid   = bin_idx < n_bins
    bin_idx, vals = bin_idx[valid], vals[valid]

    bin_counts  = np.bincount(bin_idx, minlength=n_bins).astype(int)
    bin_sums    = np.bincount(bin_idx, weights=vals,      minlength=n_bins)
    bin_sq_sums = np.bincount(bin_idx, weights=vals ** 2, minlength=n_bins)

    with np.errstate(invalid='ignore'):
        g = np.where(bin_counts > 0, bin_sums / bin_counts, np.nan)
        variance = np.where(
            bin_counts > 1,
            (bin_sq_sums / bin_counts - g ** 2) / (bin_counts - 1),
            np.nan,
        )
        sem = np.sqrt(np.where(variance > 0, variance, 0.0))

    return {'r': centres, 'g': g, 'sem': sem, 'n': bin_counts}



# ─────────────────────────────────────────────────────────────────────────────
# Multi-frame aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _combine(per_frame: list, max_r: float, dr: float) -> dict:
    """Weighted average of g(r) across frames (weight = pair count per bin)."""
    if len(per_frame) == 1:
        return dict(per_frame[0])

    _, centres, n_bins = _radial_bins(max_r, dr)
    total_n  = np.zeros(n_bins, dtype=int)
    g_sum    = np.zeros(n_bins)
    sem_sum  = np.zeros(n_bins)   # carry through for first-order propagation

    for frame_result in per_frame:
        n = frame_result['n']
        g = frame_result['g']
        total_n += n
        g_sum   += np.where(np.isnan(g), 0.0, g * n)
        if 'sem' in frame_result:
            s = frame_result['sem']
            sem_sum += np.where(np.isnan(s), 0.0, (s * n) ** 2)

    with np.errstate(invalid='ignore', divide='ignore'):
        g_avg = np.where(total_n > 0, g_sum / total_n, np.nan)

    result = {'r': centres, 'g': g_avg, 'n': total_n}
    if 'sem' in per_frame[0]:
        with np.errstate(invalid='ignore', divide='ignore'):
            sem_avg = np.where(total_n > 0,
                               np.sqrt(sem_sum) / total_n,
                               np.nan)
        result['sem'] = sem_avg

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_radial(results, labels=None, ax=None, colors=None, alpha_fill=0.2,
                min_pairs=5, figsize=(6, 4), fit_results=None):
    """Plot radial g(r) curves with shaded ±1 SEM bands.

    Works with the output of both radial_corr and radial_density.
    Pass fit_results (output of fit_lengthscale) to overlay exponential fits.

    Parameters
    ----------
    results : dict or list of dict
        One result dict or a list of result dicts for multi-curve comparison.
    labels : str or list of str, optional
        Legend labels.  Auto-generated from ch_i/ch_j or 'g(r)' if not given.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created if None.
    colors : list, optional
        Line colours.  Cycles through the default colour cycle if None.
    alpha_fill : float
        Opacity of the ±SEM shaded band.
    min_pairs : int
        Bins with fewer pairs than this are masked (shown as NaN).
    figsize : tuple
        Figure size when ax is None.
    fit_results : dict or list of dict, optional
        Output(s) of fit_lengthscale.  Fit curves are overlaid as dashed lines,
        one per result.  Must be the same length as *results* when a list.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    # Normalise inputs to lists
    if isinstance(results, dict):
        results = [results]
    if labels is None:
        labels = [_auto_label(r) for r in results]
    elif isinstance(labels, str):
        labels = [labels]

    if fit_results is not None:
        if isinstance(fit_results, dict):
            fit_results = [fit_results]
        if len(fit_results) != len(results):
            raise ValueError("fit_results must have the same length as results.")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if colors is None:
        colors = [prop_cycle[i % len(prop_cycle)] for i in range(len(results))]

    is_crosscorr = 'ch_i' in results[0]
    ref_y = 0.0 if is_crosscorr else 1.0

    for idx, (res, label, color) in enumerate(zip(results, labels, colors)):
        r   = res['r']
        g   = res['g'].copy().astype(float)
        n   = res['n']
        sem = res.get('sem', np.full_like(g, np.nan))

        # Mask low-statistics bins
        mask = n < min_pairs
        g[mask]   = np.nan
        sem[mask] = np.nan

        ax.plot(r, g, color=color, linewidth=2, label=label)

        # Shaded ±1 SEM band
        valid = ~np.isnan(g) & ~np.isnan(sem)
        if valid.any():
            ax.fill_between(r[valid],
                            (g - sem)[valid],
                            (g + sem)[valid],
                            color=color, alpha=alpha_fill, linewidth=0)

        # Overlay exponential fit
        if fit_results is not None:
            fr = fit_results[idx]
            lam = fr['lambda']
            rsq = fr['r_sq']
            fit_label = f'fit λ={lam:.1f} µm, R²={rsq:.2f}'
            ax.plot(fr['fit_r'], fr['fit_curve'],
                    color=color, linewidth=1.5, linestyle='--',
                    label=fit_label, zorder=3)

    # Reference line
    ax.axhline(ref_y, color='0.5', linewidth=1, linestyle='--', zorder=0)

    # Axis labels
    ax.set_xlabel('Distance (µm)', fontsize=12)
    if is_crosscorr:
        ch_i = results[0].get('ch_i', '')
        ch_j = results[0].get('ch_j', '')
        if len(results) == 1 and ch_i:
            if ch_i != ch_j:
                ylabel = f'Pearson g(r)  [{ch_i} × {ch_j}]'
            else:
                ylabel = f'Autocorrelation g(r)  [{ch_i}]'
        else:
            ylabel = 'Pearson correlation g(r)'
        ax.set_ylabel(ylabel, fontsize=12)
    else:
        ax.set_ylabel('Pair correlation g(r)', fontsize=12)

    ax.set_xlim(left=0)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which='both', direction='in', top=True, right=True)

    if len(results) > 1 or fit_results is not None or (len(results) == 1 and labels[0]):
        ax.legend(frameon=False, fontsize=10)

    ax.figure.tight_layout()
    return ax


def _auto_label(result: dict) -> str:
    ch_i = result.get('ch_i', '')
    ch_j = result.get('ch_j', '')
    if not ch_i:
        return 'g(r)'
    return ch_i if ch_i == ch_j else f'{ch_i} × {ch_j}'


def local_moran_I(
    pos,
    ch: str,
    frame: Optional[Union[int, List[int]]] = None,
    radius: float = 50.0,
    n_permutations: int = 999,
    intensity: str = 'mean',
    periring: bool = False,
    seed: int = 42,
    ffield: bool = True,
) -> dict:
    """Local Indicators of Spatial Association (LISA / Local Moran's I).

    For each cell i, computes:
        I_i = z_i * lag_i
    where lag_i = mean(z_j for j ≠ i within *radius*), corrected for the
    fraction of the search circle that falls inside the field of view
    (Ripley-style edge correction).  Without correction, boundary cells
    appear artificially suppressed and create a spurious ring of low-I cells.

    z = (x − μ) / σ are globally standardized intensities.  A permutation
    test (shuffle z-scores, keep coordinates) yields a two-sided p-value.

    Mirrors the method in Gagliardi et al. 2021 (Pertz lab, ncf::lisa in R)
    with the addition of edge correction.

    Parameters
    ----------
    pos : PosLbl
    ch : str
        Channel whose intensity is the spatial mark.
    frame : int, list of int, or None
        Frame index.  None returns a list of per-frame result dicts.
    radius : float
        Neighborhood radius in µm.
    n_permutations : int
        Number of random permutations for p-value estimation.
    intensity : str
        Per-cell intensity metric: 'mean', 'median', 'max', 'min', 'ninety'.
    periring : bool
        Use perinuclear-ring intensity.
    seed : int

    Returns
    -------
    dict (single frame) or list of dicts (multiple frames):
        coords      : (N, 2)  cell positions in µm
        I           : (N,)    Local Moran's I per cell (edge-corrected)
        pvalue      : (N,)    two-sided permutation p-value
        z           : (N,)    standardized intensities
        n_neighbors : (N,)    observed neighbor count within radius
        edge_frac   : (N,)    fraction of search circle inside FOV (1=fully inside)
        is_edge     : (N,)    True for cells with < 90 % circle coverage
        ch, radius, frame_index
    """
    frames = _resolve_frames(pos, frame)

    if ffield:
        uncorrected = [t for t in frames
                       if not getattr(pos.framelabels[t], '_ffield', False)]
        if uncorrected:
            warnings.warn(
                f"ffield=True requested but the FrameLbl data for "
                f"{len(uncorrected)} frame(s) was segmented without flat-field "
                f"correction.  Cell intensities may contain illumination bias.  "
                f"Re-segment with ffield=True to correct this.",
                stacklevel=2,
            )
    else:
        corrected = [t for t in frames
                     if getattr(pos.framelabels[t], '_ffield', False)]
        if corrected:
            warnings.warn(
                f"ffield=False requested but the FrameLbl data for "
                f"{len(corrected)} frame(s) was segmented with flat-field "
                f"correction.  Cell intensities are already ffield-corrected.",
                stacklevel=2,
            )

    per_frame = []
    for t in frames:
        fl = pos.framelabels[t]
        coords, vals, _ = _frame_data(fl, ch, ch, intensity, periring)
        if coords is None or len(coords) < 3:
            continue
        # Compute actual FOV bounding box from image dimensions (not cell extent)
        ps = float(fl._pixelsize)
        H, W = fl.imagedims
        xy = np.asarray(fl.XY, dtype=np.float64)
        # coords[:, 0] = xy[0] + ps * row,  coords[:, 1] = xy[1] + ps * col
        fov_bbox = (xy[0], xy[0] + H * ps, xy[1], xy[1] + W * ps)
        res = _local_moran_core(coords, vals, radius, n_permutations, seed, fov_bbox)
        res.update({'ch': ch, 'radius': radius, 'frame_index': t})
        per_frame.append(res)

    if not per_frame:
        raise ValueError("No frames with enough cells to compute LISA.")

    if isinstance(frame, (int, np.integer)) or (
        isinstance(frame, list) and len(frame) == 1
    ):
        return per_frame[0]
    return per_frame


def find_activity_clusters(
    lisa_result: dict,
    min_I: float = 0.5,
    max_pvalue: float = 0.05,
    min_cells: int = 3,
    eps: Optional[float] = None,
) -> dict:
    """Identify clusters of collectively active cells from a LISA result.

    Applies DBSCAN to cells that pass both the Moran's I and p-value
    thresholds.  Mirrors Gagliardi et al. 2021 (STAR Methods).

    Parameters
    ----------
    lisa_result : dict
        Output of local_moran_I for a single frame.
    min_I : float
        Minimum Local Moran's I (default 0.5 as in the paper).
    max_pvalue : float
        Maximum p-value (default 0.05).
    min_cells : int
        Minimum cluster size (default 3 as in the paper).
    eps : float, optional
        DBSCAN search radius in µm.  Defaults to the radius used for LISA.

    Returns
    -------
    dict
        coords      : (M, 2) coordinates of significant cells
        I           : (M,) their Local Moran's I values
        pvalue      : (M,) their p-values
        labels      : (M,) DBSCAN cluster labels (−1 = noise)
        n_clusters  : number of clusters found (excluding noise)
        cluster_sizes : list of cell counts per cluster
    """
    from sklearn.cluster import DBSCAN

    if isinstance(lisa_result, list):
        raise TypeError("Pass a single-frame result dict, not a list.")

    moran  = lisa_result['I']
    pvalue = lisa_result['pvalue']
    coords = lisa_result['coords']
    radius = lisa_result.get('radius', 50.0)

    # Exclude edge cells (unreliable due to incomplete neighborhoods)
    is_edge = lisa_result.get('is_edge', np.zeros(len(moran), dtype=bool))

    valid = ~is_edge & ~np.isnan(moran) & ~np.isnan(pvalue)
    mask = valid & (moran >= min_I) & (pvalue <= max_pvalue)
    if mask.sum() < min_cells:
        return {
            'coords': coords[mask], 'I': moran[mask], 'pvalue': pvalue[mask],
            'labels': np.full(mask.sum(), -1, dtype=int),
            'n_clusters': 0, 'cluster_sizes': [],
        }

    sig_coords = coords[mask]
    sig_I      = moran[mask]
    sig_p      = pvalue[mask]

    eps_use = eps if eps is not None else radius
    db = DBSCAN(eps=eps_use, min_samples=min_cells).fit(sig_coords)
    cluster_labels = db.labels_

    unique = cluster_labels[cluster_labels >= 0]
    n_clusters = len(np.unique(unique)) if len(unique) else 0
    cluster_sizes = [int((cluster_labels == k).sum())
                     for k in range(n_clusters)]

    return {
        'coords': sig_coords,
        'I': sig_I,
        'pvalue': sig_p,
        'labels': cluster_labels,
        'n_clusters': n_clusters,
        'cluster_sizes': cluster_sizes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LISA core
# ─────────────────────────────────────────────────────────────────────────────

def _circle_fov_fraction(coords: np.ndarray, radius: float,
                          rng, fov_bbox=None, n_mc: int = 500) -> np.ndarray:
    """Monte Carlo estimate of the fraction of each cell's search circle inside the FOV.

    fov_bbox = (dim0_min, dim0_max, dim1_min, dim1_max) in the same coordinate
    system as coords.  Falls back to the cell bounding box if None.
    Returns values in (0.05, 1].
    """
    if fov_bbox is not None:
        x0, x1, y0, y1 = fov_bbox
    else:
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0)

    # Sample n_mc uniform points in the unit disc
    theta = rng.uniform(0, 2 * np.pi, n_mc)
    rr    = np.sqrt(rng.uniform(0, 1, n_mc)) * radius
    dx    = rr * np.cos(theta)   # (n_mc,)
    dy    = rr * np.sin(theta)   # (n_mc,)

    # Broadcast: pts_x[i, k] = coords[i, 0] + dx[k]
    pts_x = coords[:, 0:1] + dx[np.newaxis, :]   # (N, n_mc)
    pts_y = coords[:, 1:2] + dy[np.newaxis, :]   # (N, n_mc)

    inside = ((pts_x >= x0) & (pts_x <= x1) &
              (pts_y >= y0) & (pts_y <= y1))
    frac = inside.mean(axis=1)                    # (N,)
    return np.maximum(frac, 0.05)                 # floor to avoid div/0


def _local_moran_core(
    coords: np.ndarray,
    vals: np.ndarray,
    radius: float,
    n_permutations: int,
    seed: int,
    fov_bbox=None,
) -> dict:
    """Local Moran's I with Ripley-style edge correction and permutation p-values."""
    N = len(vals)
    mu, sig = vals.mean(), vals.std()
    if sig < 1e-12:
        warnings.warn("Near-zero variance — Local Moran's I will be 0.")
        z = np.zeros(N)
    else:
        z = (vals - mu) / sig

    rng = np.random.default_rng(seed)

    # Edge correction: fraction of each cell's circle inside the actual FOV
    edge_frac = _circle_fov_fraction(coords, radius, rng, fov_bbox=fov_bbox)
    is_edge   = edge_frac < 0.90

    tree = cKDTree(coords)
    neighbors = tree.query_ball_point(coords, r=radius)
    for i in range(N):
        neighbors[i] = [j for j in neighbors[i] if j != i]

    n_neighbors = np.array([len(nb) for nb in neighbors], dtype=int)

    # Edge-corrected lag: divide mean(z_j) by the circle coverage fraction
    # so that partial neighborhoods are upweighted to represent the full circle.
    lag = np.array([
        z[nb].mean() / edge_frac[i] if len(nb) > 0 else np.nan
        for i, nb in enumerate(neighbors)
    ])
    I_obs = z * lag   # NaN where n_neighbors == 0

    # Permutation test (same edge correction applied to each shuffle)
    z_perms = np.vstack([rng.permutation(z) for _ in range(n_permutations)])
    # z_perms: (n_perm, N)

    extreme_counts = np.zeros(N, dtype=int)
    for i in range(N):
        nb = neighbors[i]
        if len(nb) == 0:
            continue
        lag_perm  = z_perms[:, nb].mean(axis=1) / edge_frac[i]   # (n_perm,)
        I_perm_i  = z_perms[:, i] * lag_perm
        extreme_counts[i] = int((np.abs(I_perm_i) >= np.abs(I_obs[i])).sum())

    pvalue = np.where(
        n_neighbors > 0,
        (extreme_counts + 1) / (n_permutations + 1),
        np.nan,
    )

    return {
        'coords': coords,
        'z': z,
        'I': I_obs,
        'pvalue': pvalue,
        'n_neighbors': n_neighbors,
        'edge_frac': edge_frac,
        'is_edge': is_edge,
    }


def fit_lengthscale(
    result: dict,
    r_min: float = 10.0,
    min_pairs: int = 5,
) -> dict:
    """Fit a decaying exponential to g(r) to extract a spatial length scale.

    Follows Oyler-Yaniv et al. 2017 (Immunity), which showed that the cytokine
    niche autocorrelation decays as G(r) ∝ exp(−r / λ_niche).  Fitting starts
    at *r_min* to exclude within-nucleus correlations (where g(r) is dominated
    by single-cell morphology rather than cell-cell communication).

    Model::

        g(r) = A · exp(−r / λ) + C

    Parameters
    ----------
    result : dict
        Output of radial_corr (contains 'r', 'g',
        optionally 'sem' and 'n').
    r_min : float
        Minimum radius to include in the fit (µm).  Set to roughly one cell
        diameter to avoid within-nucleus autocorrelation.
    min_pairs : int
        Bins with fewer pairs than this are excluded from the fit.

    Returns
    -------
    dict
        lambda     : float   Decay length scale in µm.
        lambda_err : float   1-σ uncertainty on λ from curve_fit covariance.
        A          : float   Amplitude (g at r=0 relative to C).
        A_err      : float   1-σ uncertainty on A.
        C          : float   Asymptotic offset (g at r→∞).
        C_err      : float   1-σ uncertainty on C.
        r_sq       : float   Coefficient of determination of the fit.
        r_min      : float   r_min used.
        fit_r      : (B,)    r values used in the fit.
        fit_g      : (B,)    g values used in the fit.
        fit_curve  : (B,)    Model prediction at fit_r.
        ch_i, ch_j (if present in result)
    """
    from scipy.optimize import curve_fit

    r   = np.asarray(result['r'], dtype=np.float64)
    g   = np.asarray(result['g'], dtype=np.float64)
    n   = np.asarray(result.get('n', np.ones_like(r)), dtype=np.float64)
    sem = result.get('sem', None)
    if sem is not None:
        sem = np.asarray(sem, dtype=np.float64)

    # Select bins: r >= r_min, enough pairs, not NaN
    mask = (r >= r_min) & (n >= min_pairs) & np.isfinite(g)
    if mask.sum() < 4:
        raise ValueError(
            f"Fewer than 4 valid bins above r_min={r_min} µm. "
            "Try a smaller r_min or larger max_r."
        )

    r_fit = r[mask]
    g_fit = g[mask]

    # Weights: 1/SEM if available, else uniform
    if sem is not None:
        s_fit = sem[mask]
        sigma = np.where((s_fit > 0) & np.isfinite(s_fit), s_fit, np.nanmedian(s_fit[s_fit > 0]))
        sigma = np.where(sigma > 0, sigma, 1.0)
    else:
        sigma = None

    def model(r, A, lam, C):
        return A * np.exp(-r / lam) + C

    # Initial guesses from data
    g0, ginf = float(g_fit[0]), float(g_fit[-1])
    A0   = g0 - ginf
    lam0 = float(r_fit.mean())
    C0   = ginf
    p0   = [A0, lam0, C0]

    try:
        popt, pcov = curve_fit(
            model, r_fit, g_fit, p0=p0, sigma=sigma,
            absolute_sigma=(sigma is not None),
            bounds=([-np.inf, 1e-3, -np.inf], [np.inf, np.inf, np.inf]),
            maxfev=10_000,
        )
    except RuntimeError as exc:
        raise RuntimeError(f"fit_lengthscale: curve_fit did not converge. {exc}") from exc

    A_fit, lam_fit, C_fit = popt
    perr = np.sqrt(np.diag(pcov))

    g_pred = model(r_fit, *popt)
    ss_res = np.sum((g_fit - g_pred) ** 2)
    ss_tot = np.sum((g_fit - g_fit.mean()) ** 2)
    r_sq   = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    out = {
        'lambda':     float(lam_fit),
        'lambda_err': float(perr[1]),
        'A':          float(A_fit),
        'A_err':      float(perr[0]),
        'C':          float(C_fit),
        'C_err':      float(perr[2]),
        'r_sq':       float(r_sq),
        'r_min':      float(r_min),
        'fit_r':      r_fit,
        'fit_g':      g_fit,
        'fit_curve':  g_pred,
    }
    for k in ('ch_i', 'ch_j', 'ch'):
        if k in result:
            out[k] = result[k]
    return out


def plot_lengthscale_comparison(
    fit_results,
    labels=None,
    ax=None,
    figsize=(6, 4),
    color='steelblue',
    show_r_sq: bool = True,
) -> object:
    """Dot-plot of λ values with ±1σ error bars for comparison across positions.

    Parameters
    ----------
    fit_results : dict or list of dict
        Output(s) of fit_lengthscale.
    labels : str or list of str, optional
        X-axis labels for each result.  Auto-numbered if None.
    ax : matplotlib.axes.Axes, optional
    figsize : tuple
    color : str or list
        Dot colour(s).
    show_r_sq : bool
        Annotate each point with its R² value.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    if isinstance(fit_results, dict):
        fit_results = [fit_results]
    n = len(fit_results)

    if labels is None:
        labels = [str(i + 1) for i in range(n)]
    elif isinstance(labels, str):
        labels = [labels]

    colors = [color] * n if isinstance(color, str) else list(color)

    lams  = np.array([r['lambda']     for r in fit_results])
    errs  = np.array([r['lambda_err'] for r in fit_results])
    r_sqs = np.array([r['r_sq']       for r in fit_results])

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    xs = np.arange(n)
    ax.errorbar(xs, lams, yerr=errs, fmt='o', capsize=4,
                markersize=8, linewidth=1.5,
                color=colors[0] if len(set(colors)) == 1 else 'k',
                ecolor='0.4', zorder=3)

    # Individual colours when multiple colours provided
    if len(set(colors)) > 1:
        for x, lam, c in zip(xs, lams, colors):
            ax.scatter([x], [lam], color=c, s=80, zorder=4)

    if show_r_sq:
        for x, lam, err, rsq in zip(xs, lams, errs, r_sqs):
            ax.annotate(f'R²={rsq:.2f}',
                        xy=(x, lam + err),
                        xytext=(0, 6), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, color='0.4')

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('Length scale λ (µm)', fontsize=12)
    ax.set_title('Spatial autocorrelation length scale', fontsize=12)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.set_xlim(-0.5, n - 0.5)
    ax.figure.tight_layout()
    return ax
