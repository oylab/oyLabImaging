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


def _frame_data_multi(fl, channels, intensity, periring):
    """Return (coords, vals_matrix) for one FrameLbl with multiple channels.

    vals_matrix : (n_cells, n_channels) float64 array.
    Returns (None, None) if the frame has no cells.
    """
    if fl.num == 0:
        return None, None

    method_name = _INTENSITY_METHODS.get(intensity)
    if method_name is None:
        raise ValueError(
            f"intensity must be one of {list(_INTENSITY_METHODS)}, got '{intensity}'"
        )

    coords = np.asarray(fl.centroid_um, dtype=np.float64)
    if len(coords) == 0:
        return None, None

    cols = []
    for ch in channels:
        v = np.asarray(getattr(fl, method_name)(ch, periring=periring), dtype=np.float64)
        if len(v) != len(coords):
            return None, None
        cols.append(v)

    return coords, np.column_stack(cols)


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

    is_variogram = results[0].get('variogram', False)
    is_crosscorr = 'ch_i' in results[0] and not is_variogram
    if is_variogram:
        ref_y = results[0].get('variogram_ref', 1.0)
    elif is_crosscorr:
        ref_y = 0.0
    else:
        ref_y = 1.0

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
    if is_variogram:
        ch_i = results[0].get('ch_i', '')
        ch_j = results[0].get('ch_j', ch_i)
        if len(results) == 1 and ch_i:
            if ch_i == ch_j:
                ylabel = f'Mark variogram γ̃(r)  [{ch_i}]'
            else:
                ylabel = f'Cross-variogram γ̃(r)  [{ch_i} × {ch_j}]'
        else:
            ylabel = 'Mark variogram γ̃(r)'
        ax.set_ylabel(ylabel, fontsize=12)
    elif is_crosscorr:
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
    if result.get('variogram'):
        ch_i = result.get('ch_i', '')
        ch_j = result.get('ch_j', ch_i)
        if not ch_i:
            return 'γ̃(r)'
        return f'{ch_i} γ̃(r)' if ch_i == ch_j else f'{ch_i} × {ch_j} γ̃(r)'
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
# Mark variogram
# ─────────────────────────────────────────────────────────────────────────────

def _mark_variogram_core(coords, vals_i, vals_j, max_r, dr, n_max, seed) -> dict:
    """Normalized mark variogram / cross-variogram.

    Auto (vals_j is vals_i):
        γ̃(r) = E[(m_i − m_j)²] / (2σ²)  →  1 at large r.
    Cross (vals_j is not vals_i):
        γ̃_AB(r) = E[(A_i − A_j)(B_i − B_j)] / (2σ_A σ_B)  →  Corr(A,B) at large r.
    """
    is_cross = vals_j is not vals_i

    sigma_i = float(vals_i.std())
    sigma_j = float(vals_j.std()) if is_cross else sigma_i

    if sigma_i < 1e-12 or sigma_j < 1e-12:
        warnings.warn("Near-zero variance in one channel — mark variogram is undefined.")
        sigma_i = sigma_j = 1.0

    norm = 2.0 * sigma_i * sigma_j
    if is_cross:
        ref = float(np.corrcoef(vals_i, vals_j)[0, 1])
    else:
        ref = 1.0

    edges, centres, n_bins = _radial_bins(max_r, dr)

    tree = cKDTree(coords)
    dist_mat = tree.sparse_distance_matrix(tree, max_distance=max_r,
                                           output_type='coo_matrix')

    rows  = np.asarray(dist_mat.row,  dtype=int)
    cols  = np.asarray(dist_mat.col,  dtype=int)
    dists = np.asarray(dist_mat.data)

    nonself = rows != cols
    rows, cols, dists = rows[nonself], cols[nonself], dists[nonself]

    if is_cross:
        contribs = (vals_i[rows] - vals_i[cols]) * (vals_j[rows] - vals_j[cols])
    else:
        contribs = (vals_i[rows] - vals_i[cols]) ** 2

    bin_idx = np.searchsorted(edges[1:], dists, side='right')
    valid   = bin_idx < n_bins
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
        msd = np.where(bin_counts > 0, bin_sums / bin_counts, np.nan)
        var_of_mean = np.where(
            bin_counts > 1,
            (bin_sq_sums / bin_counts - msd ** 2) / (bin_counts - 1),
            np.nan,
        )
        sem = np.sqrt(np.where(var_of_mean > 0, var_of_mean, 0.0))

    return {'r': centres, 'g': msd / norm, 'sem': sem / norm, 'n': bin_counts,
            'variogram_ref': ref}


def mark_variogram(
    pos,
    ch_i: str,
    ch_j: Optional[str] = None,
    frame=None,
    max_r: float = 200.0,
    dr: float = 5.0,
    n_max: int = 50_000,
    intensity: str = 'mean',
    periring: bool = False,
    seed: int = 42,
    ffield: bool = True,
    img: bool = False,
) -> dict:
    """Normalized mark variogram / cross-variogram γ̃(r).

    Auto (ch_j omitted or equal to ch_i):
        γ̃(r) = E[(m_i − m_j)²] / (2σ²).  γ̃ → 1 at large r.

    Cross (ch_j given, different from ch_i):
        γ̃_AB(r) = E[(A_i − A_j)(B_i − B_j)] / (2σ_A σ_B).
        γ̃_AB → Corr(A, B) at large r (reference line in plot).

    γ̃ < reference: marks at distance r co-vary more than at large r (positive spatial autocorrelation).
    γ̃ > reference: marks at distance r co-vary less / in opposite direction.

    Parameters
    ----------
    pos : PosLbl
    ch_i : str
        First channel (or the only channel for auto-variogram).
    ch_j : str, optional
        Second channel.  Defaults to ch_i (auto-variogram).
    frame : int, list of int, or None
    max_r : float   Maximum radius in µm.
    dr : float      Bin width in µm.
    n_max : int     Max pairs subsampled per bin (speed vs. accuracy; cell-level only).
    intensity : str Per-cell metric: 'mean', 'median', 'max', 'min', 'ninety' (cell-level only).
    periring : bool Use perinuclear-ring intensity (cell-level only).
    seed : int
    img : bool
        If True, derive the variogram from the pixel-level FFT autocorrelation
        (γ̃(r) = 1 − ρ(r) for auto; γ̃_AB(r) = Corr(A,B) − ρ_AB(r) for cross).
        Gives sub-cell resolution but is sensitive to background.

    Returns
    -------
    dict
        r             : (B,) bin centres in µm
        g             : (B,) γ̃(r) values
        sem           : (B,) standard error of the mean per bin
        n             : (B,) pair counts per bin
        variogram_ref : float — reference value at large r (1.0 for auto; Corr(A,B) for cross)
        ch_i, ch_j, variogram=True, img, max_r, dr
    """
    ch_j = ch_j or ch_i

    if img:
        # Pixel-level variogram derived from FFT autocorrelation:
        #   auto:  γ̃(r) = 1 − ρ(r)
        #   cross: γ̃_AB(r) = Corr(A,B) − ρ_AB(r)
        chj_arg = None if ch_j == ch_i else ch_j
        rc = radial_corr(pos, ch_i=ch_i, ch_j=chj_arg, frame=frame,
                         img=True, ffield=ffield, max_r=max_r, dr=dr)
        is_cross = ch_j != ch_i
        if is_cross:
            corr_ab = float(np.nanmean(rc['g'][-5:]))   # global Corr ≈ g at large r
            g_vario = corr_ab - rc['g']
            ref = 0.0
        else:
            g_vario = 1.0 - rc['g']
            ref = 1.0
        result = dict(rc)
        result['g']             = g_vario
        result['variogram_ref'] = ref
        result.update({'ch_i': ch_i, 'ch_j': ch_j, 'variogram': True, 'img': True})
        return result

    frames = _resolve_frames(pos, frame)

    if ffield:
        uncorrected = [t for t in frames
                       if not getattr(pos.framelabels[t], '_ffield', False)]
        if uncorrected:
            warnings.warn(
                f"ffield=True requested but {len(uncorrected)} frame(s) were "
                f"segmented without flat-field correction.  Intensities may "
                f"contain illumination bias.",
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
        # Pass same object for auto so core can detect it via identity
        vj = vals_j if ch_j != ch_i else vals_i
        per_frame.append(_mark_variogram_core(coords, vals_i, vj, max_r, dr, n_max, seed))

    if not per_frame:
        raise ValueError("No frames with enough cells to compute mark variogram.")

    result = _combine(per_frame, max_r, dr)
    refs = [f['variogram_ref'] for f in per_frame]
    ns   = [f['n'].sum() for f in per_frame]
    result['variogram_ref'] = float(np.average(refs, weights=ns)) if ns else per_frame[0]['variogram_ref']
    result.update({'ch_i': ch_i, 'ch_j': ch_j, 'variogram': True, 'img': False, 'max_r': max_r, 'dr': dr})
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Geographically Weighted Regression (GWR)
# ─────────────────────────────────────────────────────────────────────────────

def _gwr_core(coords: np.ndarray, y: np.ndarray, X_pred: np.ndarray,
              bandwidth: float, kernel: str) -> dict:
    """Fit a local WLS regression at every cell (GWR).

    Parameters
    ----------
    coords   : (N, 2) cell positions in µm
    y        : (N,)   response variable
    X_pred   : (N, p) predictor columns — intercept added internally
    bandwidth: kernel bandwidth in µm
    kernel   : 'gaussian' (soft cutoff at 3σ) or 'bisquare' (hard cutoff)

    Returns
    -------
    dict with arrays of shape (N, p+1) for beta / t_stat / se,
    and (N,) for r_squared and pvalue (normal approximation on t-stats).
    """
    from scipy.stats import norm as _norm

    N = len(coords)
    p = X_pred.shape[1]
    X = np.column_stack([np.ones(N), X_pred])   # (N, p+1)

    beta   = np.full((N, p + 1), np.nan)
    r_sq   = np.full(N, np.nan)
    t_stat = np.full((N, p + 1), np.nan)
    se_arr = np.full((N, p + 1), np.nan)

    cutoff = bandwidth * 3.0 if kernel == 'gaussian' else bandwidth
    tree = cKDTree(coords)
    nb_list = tree.query_ball_point(coords, r=cutoff)

    for i in range(N):
        nb = np.array(nb_list[i], dtype=int)
        if len(nb) < p + 2:
            continue

        d = np.linalg.norm(coords[nb] - coords[i], axis=1)
        if kernel == 'gaussian':
            w = np.exp(-0.5 * (d / bandwidth) ** 2)
        else:
            u = np.clip(d / bandwidth, 0.0, 1.0)
            w = (1.0 - u ** 2) ** 2

        pos_w = w > 1e-10
        nb, w = nb[pos_w], w[pos_w]
        if len(nb) < p + 2:
            continue

        Xi     = X[nb]           # (k, p+1)
        yi     = y[nb]           # (k,)
        w_sqrt = np.sqrt(w)      # (k,)

        # WLS via transformed OLS: multiply rows by √w
        Xw = Xi * w_sqrt[:, None]
        yw = yi * w_sqrt

        b, _, rank, _ = np.linalg.lstsq(Xw, yw, rcond=None)
        if rank < p + 1:
            continue

        beta[i] = b

        y_hat  = Xi @ b
        resid  = yi - y_hat
        y_mean = np.average(yi, weights=w)
        SS_res = float(w @ resid ** 2)
        SS_tot = float(w @ (yi - y_mean) ** 2)
        r_sq[i] = 1.0 - SS_res / SS_tot if SS_tot > 1e-12 else np.nan

        dof    = max(len(nb) - (p + 1), 1)
        sigma2 = SS_res / dof
        try:
            cov  = sigma2 * np.linalg.inv(Xw.T @ Xw)
            se_i = np.sqrt(np.maximum(np.diag(cov), 0.0))
            se_arr[i] = se_i
            t_stat[i] = np.where(se_i > 1e-12, b / se_i, np.nan)
        except np.linalg.LinAlgError:
            pass

    # Two-sided p-values using normal approximation (large-n)
    pvalue = 2.0 * (1.0 - _norm.cdf(np.abs(t_stat)))

    return {'beta': beta, 'r_squared': r_sq, 't_stat': t_stat,
            'se': se_arr, 'pvalue': pvalue}


def gwr(
    pos,
    ch_y: str,
    ch_x,
    frame=None,
    bandwidth: float = 50.0,
    kernel: str = 'gaussian',
    intensity: str = 'mean',
    periring: bool = False,
    seed: int = 42,
    ffield: bool = True,
) -> dict:
    """Geographically Weighted Regression (GWR).

    Fits a local weighted linear regression at each cell using a spatial
    kernel, yielding per-cell estimates of slope, intercept, R², and
    significance.  Where the slope is high and R² is high, the two channels
    are strongly locally correlated; where the slope varies spatially,
    the correlation is non-stationary — the channels couple differently in
    different tissue regions.

    Model at cell i:
        ch_y = β₀(i) + β₁(i)·ch_x₁ + β₂(i)·ch_x₂ + … + ε
    Weights: Gaussian  w_ij = exp(−d²/(2·bw²))
             Bisquare  w_ij = (1−(d/bw)²)²  for d < bw, else 0.

    Parameters
    ----------
    pos : PosLbl
    ch_y : str           Response channel.
    ch_x : str or list   Predictor channel(s).
    frame : int or None
    bandwidth : float    Kernel bandwidth in µm.
    kernel : str         'gaussian' (default) or 'bisquare'.
    intensity : str      Per-cell metric: 'mean', 'median', 'max', 'min', 'ninety'.
    periring : bool
    ffield : bool

    Returns
    -------
    dict (single frame) or list of dicts:
        beta      : (N, p+1)  local intercept + slopes
        r_squared : (N,)      local R²
        t_stat    : (N, p+1)  t-statistics
        se        : (N, p+1)  standard errors
        pvalue    : (N, p+1)  two-sided p-values (normal approx)
        coords    : (N, 2)    cell positions in µm
        ch_y, ch_x, bandwidth, kernel, frame_index
    """
    if isinstance(ch_x, str):
        ch_x = [ch_x]

    channels = [ch_y] + list(ch_x)
    frames = _resolve_frames(pos, frame)

    if ffield:
        uncorrected = [t for t in frames
                       if not getattr(pos.framelabels[t], '_ffield', False)]
        if uncorrected:
            warnings.warn(
                f"ffield=True requested but {len(uncorrected)} frame(s) were "
                f"segmented without flat-field correction.",
                stacklevel=2,
            )
    else:
        corrected = [t for t in frames
                     if getattr(pos.framelabels[t], '_ffield', False)]
        if corrected:
            warnings.warn(
                f"ffield=False requested but {len(corrected)} frame(s) were "
                f"segmented with flat-field correction.  Intensities are already corrected.",
                stacklevel=2,
            )

    per_frame = []
    for t in frames:
        fl = pos.framelabels[t]
        coords, vals = _frame_data_multi(fl, channels, intensity, periring)
        if coords is None or len(coords) < len(ch_x) + 2:
            continue

        y      = vals[:, 0]
        X_pred = vals[:, 1:]

        res = _gwr_core(coords, y, X_pred, bandwidth, kernel)
        res.update({'coords': coords, 'ch_y': ch_y, 'ch_x': list(ch_x),
                    'bandwidth': bandwidth, 'kernel': kernel, 'frame_index': t})
        per_frame.append(res)

    if not per_frame:
        raise ValueError("No frames with enough cells to compute GWR.")

    if isinstance(frame, (int, np.integer)) or (
        isinstance(frame, list) and len(frame) == 1
    ):
        return per_frame[0]
    return per_frame


# ─────────────────────────────────────────────────────────────────────────────
# Spatial regionalization
# ─────────────────────────────────────────────────────────────────────────────

def _spatial_niche_features(coords: np.ndarray, vals_matrix: np.ndarray,
                             radius: float) -> np.ndarray:
    """Mean marker expression within *radius* µm for each cell (includes self).

    Returns (n_cells, n_markers) niche matrix.
    """
    tree = cKDTree(coords)
    neighbors = tree.query_ball_point(coords, r=radius)
    niche = np.zeros_like(vals_matrix)
    for i, nb in enumerate(neighbors):
        niche[i] = vals_matrix[nb].mean(axis=0)
    return niche


def spatial_regions(
    pos,
    channels,
    frame=None,
    radius: float = 50.0,
    n_regions=None,
    method: str = 'gmm',
    intensity: str = 'mean',
    periring: bool = False,
    seed: int = 42,
    ffield: bool = True,
    max_k: int = 10,
) -> dict:
    """Multivariate spatial regionalization.

    Tissue is partitioned into spatial regions based on the local neighborhood
    expression pattern of multiple marker channels.  For each cell a *niche
    vector* is computed — the mean intensity of every marker within *radius* µm
    — then z-score normalized and clustered.  Regions are defined by which
    markers are collectively elevated in the neighborhood, not by co-expression
    on individual cells.

    Parameters
    ----------
    pos : PosLbl
    channels : list of str
        Marker channels to include in the niche vector.
    frame : int, list of int, or None
        Frame index.  None processes all frames independently.
    radius : float
        Neighborhood radius in µm for niche averaging.
    n_regions : int or None
        Number of regions.  None → auto-selected (k = 2 … max_k) via silhouette.
    method : str
        'gmm' (default) or 'kmeans'.
    intensity : str
        Per-cell intensity metric: 'mean', 'median', 'max', 'min', 'ninety'.
    periring : bool
        Use perinuclear-ring intensity.
    seed : int
    ffield : bool
    max_k : int
        Upper bound for auto k-selection.

    Returns
    -------
    dict (single frame) or list of dicts:
        labels          : (N,) int    — region index per cell (0-based)
        coords          : (N, 2)      — cell positions in µm
        niche           : (N, C)      — z-scored niche feature matrix used for clustering
        marker_profiles : (k, C)      — mean raw niche expression per region per channel
        n_regions       : int
        channels        : list of str
        silhouette      : float       — silhouette score of the clustering
        frame_index     : int
    """
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    frames = _resolve_frames(pos, frame)

    if ffield:
        uncorrected = [t for t in frames
                       if not getattr(pos.framelabels[t], '_ffield', False)]
        if uncorrected:
            warnings.warn(
                f"ffield=True requested but {len(uncorrected)} frame(s) were "
                f"segmented without flat-field correction.  Intensities may "
                f"contain illumination bias.",
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
        coords, vals = _frame_data_multi(fl, channels, intensity, periring)
        min_cells = max(3, n_regions if n_regions is not None else 2)
        if coords is None or len(coords) < min_cells:
            continue

        niche_raw = _spatial_niche_features(coords, vals, radius)

        scaler = StandardScaler()
        X = scaler.fit_transform(niche_raw)

        sample_size = min(5000, len(X))

        # Auto-select k via silhouette score
        k = n_regions
        if k is None:
            sil_scores = []
            for ki in range(2, max_k + 1):
                km_tmp = KMeans(n_clusters=ki, n_init=10, random_state=seed)
                lbl_tmp = km_tmp.fit_predict(X)
                sil_scores.append(
                    silhouette_score(X, lbl_tmp, sample_size=sample_size,
                                     random_state=seed)
                )
            k = int(np.argmax(sil_scores)) + 2

        if method == 'gmm':
            model = GaussianMixture(n_components=k, random_state=seed, n_init=5)
            labels = model.fit_predict(X)
        else:
            model = KMeans(n_clusters=k, n_init=20, random_state=seed)
            labels = model.fit_predict(X)

        sil = float(silhouette_score(X, labels, sample_size=sample_size,
                                     random_state=seed))

        marker_profiles = np.array([
            niche_raw[labels == ki].mean(axis=0) if (labels == ki).any()
            else np.zeros(len(channels))
            for ki in range(k)
        ])

        per_frame.append({
            'labels':          labels,
            'coords':          coords,
            'niche':           X,
            'marker_profiles': marker_profiles,
            'n_regions':       k,
            'channels':        list(channels),
            'silhouette':      sil,
            'frame_index':     t,
        })

    if not per_frame:
        raise ValueError("No frames with enough cells to compute spatial regions.")

    if isinstance(frame, (int, np.integer)) or (
        isinstance(frame, list) and len(frame) == 1
    ):
        return per_frame[0]
    return per_frame


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


# ─────────────────────────────────────────────────────────────────────────────
# Getis-Ord Gi*
# ─────────────────────────────────────────────────────────────────────────────

def _gistar_core(
    coords: np.ndarray,
    vals: np.ndarray,
    radius: float,
    seed: int,
    fov_bbox=None,
) -> dict:
    """Getis-Ord Gi* with analytical z-score and Ripley-style edge flagging.

    Gi* (with asterisk) includes the focal cell in its own neighbourhood,
    making the statistic a local z-score of the neighbourhood sum:

        z_i = (Σ_{j ∈ N*_i} x_j − X̄·W_i) / (S · √((n·W_i − W_i²) / (n−1)))

    where X̄ is the global mean, S the population std, W_i = |N*_i|.
    Positive z: hot spot (high-value cluster); negative: cold spot.
    """
    from scipy.stats import norm as _norm

    n = len(vals)
    x_bar = float(vals.mean())
    s = float(np.sqrt(max(np.mean(vals ** 2) - x_bar ** 2, 0.0)))  # population std

    rng = np.random.default_rng(seed)
    edge_frac = _circle_fov_fraction(coords, radius, rng, fov_bbox=fov_bbox)
    is_edge = edge_frac < 0.90

    tree = cKDTree(coords)
    neighbors = tree.query_ball_point(coords, r=radius)  # includes self (Gi*)

    z_scores = np.full(n, np.nan, dtype=np.float64)

    if s < 1e-12:
        warnings.warn("Near-zero variance — Gi* will be 0.")
        z_scores[:] = 0.0
    else:
        for i in range(n):
            nb = np.array(neighbors[i], dtype=int)
            w_i = len(nb)
            if w_i == 0:
                continue
            numer = float(vals[nb].sum()) - x_bar * w_i
            denom_sq = s ** 2 * (n * w_i - w_i ** 2) / max(n - 1, 1)
            z_scores[i] = numer / np.sqrt(denom_sq) if denom_sq > 0 else 0.0

    pvalue = np.where(
        np.isfinite(z_scores),
        2.0 * (1.0 - _norm.cdf(np.abs(z_scores))),
        np.nan,
    )

    return {
        'coords':    coords,
        'vals':      vals,
        'z_score':   z_scores,
        'pvalue':    pvalue,
        'edge_frac': edge_frac,
        'is_edge':   is_edge,
    }


def gistar(
    pos,
    ch: str,
    frame=None,
    radius: float = 50.0,
    intensity: str = 'mean',
    periring: bool = False,
    seed: int = 42,
    ffield: bool = True,
) -> dict:
    """Getis-Ord Gi* hot-spot statistic for each cell.

    Returns a per-cell z-score indicating whether cell i sits within a
    local cluster of high (z > 0) or low (z < 0) values relative to the
    global distribution.  Unlike LISA, Gi* does not require the focal cell
    to be atypical itself — it measures whether its neighbourhood as a
    whole is elevated.

    Parameters
    ----------
    pos : PosLbl
    ch : str
        Channel whose intensity is the spatial mark.
    frame : int, list of int, or None
    radius : float
        Neighbourhood radius in µm.
    intensity : str
        Per-cell metric: 'mean', 'median', 'max', 'min', 'ninety'.
    periring : bool
        Use perinuclear-ring intensity.
    seed : int

    Returns
    -------
    dict (single frame) or list of dicts:
        coords    : (N, 2)  cell positions in µm
        z_score   : (N,)    Gi* z-score per cell
        pvalue    : (N,)    two-sided p-value (normal approximation)
        vals      : (N,)    raw intensity values
        edge_frac : (N,)    fraction of search circle inside FOV
        is_edge   : (N,)    True for cells with < 90 % circle coverage
        ch, radius, frame_index
    """
    frames = _resolve_frames(pos, frame)

    if ffield:
        uncorrected = [t for t in frames
                       if not getattr(pos.framelabels[t], '_ffield', False)]
        if uncorrected:
            warnings.warn(
                f"ffield=True requested but {len(uncorrected)} frame(s) were "
                f"segmented without flat-field correction.  Intensities may "
                f"contain illumination bias.",
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
        ps = float(fl._pixelsize)
        H, W = fl.imagedims
        xy = np.asarray(fl.XY, dtype=np.float64)
        fov_bbox = (xy[0], xy[0] + H * ps, xy[1], xy[1] + W * ps)
        res = _gistar_core(coords, vals, radius, seed, fov_bbox)
        res.update({'ch': ch, 'radius': radius, 'frame_index': t})
        per_frame.append(res)

    if not per_frame:
        raise ValueError("No frames with enough cells to compute Gi*.")

    if isinstance(frame, (int, np.integer)) or (
        isinstance(frame, list) and len(frame) == 1
    ):
        return per_frame[0]
    return per_frame


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




# ─────────────────────────────────────────────────────────────────────────────
# Spectral power spectrum  (Jerison et al. 2025 PNAS)
# ─────────────────────────────────────────────────────────────────────────────

def _alpha_shape_faces(coords: np.ndarray, alpha: float):
    """Alpha-shape triangulation of 2-D cell centroids.

    Builds the full Delaunay triangulation then keeps only triangles whose
    circumscribed circle radius < 1/alpha.  This removes long boundary
    triangles that span empty regions, matching the paper's mesh construction.

    Parameters
    ----------
    coords : (N, 2) float  cell centroids in µm
    alpha  : float         ~1 / max_circumradius (µm^{-1}).
                           Larger alpha = tighter (more concave) boundary.
                           A typical starting point is alpha ≈ 1/(3*cell_spacing).

    Returns
    -------
    faces : (M, 3) int32
    """
    import math
    from scipy.spatial import Delaunay

    tri   = Delaunay(coords)
    faces = []
    max_r = 1.0 / alpha

    for ia, ib, ic in tri.simplices:
        pa, pb, pc = coords[ia], coords[ib], coords[ic]
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        s = (a + b + c) / 2.0
        area2 = s*(s-a)*(s-b)*(s-c)
        if area2 <= 0:
            continue
        circum_r = a * b * c / (4.0 * math.sqrt(area2))
        if circum_r < max_r:
            faces.append([ia, ib, ic])

    if not faces:
        raise ValueError(
            f"Alpha-shape produced no triangles (alpha={alpha}).  "
            "Try a smaller alpha value (larger max circumradius)."
        )
    return np.array(faces, dtype=np.int32)


def _build_normalized_laplacian(coords: np.ndarray, faces: np.ndarray):
    """Compute M^{-1/2} L M^{-1/2} using gpytoolbox.

    Parameters
    ----------
    coords : (N, 2) float  cell centroids in µm
    faces  : (M, 3) int    triangle face list from _alpha_shape_faces

    Returns
    -------
    Lm : (N, N) scipy sparse CSR  symmetric normalised Laplacian
    """
    import gpytoolbox as gpy
    import scipy.sparse as sp

    L = gpy.cotangent_laplacian(coords, faces)
    M = gpy.massmatrix(coords, faces).tocsc()

    m_diag = M.diagonal()

    if np.any(m_diag == 0):
        # Isolated vertex (e.g. at mesh boundary): fall back to mean mass
        m_mean = m_diag[m_diag > 0].mean()
        Lm = L / m_mean
    else:
        inv_sqrt_m = 1.0 / np.sqrt(m_diag)
        D  = sp.diags(inv_sqrt_m)
        Lm = D @ L @ D

    # Numerical symmetrisation
    Lm = (Lm + Lm.T) / 2.0
    return Lm.tocsr()


def _laplacian_eigenmodes(Lm, n_modes):
    """Eigendecompose the normalised Laplacian, stripping the DC mode.

    Parameters
    ----------
    n_modes : int or None
        None → full dense decomposition via scipy.linalg.eigh (all N-1 modes;
               practical for N ≲ 5 000).
        int  → n_modes smallest oscillatory modes via sparse shift-invert
               (efficient for large N).

    Returns
    -------
    eigenvalues  : (k,) ascending, non-negative
    eigenvectors : (N, k)  orthonormal columns (V^T V = I)
    """
    import scipy.linalg as la
    from scipy.sparse.linalg import eigsh

    N = Lm.shape[0]

    if n_modes is None:
        vals, vecs = la.eigh(Lm.toarray())
        vals = np.maximum(vals, 0.0)
        return vals[1:], vecs[:, 1:]          # strip DC mode (index 0)

    k = min(n_modes + 1, N - 2)
    try:
        vals, vecs = eigsh(Lm, k=k, which='LM', sigma=0.0, tol=1e-6)
    except Exception:
        vals, vecs = eigsh(Lm, k=k, which='SM', tol=1e-6)

    idx  = np.argsort(vals)
    vals = np.maximum(vals[idx], 0.0)
    vecs = vecs[:, idx]
    # Strip DC mode then return exactly n_modes oscillatory modes
    return vals[1:n_modes + 1], vecs[:, 1:n_modes + 1]


class SpectralBasis:
    """Cotangent Laplacian spectral basis for one frame.

    Build via ``PosLbl.spectral_basis(frame=0)`` or
    ``build_spectral_basis(pos, frame=0)``.

    Attributes
    ----------
    eigenvectors : (N, k)   orthonormal columns from scipy.linalg.eigh
    eigenvalues  : (k,)     ascending, non-negative; DC mode excluded
    lengthscales_um : (k,)  2 / sqrt(λ) in µm — Jerison et al. convention
    coords       : (N, 2)   cell centroids in µm
    frame_index  : int
    """

    def __init__(self, eigenvectors, eigenvalues, coords, frame_index):
        self.eigenvectors = eigenvectors
        self.eigenvalues  = eigenvalues
        self.coords       = coords
        self.frame_index  = frame_index

    @property
    def n_modes(self):
        return self.eigenvectors.shape[1]

    @property
    def n_cells(self):
        return self.eigenvectors.shape[0]

    @property
    def lengthscales_um(self):
        """Spatial length scale per mode: 2 / sqrt(λ_k) in µm.

        Matches Jerison et al. 2025 convention.  Larger = coarser pattern.
        """
        return 2.0 / np.sqrt(np.maximum(self.eigenvalues, 1e-12))

    def project(self, expression):
        """Project expression onto eigenmodes.

        For orthonormal eigenvectors V (from eigh), the projection is simply
        V^T @ expression — a straightforward matrix multiply.

        Parameters
        ----------
        expression : (N,) or (N, n_ch)

        Returns
        -------
        coefficients : (k,) or (k, n_ch)
        """
        return self.eigenvectors.T @ expression

    def power_spectrum(self, expression, n_permutations=100, seed=42):
        """Spatial power spectrum with permutation null model.

        Projects expression onto the eigenmodes, computes fractional variance
        per mode, and runs a permutation null (shuffle cell labels, keep
        geometry) to identify biologically organised length scales.

        The characteristic length scale per channel follows Jerison et al.:
        power-weighted mean spatial frequency over modes that exceed the null.

        Parameters
        ----------
        expression : (N, n_ch)  z-scored expression (channels as columns)
        n_permutations : int
        seed : int

        Returns
        -------
        dict
            power            (k, n_ch)  fractional variance per mode
            power_null_mean  (k, n_ch)
            power_null_std   (k, n_ch)
            signal_to_null   (k, n_ch)  power / (null_mean + null_std)
            char_lengthscale (n_ch,)    power-weighted mean (paper's estimator)
            peak_lengthscale (n_ch,)    length scale of highest-SNR mode
            lengthscales_um  (k,)
            eigenvalues      (k,)
            frame_index      int
        """
        if expression.ndim == 1:
            expression = expression[:, None]
        n_ch = expression.shape[1]

        B    = self.project(expression)                 # (k, n_ch)
        B2   = B ** 2
        psum = B2.sum(axis=0, keepdims=True) + 1e-12
        power = B2 / psum

        rng  = np.random.default_rng(seed)
        null = np.empty((n_permutations, self.n_modes, n_ch))
        for p in range(n_permutations):
            idx  = rng.permutation(self.n_cells)
            Bn   = self.project(expression[idx])
            Bn2  = Bn ** 2
            null[p] = Bn2 / (Bn2.sum(axis=0, keepdims=True) + 1e-12)

        null_mean = null.mean(axis=0)
        null_std  = null.std(axis=0)
        snr       = power / (null_mean + null_std + 1e-12)

        # Characteristic length scale: power-weighted mean spatial frequency
        # over modes above the null baseline (Jerison et al. kscale estimator).
        try:
            from scipy.ndimage import gaussian_filter1d
            def _smooth(p):
                return gaussian_filter1d(p.astype(float), sigma=3)
        except ImportError:
            def _smooth(p):
                return p

        freq    = np.sqrt(self.eigenvalues)   # k  (µm^{-1}), ascending
        char_ls = np.empty(n_ch)
        for c in range(n_ch):
            p_sm  = _smooth(power[:, c])
            fnull = null_mean[:, c].mean()
            above = p_sm > fnull
            if above.any():
                pw = power[above, c]
                kw = freq[above]
                char_ls[c] = 2.0 / (np.dot(pw, kw) / pw.sum())
            else:
                char_ls[c] = self.lengthscales_um[power[:, c].argmax()]

        return {
            'power':             power,
            'power_null_mean':   null_mean,
            'power_null_std':    null_std,
            'signal_to_null':    snr,
            'char_lengthscale':  char_ls,
            'peak_lengthscale':  self.lengthscales_um[snr.argmax(axis=0)],
            'lengthscales_um':   self.lengthscales_um.copy(),
            'eigenvalues':       self.eigenvalues.copy(),
            'frame_index':       self.frame_index,
        }


def build_spectral_basis(
    pos,
    frame=None,
    n_modes=None,
    alpha=0.05,
):
    """Build cotangent Laplacian spectral basis for one or more frames.

    Computes the alpha-shape triangulation, the mass-normalised cotangent
    Laplacian (via gpytoolbox), and its eigenmodes.  The result is
    geometry-only: it can be reused to project any number of channels.

    Parameters
    ----------
    pos     : PosLbl
    frame   : int, list of int, or None
    n_modes : int or None
        None (default) uses scipy.linalg.eigh for all N-1 oscillatory modes —
        matches the paper exactly; practical for N ≲ 5 000.
        Pass an integer to use the sparse shift-invert solver for large N.
    alpha   : float
        Alpha-shape parameter (~1/max_circumradius in µm^{-1}).
        Controls how tightly the mesh boundary hugs the point cloud.
        Default 0.05 µm^{-1} (max circumradius ≈ 20 µm).  Increase for
        denser fields; decrease for sparser or larger fields.

    Returns
    -------
    SpectralBasis  (single frame) or list of SpectralBasis
    """
    frames    = _resolve_frames(pos, frame)
    per_frame = []
    for t in frames:
        fl     = pos.framelabels[t]
        coords = np.asarray(fl.centroid_um, dtype=np.float64)
        min_n  = 4 if n_modes is None else n_modes + 4
        if len(coords) < min_n:
            raise ValueError(
                f"Frame {t}: only {len(coords)} cells — need at least {min_n}."
            )
        faces = _alpha_shape_faces(coords, alpha)
        Lm    = _build_normalized_laplacian(coords, faces)
        vals, vecs = _laplacian_eigenmodes(Lm, n_modes)
        per_frame.append(SpectralBasis(vecs, vals, coords, t))

    if not per_frame:
        raise ValueError("No valid frames found.")

    if isinstance(frame, (int, np.integer)) or (
        isinstance(frame, list) and len(frame) == 1
    ):
        return per_frame[0]
    return per_frame


def spectral_power_spectrum(
    pos,
    ch,
    frame=None,
    n_modes=None,
    alpha=0.05,
    intensity: str = 'mean',
    periring: bool = False,
    ffield: bool = True,
    n_permutations: int = 100,
    seed: int = 42,
):
    """Spatial power spectrum via cotangent Laplacian spectral decomposition.

    Full pipeline in one call: builds the spectral basis, z-scores expression,
    projects onto eigenmodes, and runs the permutation null model.

    Prefer ``PosLbl.spectral_power_spectrum`` if you plan to analyze multiple
    channels (it reuses the cached basis).

    Parameters
    ----------
    pos           : PosLbl
    ch            : str or list of str
    frame         : int, list of int, or None
    n_modes       : int or None  (see build_spectral_basis)
    alpha         : float        (see build_spectral_basis)
    intensity     : str
    periring      : bool
    ffield        : bool
    n_permutations: int
    seed          : int

    Returns
    -------
    dict or list of dicts — see SpectralBasis.power_spectrum
    """
    import warnings
    channels = [ch] if isinstance(ch, str) else list(ch)
    frames   = _resolve_frames(pos, frame)

    if ffield:
        uncorrected = [t for t in frames
                       if not getattr(pos.framelabels[t], '_ffield', False)]
        if uncorrected:
            warnings.warn(
                f"ffield=True but {len(uncorrected)} frame(s) were segmented "
                "without flat-field correction.", stacklevel=2)

    per_frame = []
    for t in frames:
        fl = pos.framelabels[t]
        coords, vals = _frame_data_multi(fl, channels, intensity, periring)
        if coords is None:
            continue

        faces = _alpha_shape_faces(coords, alpha)
        Lm    = _build_normalized_laplacian(coords, faces)
        evals, evecs = _laplacian_eigenmodes(Lm, n_modes)
        basis = SpectralBasis(evecs, evals, coords, t)

        mu    = vals.mean(axis=0)
        sigma = vals.std(axis=0)
        sigma[sigma < 1e-12] = 1.0
        z = (vals - mu) / sigma

        res = basis.power_spectrum(z, n_permutations=n_permutations, seed=seed)
        res['channels'] = channels
        per_frame.append(res)

    if not per_frame:
        raise ValueError("No valid frames.")

    if isinstance(frame, (int, np.integer)) or (
        isinstance(frame, list) and len(frame) == 1
    ):
        return per_frame[0]
    return per_frame


def plot_spatial_power_spectrum(
    results,
    channels=None,
    ax=None,
    figsize=None,
    show_null: bool = True,
    log_x: bool = True,
):
    """Plot spatial power spectrum: fractional variance vs length scale (µm).

    Parameters
    ----------
    results  : dict or list of dicts
        Output of SpectralBasis.power_spectrum or PosLbl.spectral_power_spectrum.
    channels : list of str, optional
    ax       : matplotlib.axes.Axes, optional
    figsize  : tuple, optional
    show_null: bool  shade permutation null ± 1 std (default True)
    log_x    : bool  log-scale x-axis (default True)

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    res   = results[0] if isinstance(results, list) else results
    ls    = res['lengthscales_um']          # (k,) descending (coarse→fine)
    power = res['power']                    # (k, n_ch)
    n_ch  = power.shape[1]

    # Sort ascending (fine→coarse on log-x reads left→right)
    order = np.argsort(ls)
    ls    = ls[order]
    power = power[order]

    if channels is None:
        channels = res.get('channels', [f'ch{i}' for i in range(n_ch)])

    if ax is None:
        fw = figsize[0] if figsize else max(6, n_ch * 2)
        fh = figsize[1] if figsize else 4
        _, ax = plt.subplots(figsize=(fw, fh))

    colors = plt.cm.tab10(np.linspace(0, 0.9, n_ch))

    for c, (ch_name, color) in enumerate(zip(channels, colors)):
        p = power[:, c]
        ax.plot(ls, p, color=color, label=ch_name, lw=1.5)

        if show_null and 'power_null_mean' in res:
            nm = res['power_null_mean'][order, c]
            ns = res['power_null_std'][order, c]
            ax.fill_between(ls, nm - ns, nm + ns, color=color, alpha=0.15)
            ax.plot(ls, nm, color=color, lw=0.8, ls='--', alpha=0.5)

        if 'char_lengthscale' in res:
            ax.axvline(res['char_lengthscale'][c], color=color,
                       lw=0.8, ls=':', alpha=0.8)

    if log_x:
        ax.set_xscale('log')

    ax.set_xlabel('Length scale (µm)')
    ax.set_ylabel('Fractional power')
    ax.set_title(f'Spatial power spectrum  (frame {res.get("frame_index", 0)})')
    ax.legend(fontsize=9)
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.figure.tight_layout()
    return ax


def plot_wavenumber_power_spectrum(
    results,
    channels=None,
    ax=None,
    figsize=None,
    show_null: bool = True,
    show_null_fill: bool = False,
    smooth_sigma: float = 3.0,
    smooth_sigma_null: float = 2.0,
    secondary_axis: bool = True,
):
    """Plot fraction of power vs wavenumber — Jerison et al. Figure 6 style.

    X-axis (bottom): k = sqrt(eigenvalue) in µm⁻¹ (log scale).
    X-axis (top):    corresponding length scale λ = 2/k in µm.
    Y-axis:          fraction of power (Gaussian-smoothed, log scale).

    Permutation null is shown in grey.

    Parameters
    ----------
    results      : dict or list of dicts
        Output of SpectralBasis.power_spectrum or PosLbl.spectral_power_spectrum.
    channels     : list of str, optional
    ax           : matplotlib.axes.Axes, optional
    figsize      : tuple, optional
    show_null    : bool   plot permutation null mean as grey line (default True)
    show_null_fill: bool  shade ±1 std around null (default False, matches paper)
    smooth_sigma : float  Gaussian smoothing sigma for data (default 3, matches paper)
    smooth_sigma_null : float  smoothing sigma for null (default 2, matches paper)
    secondary_axis : bool  add length-scale axis on top (default True)

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d

    res   = results[0] if isinstance(results, list) else results
    power = res['power']        # (k, n_ch) in ascending eigenvalue order
    evals = res['eigenvalues']  # (k,) ascending
    n_ch  = power.shape[1]

    if channels is None:
        channels = res.get('channels', [f'ch{i}' for i in range(n_ch)])

    if ax is None:
        fw = figsize[0] if figsize else 5
        fh = figsize[1] if figsize else 5
        _, ax = plt.subplots(figsize=(fw, fh))

    k    = np.sqrt(np.maximum(evals, 0.0))         # (k,) µm⁻¹, ascending
    k_sm = gaussian_filter1d(k.astype(float), sigma=smooth_sigma)

    colors = plt.cm.tab10(np.linspace(0, 0.9, n_ch))

    for c, (ch_name, color) in enumerate(zip(channels, colors)):
        p_sm = gaussian_filter1d(power[:, c].astype(float), sigma=smooth_sigma)
        ax.loglog(k_sm, p_sm, color=color, label=ch_name, lw=2.0, alpha=0.8)

        if show_null and 'power_null_mean' in res:
            nm = res['power_null_mean'][:, c]
            nm_sm = gaussian_filter1d(nm.astype(float), sigma=smooth_sigma_null)
            ax.loglog(k_sm, nm_sm, color='grey', lw=1.0, alpha=0.5)
            if show_null_fill and 'power_null_std' in res:
                ns_sm = gaussian_filter1d(res['power_null_std'][:, c].astype(float),
                                          sigma=smooth_sigma_null)
                ax.fill_between(k_sm, nm_sm - ns_sm, nm_sm + ns_sm,
                                color='grey', alpha=0.15)

        if 'char_lengthscale' in res and res['char_lengthscale'][c] > 0:
            k_char = 2.0 / res['char_lengthscale'][c]
            ax.axvline(k_char, color=color, lw=0.8, ls=':', alpha=0.8)

    ax.set_xlabel('k (µm⁻¹)', fontsize=12)
    ax.set_ylabel('Fraction of power', fontsize=12)
    ax.set_title(f'Spatial power spectrum  (frame {res.get("frame_index", 0)})')
    ax.legend(fontsize=9, frameon=False)
    ax.set_box_aspect(1)
    ax.tick_params(which='both', direction='in', top=not secondary_axis, right=True)

    if secondary_axis:
        def _k_to_ls(k_):
            return 2.0 / np.where(np.asarray(k_) > 0, np.asarray(k_), np.inf)
        def _ls_to_k(ls_):
            return 2.0 / np.where(np.asarray(ls_) > 0, np.asarray(ls_), np.inf)
        secax = ax.secondary_xaxis('top', functions=(_k_to_ls, _ls_to_k))
        secax.set_xlabel('Length scale (µm)', fontsize=12)
        secax.tick_params(which='both', direction='in')

    ax.figure.tight_layout()
    return ax


def plot_spatial_reconstruction(
    coords,
    expression,
    basis,
    channels=None,
    n_modes_reconstruct=50,
    figsize=None,
    cmap='coolwarm',
    point_size=4,
    vclip=99,
):
    """Figure 4C–style: raw data vs low-pass spatial reconstruction.

    For each channel, plots two scatter maps side by side:
      left  — centred expression values coloured onto cell positions
      right — reconstruction using only the first *n_modes_reconstruct*
              eigenmodes (i.e. a spatial low-pass filter)

    Parameters
    ----------
    coords : (N, 2) array
        Cell x/y coordinates in µm.
    expression : (N, n_ch) array
        Mean-centred (or z-scored) expression values, one column per channel.
    basis : SpectralBasis
        Cotangent Laplacian eigenbasis (from build_spectral_basis / P.spectral_basis).
    channels : list of str, optional
        Channel labels for the column titles.
    n_modes_reconstruct : int
        Number of low-frequency modes to include in the reconstruction (default 50).
    figsize : tuple, optional
    cmap : str  diverging colourmap (default 'coolwarm')
    point_size : float  scatter marker size (default 4)
    vclip : float  percentile for symmetric colour clipping (default 99)

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if expression.ndim == 1:
        expression = expression[:, None]
    n_ch = expression.shape[1]
    if channels is None:
        channels = [f'ch{i}' for i in range(n_ch)]

    # Mean-centre
    expr_c = expression - expression.mean(axis=0)

    # Project and reconstruct
    B = basis.eigenvectors.T @ expr_c          # (k, n_ch)
    N = min(n_modes_reconstruct, basis.n_modes)
    recon = basis.eigenvectors[:, :N] @ B[:N]  # (n_cells, n_ch)

    n_rows = n_ch
    fw = figsize[0] if figsize else 8
    fh = figsize[1] if figsize else max(3 * n_rows, 4)
    fig, axes = plt.subplots(n_rows, 2, figsize=(fw, fh),
                             squeeze=False)

    x, y = coords[:, 0], coords[:, 1]

    for row, (ch, ax_data, ax_rec) in enumerate(
            zip(channels, axes[:, 0], axes[:, 1])):

        for col_idx, (ax, vals, title) in enumerate([
            (ax_data, expr_c[:, row], 'Data'),
            (ax_rec,  recon[:, row],  f'{N} modes'),
        ]):
            vmax = np.percentile(np.abs(vals), vclip)
            sc = ax.scatter(x, y, c=vals, cmap=cmap,
                            vmin=-vmax, vmax=vmax,
                            s=point_size, linewidths=0, rasterized=True)
            ax.set_aspect('equal')
            ax.axis('off')
            if row == 0:
                ax.set_title(title, fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(ch, fontsize=10, rotation=90, labelpad=4)
            plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)

    fig.tight_layout()
    return fig
