# Results module for microscopy data.
# Aggregates all Pos labels for a specific experiment

from os import listdir
from os.path import join

import cloudpickle
import dill
import numpy as np

# AOY
from oyLabImaging import Metadata
from oyLabImaging.Processing import PosLbl
from oyLabImaging.Processing.generalutils import alias
from natsort import natsorted


class results(object):
    """
    Class for experiment results (multi timepoint, multi position, single experiment, multi channel).
    Parameters
    ----------
    MD : relevant metadata OR
    pth : str path to relevant metadata

    Segmentation parameters
    -----------------------
    **kwargs : specific args for segmentation function, anything that goes into FrameLbl
    Threads : how many threads to use for parallel execution. Limited to ~6 for GPU based segmentation and 128 for CPU (but don't use all 128)

    Returns
    -------
    results instance

    Class properties
    ----------------
     'PosLbls',
     'PosNames',
     'acq',
     'channels',
     'frames',
     'groups',
     'pth',
     'tracks'

    Class methods
    -------------
     'calculate_tracks',
     'load',
     'save',
     'setPosLbls',
     'show_images',
     'show_points',
     'show_tracks',

    """

    def __init__(self, MD=None, pth=None, threads=10, fname="results.pickle", **kwargs):
        if pth is None:
            if MD is not None:
                self.pth = MD.base_pth
        else:
            self.pth = pth

        pth = self.pth

        if fname in listdir(pth):
            r = results.load(pth, fname=fname)
            self.__dict__.update(r.__dict__)
            self.pth = pth
            for p in self.PosLbls.values():
                p.pth = pth
            print("\nloaded results from pickle file")
        else:
            if MD is None:
                MD = Metadata(pth)

            if MD().empty:
                raise AssertionError("No metadata found in supplied path")

            self.PosNames = natsorted(MD.unique("Position"))
            self.channels = MD.unique("Channel")
            self.acq = MD.unique("acq")
            self.frames = natsorted(MD.unique("frame"))
            self.groups = natsorted(MD.unique("group"))
            self.PosLbls = {}

    def __call__(self):
        print("Results object for path to experiment in path: \n " + self.pth)
        print("\nAvailable channels are : " + ", ".join(list(self.channels)) + ".")
        print(
            "\nPositions already segmented are : "
            + ", ".join(natsorted([str(a) for a in self.PosLbls.keys()]))
        )
        print(
            "\nAvailable positions : "
            + ", ".join(list([str(a) for a in self.PosNames]))
            + "."
        )
        print("\nAvailable frames : " + str(len(self.frames)) + ".")

    @alias(
        {
            "pos": "Position",
            "Pos": "Position",
            "position": "Position",
            "p": "Position",
            "frame": "frames",
            "Frame": "frames",
            "f": "frames",
        }
    )
    def setPosLbls(self, MD=None, groups=None, Position=None, override=True, **kwargs):
        """
        function to create PosLbl instances.

        Parameters
        ----------
        MD - experiment metadata
        Position - [All Positions] position name or list of position names

        Segmentation parameters
        -----------------------
        NucChannel : ['DeepBlue'] list or str name of nuclear channel
        CytoChannel : optional cytoplasm channel
        segment_type : ['watershed'] function to use for segmentatiion
        **kwargs : specific args for segmentation function, anything that goes into FrameLbl
        Threads : how many threads to use for parallel execution. Limited to ~6 for GPU based segmentation and 128 for CPU (but don't use all 128)
        """
        if MD is None:
            MD = Metadata(self.pth)
        if groups is not None:
            assert np.all(np.isin(groups, self.groups)), (
                "some provided groups don't exist, try %s"
                % ", ".join(list(self.groups))
            )
            Position = MD.unique("Position", group=groups)
        if Position is None:
            Position = self.PosNames

        elif type(Position) is not list:
            Position = [Position]

        for p in Position:
            print("\nProcessing position " + str(p))
            self.PosLbls.update(
                {p: PosLbl(MD=MD, Pos=p, pth=MD.base_pth, override=override, **kwargs)}
            )
        self.save()

    @alias(
        {
            "pos": "Position",
            "Pos": "Position",
            "position": "Position",
            "p": "Position",
        }
    )
    def segment_and_extract_features(
        self, MD=None, groups=None, Position=None, override=True, **kwargs
    ):
        """
        function to create PosLbl instances.

        Parameters
        ----------
        MD - experiment metadata
        Position - [All Positions] position name or list of position names

        Segmentation parameters
        -----------------------
        NucChannel : ['DeepBlue'] list or str name of nuclear channel
        CytoChannel : optional cytoplasm channel
        segment_type : ['watershed'] function to use for segmentatiion
        **kwargs : specific args for segmentation function, anything that goes into FrameLbl
        Threads : how many threads to use for parallel execution. Limited to ~6 for GPU based segmentation and 128 for CPU (but don't use all 128)
        """
        return self.setPosLbls(
            MD=MD, groups=groups, Position=Position, override=override, **kwargs
        )

    def calculate_tracks(self, Position=None, save=True, split=True, **kwargs):
        """
        function to calculate tracks for a PosLbl instance.

        Parameters
        ----------
        Position : [All Positions] position name or list of position names
        kwargs that go into tracking helper functions: search_radius, params (list of tuples, (channel, weight)), maxStep for skip ,maxAmpRatio for skip, mintracklength
        """
        pos = Position

        if np.all(pos == None):
            pos = list(self.PosLbls.keys())
        pos = pos if isinstance(pos, (list, np.ndarray)) else [pos]
        assert any(elem in self.PosLbls.keys() for elem in pos), (
            str(pos) + " not segmented yet"
        )
        for p in pos:
            print("Calculating tracks for position " + str(p))
            self.PosLbls[p].trackcells(split=split, **kwargs)
        if save:
            self.save()

    @alias(
        {
            "Position": "pos",
            "Pos": "pos",
            "position": "pos",
            "p": "pos",
        }
    )
    def tracks(self, pos):
        """
        Wrapper for PosLbl.get_track
        Parameters
        ----------
        pos : position name

        Returns
        -------
        function handle for track generator
        """
        assert pos in self.PosLbls.keys(), str(pos) + " not segmented yet"
        assert self.PosLbls[pos]._tracked, str(pos) + " not tracked yet"
        return self.PosLbls[pos].get_track

    @alias(
        {
            "Position": "pos",
            "Pos": "pos",
            "position": "pos",
            "p": "pos",
        }
    )
    def tracklist(self, pos=None):
        """
        Function to consolidate tracks from different positions
        Parameters
        ----------
        pos : [All positions] position name, list of position names

        Returns
        -------
        List of tracks in pos
        """
        if pos is None:
            pos = list(self.PosLbls.keys())
        pos = pos if isinstance(pos, list) or isinstance(pos, np.ndarray) else [pos]
        ts = []
        for p in pos:
            t0 = self.tracks(p)
            ([ts.append(t0(i)) for i in np.arange(t0(0).numtracks)])
        return ts

    @alias(
        {
            "Position": "pos",
            "Pos": "pos",
            "position": "pos",
            "p": "pos",
        }
    )
    def show_tracks(self, pos, J=None, **kwargs):
        """
        Wrapper for PosLbl.plot_tracks
        Parameters
        ----------
        pos : position name
        J : track indices - plots all tracks if not provided
        Zindex : [0]


        Draws image stks with overlaying tracks in current napari viewer

        """

        assert pos in self.PosLbls.keys(), str(pos) + " not segmented yet"
        tracks = self.PosLbls[pos].plot_tracks(J=J, **kwargs)
        return tracks

    @alias(
        {
            "Position": "pos",
            "Pos": "pos",
            "position": "pos",
            "p": "pos",
            "channel": "Channel",
            "ch": "Channel",
            "c": "Channel",
        }
    )
    def show_points(self, pos, Channel=None, **kwargs):
        """
        Wrapper for PosLbl.plot_points
        Parameters
        ----------
        pos : position name
        Channel : [DeepBlue] str
        Zindex : [0]

        Draws cells as points in current napari viewer. Color codes for intensity


        """
        if Channel not in self.channels:
            Channel = self.channels[0]
            print("showing channel " + str(Channel))
        assert pos in self.PosLbls.keys(), str(pos) + " not segmented yet"
        points = self.PosLbls[pos].plot_points(Channel=Channel, **kwargs)
        return points

    @alias(
        {
            "Position": "pos",
            "Pos": "pos",
            "position": "pos",
            "p": "pos",
            "frame": "frames",
            "Frame": "frames",
            "f": "frames",
            "channel": "Channel",
            "ch": "Channel",
            "c": "Channel",
        }
    )
    def show_images(self, pos, Channel=None, **kwargs):
        """
        Wrapper for PosLbl.plot_images
        Parameters
        ----------
        pos : position name
        Channel : [DeepBlue] str or list of strings
        Zindex : [0]

        Draws image stks in current napari viewer

        """
        if not isinstance(Channel, list):
            Channel = [Channel]
        Channel = [ch for ch in Channel if ch in self.channels]
        if not Channel:
            Channel = [self.channels[0]]
        print("showing channel " + str(Channel))
        self.PosLbls[pos].plot_images(Channel=Channel, **kwargs)

    @alias(
        {
            "pos": "Position",
            "Pos": "Position",
            "position": "Position",
            "p": "Position",
        }
    )
    def numtracks(self, Position=None):
        """
        Wrapper for PosLbl.numtracks
        Parameters
        ----------
        pos : str / [str] position name

        returns number of tracks per position

        """
        if Position == None:
            Position = list(self.PosNames)
        Position = (
            Position
            if isinstance(Position, list) or isinstance(Position, np.ndarray)
            else [Position]
        )
        ntracks = []
        for pos in Position:
            ntracks.append(self.PosLbls[pos].numtracks)
        return ntracks

    def calculate_spatial_stats(self, Position=None, channels=None, frame=None,
                                stats=None, ffield=True, img=False, n_jobs=4, **kwargs):
        """
        Compute spatial statistics across multiple positions and cache results.

        Parameters
        ----------
        Position : str or list, optional
            Positions to process. Defaults to all segmented positions.
        channels : list, optional
            Channels to use. Defaults to all channels. Auto-correlations and all
            unique cross-pairs (A×B but not both A×B and B×A) are computed.
        frame : int or list, optional
            Frame(s) to process. Defaults to None (all frames aggregated).
        stats : list of str, optional
            Which statistics to compute. Any subset of:
              'radial_corr'    — radial intensity correlation g(r)
              'radial_density' — pair correlation function (geometry only)
              'lengthscale'    — exponential fit λ of g(r); computed automatically
                                 whenever 'radial_corr' is requested
              'mark_variogram' — normalised mark variogram γ̃(r)
            Defaults to all four.
        ffield : bool
            Apply flat-field correction (default True).
        img : bool
            Use pixel-level FFT correlation instead of cell-level for
            `radial_corr` and `lengthscale` (default False).
        n_jobs : int
            Number of parallel threads (default 4).

        Returns
        -------
        dict stored on self.spatial_stats with keys matching requested stats.
        Curve stats (radial_corr, radial_density, mark_variogram) are stored as
        nested dicts  {pos: {(ch_i, ch_j): result}} or {pos: result}.
        lengthscale is stored as both a nested dict and a summary DataFrame.
        """
        import itertools
        import pandas as pd
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm

        _valid = {'radial_corr', 'radial_density', 'lengthscale', 'mark_variogram'}
        if stats is None:
            stats = list(_valid)
        else:
            stats = list(stats)
            unknown = set(stats) - _valid
            if unknown:
                raise ValueError(f"Unknown stats: {unknown}. Valid: {_valid}")

        # lengthscale requires radial_corr
        if 'lengthscale' in stats and 'radial_corr' not in stats:
            stats.append('radial_corr')

        if Position is None:
            Position = list(self.PosLbls.keys())
        elif not isinstance(Position, (list, np.ndarray)):
            Position = [Position]
        missing = [p for p in Position if p not in self.PosLbls]
        if missing:
            raise ValueError(f"Positions not segmented: {missing}")

        if channels is None:
            channels = list(self.channels)
        elif not isinstance(channels, list):
            channels = [channels]

        # auto + upper-triangle cross pairs
        ch_pairs = [(ch, ch) for ch in channels]
        ch_pairs += [(a, b) for a, b in itertools.combinations(channels, 2)]

        def _process_pos(pos):
            P = self.PosLbls[pos]
            pos_result = {}
            if 'radial_density' in stats:
                pos_result['radial_density'] = P.radial_density(frame=frame)

            if 'radial_corr' in stats:
                pos_result['radial_corr'] = {}
                if 'lengthscale' in stats:
                    pos_result['lengthscale'] = {}
                for (ch_i, ch_j) in ch_pairs:
                    chj = None if ch_j == ch_i else ch_j
                    pos_result['radial_corr'][(ch_i, ch_j)] = P.radial_corr(
                        ch_i, ch_j=chj, frame=frame, img=img, ffield=ffield)
                    if 'lengthscale' in stats:
                        pos_result['lengthscale'][(ch_i, ch_j)] = \
                            P.fit_corr_lengthscale(ch_i, ch_j=chj, frame=frame,
                                                   img=img, ffield=ffield)

            if 'mark_variogram' in stats:
                pos_result['mark_variogram'] = {}
                for (ch_i, ch_j) in ch_pairs:
                    chj = None if ch_j == ch_i else ch_j
                    pos_result['mark_variogram'][(ch_i, ch_j)] = P.mark_variogram(
                        ch_i, ch_j=chj, frame=frame, img=img, ffield=ffield)

            return pos, pos_result

        compiled = {s: {} for s in stats}
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            futures = {pool.submit(_process_pos, pos): pos for pos in Position}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc='spatial stats', unit='pos'):
                pos, pos_result = fut.result()
                for stat, data in pos_result.items():
                    compiled[stat][pos] = data

        # build lengthscale summary DataFrame
        if 'lengthscale' in compiled:
            rows = []
            for pos, fits in compiled['lengthscale'].items():
                for (ch_i, ch_j), fit in fits.items():
                    if fit is not None:
                        rows.append(dict(position=pos, ch_i=ch_i, ch_j=ch_j,
                                         lam=fit.get('lambda'), lam_err=fit.get('lambda_err'),
                                         r_sq=fit.get('r_sq')))
            compiled['lengthscale_summary'] = pd.DataFrame(rows)

        if not hasattr(self, 'spatial_stats'):
            self.spatial_stats = {}
        self.spatial_stats.update(compiled)
        self.save()
        return self.spatial_stats

    def spatial_report(self, groups=None, stats=None, channels=None, panel_size=(4, 3)):
        """
        Plot a multi-figure report comparing spatial statistics across positions.

        Parameters
        ----------
        groups : dict, optional
            {label: [list of positions]} to compare. If None, uses the 'group'
            field from the metadata if available, otherwise each position is its
            own group. Groups with more than one member are shown as mean ± SEM.
        stats : list of str, optional
            Which stats to include. Defaults to all computed stats.
        channels : list, optional
            Restrict to specific channels. Defaults to all channels in spatial_stats.
        panel_size : tuple
            (width, height) in inches per subplot panel.

        Returns
        -------
        list of matplotlib Figure objects
        """
        from collections import defaultdict

        import matplotlib.pyplot as plt
        import numpy as np

        if not hasattr(self, 'spatial_stats') or not self.spatial_stats:
            raise RuntimeError(
                "No spatial statistics found. Run R.calculate_spatial_stats() first."
            )

        _curve_stats = ['radial_corr', 'radial_density', 'mark_variogram']
        _all_stats = _curve_stats + ['lengthscale']
        available = set(self.spatial_stats) - {'lengthscale_summary'}

        if stats is None:
            stats = [s for s in _all_stats if s in available]
        else:
            missing = [s for s in stats if s not in available]
            if missing:
                raise ValueError(
                    f"Stats not computed: {missing}. Run calculate_spatial_stats() first."
                )

        # collect all positions that appear in any stat
        all_positions = set()
        for s in stats:
            if s in self.spatial_stats:
                all_positions.update(self.spatial_stats[s].keys())
        all_positions = sorted(all_positions)

        # resolve groups
        if groups is not None:
            pass  # use as supplied
        else:
            # try metadata group field
            try:
                MD = Metadata(self.pth)
                pos_to_group = {}
                for pos in all_positions:
                    grp_vals = MD.unique('group', Position=pos)
                    grp = grp_vals[0] if (grp_vals and grp_vals[0] is not None) else None
                    pos_to_group[pos] = grp if grp else pos
                if len(set(pos_to_group.values())) < len(all_positions):
                    g = defaultdict(list)
                    for pos, grp in pos_to_group.items():
                        g[grp].append(pos)
                    groups = dict(g)
                else:
                    groups = {pos: [pos] for pos in all_positions}
            except Exception:
                groups = {pos: [pos] for pos in all_positions}

        n_groups = len(groups)
        use_sem = any(len(v) > 1 for v in groups.values())
        colors = plt.cm.tab10(np.linspace(0, 0.9, max(n_groups, 1)))

        # determine channel pairs from stored data
        def _detect_ch_pairs(stat_name):
            d = self.spatial_stats.get(stat_name, {})
            for pos_data in d.values():
                if isinstance(pos_data, dict):
                    return list(pos_data.keys())
            return []

        if channels is not None:
            ch_set = set(channels)
            raw_pairs = _detect_ch_pairs('radial_corr') or _detect_ch_pairs('mark_variogram')
            ch_pairs = [(a, b) for a, b in raw_pairs
                        if a in ch_set and b in ch_set]
        else:
            ch_pairs = _detect_ch_pairs('radial_corr') or _detect_ch_pairs('mark_variogram')

        def _pair_label(ch_i, ch_j):
            return ch_i if ch_i == ch_j else f'{ch_i} × {ch_j}'

        def _subplot_grid(n, panel_size):
            ncols = min(n, 3)
            nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
                squeeze=False,
            )
            return fig, axes, nrows, ncols

        def _plot_curves(ax, stat_name, key, groups, colors, ref=None):
            """Plot mean ± SEM curves for one channel pair across groups."""
            stat_data = self.spatial_stats[stat_name]
            for g_idx, (label, pos_list) in enumerate(groups.items()):
                curves = []
                for pos in pos_list:
                    pos_data = stat_data.get(pos)
                    if pos_data is None:
                        continue
                    entry = pos_data.get(key) if key is not None else pos_data
                    if entry is not None:
                        curves.append(entry)
                if not curves:
                    continue
                r = curves[0]['r']
                g_mat = np.array([c['g'] for c in curves], dtype=float)
                mean_g = np.nanmean(g_mat, axis=0)
                color = colors[g_idx]
                ax.plot(r, mean_g, color=color, label=label, lw=1.5)
                if len(curves) > 1 and use_sem:
                    sem_g = np.nanstd(g_mat, axis=0) / np.sqrt(len(curves))
                    ax.fill_between(r, mean_g - sem_g, mean_g + sem_g,
                                    color=color, alpha=0.2)
            if ref is not None:
                ax.axhline(ref, color='k', lw=0.8, ls='--', zorder=0)

        figs = []

        # --- Radial correlation ---
        if 'radial_corr' in stats and ch_pairs:
            fig, axes, nrows, ncols = _subplot_grid(len(ch_pairs), panel_size)
            fig.suptitle('Radial Correlation  g(r)', fontsize=12, fontweight='bold')
            for idx, (ch_i, ch_j) in enumerate(ch_pairs):
                ax = axes[idx // ncols][idx % ncols]
                _plot_curves(ax, 'radial_corr', (ch_i, ch_j), groups, colors, ref=0)
                ax.set_title(_pair_label(ch_i, ch_j), fontsize=9)
                ax.set_xlabel('r (µm)')
                ax.set_ylabel('g(r)')
                if n_groups > 1:
                    ax.legend(fontsize=7)
            for idx in range(len(ch_pairs), nrows * ncols):
                axes[idx // ncols][idx % ncols].set_visible(False)
            fig.tight_layout()
            figs.append(fig)

        # --- Pair correlation (radial density) ---
        if 'radial_density' in stats:
            fig, ax = plt.subplots(1, 1, figsize=panel_size)
            fig.suptitle('Pair Correlation Function  g(r)', fontsize=12, fontweight='bold')
            _plot_curves(ax, 'radial_density', None, groups, colors, ref=1.0)
            ax.set_xlabel('r (µm)')
            ax.set_ylabel('g(r)')
            if n_groups > 1:
                ax.legend(fontsize=7)
            fig.tight_layout()
            figs.append(fig)

        # --- Lengthscale: curves + fits + bar chart ---
        if 'lengthscale' in stats and 'lengthscale_summary' in self.spatial_stats:
            df = self.spatial_stats['lengthscale_summary']
            pairs_to_plot = ch_pairs if ch_pairs else list(
                zip(df['ch_i'], df['ch_j'])
            )
            n_pairs = len(pairs_to_plot)
            ncols = min(n_pairs, 3)
            curve_rows = (n_pairs + ncols - 1) // ncols
            # curve rows + one bar row
            fig = plt.figure(figsize=(
                panel_size[0] * ncols,
                panel_size[1] * curve_rows + panel_size[1],
            ))
            fig.suptitle('Correlation Lengthscale  λ (µm)', fontsize=12, fontweight='bold')
            import matplotlib.gridspec as gridspec
            gs = gridspec.GridSpec(
                curve_rows + 1, ncols, figure=fig,
                height_ratios=[1] * curve_rows + [1],
            )

            ls_data = self.spatial_stats.get('lengthscale', {})
            rc_data = self.spatial_stats.get('radial_corr', {})

            for idx, (ch_i, ch_j) in enumerate(pairs_to_plot):
                ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
                for g_idx, (label, pos_list) in enumerate(groups.items()):
                    color = colors[g_idx]
                    # collect g(r) curves and fit params
                    g_curves, fits = [], []
                    for pos in pos_list:
                        rc = rc_data.get(pos, {}).get((ch_i, ch_j))
                        ft = ls_data.get(pos, {}).get((ch_i, ch_j))
                        if rc is not None:
                            g_curves.append(rc)
                        if ft is not None:
                            fits.append(ft)
                    if g_curves:
                        r = g_curves[0]['r']
                        g_mat = np.array([c['g'] for c in g_curves], dtype=float)
                        mean_g = np.nanmean(g_mat, axis=0)
                        ax.plot(r, mean_g, color=color, lw=1.5)
                        if len(g_curves) > 1 and use_sem:
                            sem_g = np.nanstd(g_mat, axis=0) / np.sqrt(len(g_curves))
                            ax.fill_between(r, mean_g - sem_g, mean_g + sem_g,
                                            color=color, alpha=0.2)
                    if fits:
                        mean_A   = float(np.nanmean([f['A'] for f in fits]))
                        mean_lam = float(np.nanmean([f['lambda'] for f in fits]))
                        mean_C   = float(np.nanmean([f['C'] for f in fits]))
                        r_min_fit = float(np.nanmean([f['r_min'] for f in fits]))
                        r_fit = np.linspace(r_min_fit, fits[0]['fit_r'][-1], 200)
                        curve = mean_A * np.exp(-r_fit / mean_lam) + mean_C
                        ax.plot(r_fit, curve, color=color, lw=1.2, ls='--',
                                label=f'{label}  λ={mean_lam:.0f} µm')
                ax.axhline(0, color='k', lw=0.5, ls=':', zorder=0)
                ax.set_title(_pair_label(ch_i, ch_j), fontsize=9)
                ax.set_xlabel('r (µm)')
                ax.set_ylabel('g(r)')
                ax.legend(fontsize=7)

            # hide unused curve panels
            for idx in range(n_pairs, curve_rows * ncols):
                fig.add_subplot(gs[idx // ncols, idx % ncols]).set_visible(False)

            # bar chart spanning full bottom row
            ax_bar = fig.add_subplot(gs[curve_rows, :])
            width = 0.8 / max(n_groups, 1)
            for g_idx, (label, pos_list) in enumerate(groups.items()):
                sub = df[df['position'].isin(pos_list)]
                lam_means, lam_errs = [], []
                for ch_i, ch_j in pairs_to_plot:
                    row = sub[(sub['ch_i'] == ch_i) & (sub['ch_j'] == ch_j)]['lam']
                    lam_means.append(float(row.mean()) if not row.empty else np.nan)
                    lam_errs.append(float(row.sem()) if len(row) > 1 else 0.0)
                x = np.arange(n_pairs) + g_idx * width
                ax_bar.bar(x, lam_means,
                           width=width, color=colors[g_idx], label=label,
                           yerr=lam_errs if use_sem else None,
                           capsize=3, error_kw={'elinewidth': 1})
            ax_bar.set_xticks(np.arange(n_pairs) + width * (n_groups - 1) / 2)
            ax_bar.set_xticklabels([_pair_label(a, b) for a, b in pairs_to_plot],
                                   rotation=30, ha='right', fontsize=8)
            ax_bar.set_ylabel('λ (µm)')
            if n_groups > 1:
                ax_bar.legend(fontsize=7)

            fig.tight_layout()
            figs.append(fig)

        # --- Mark variogram ---
        if 'mark_variogram' in stats and ch_pairs:
            fig, axes, nrows, ncols = _subplot_grid(len(ch_pairs), panel_size)
            fig.suptitle('Mark Variogram  γ̃(r)', fontsize=12, fontweight='bold')
            stat_data = self.spatial_stats['mark_variogram']
            for idx, (ch_i, ch_j) in enumerate(ch_pairs):
                ax = axes[idx // ncols][idx % ncols]
                # compute grand-mean reference across all positions
                all_refs = [
                    stat_data[pos][(ch_i, ch_j)].get('variogram_ref', 1.0)
                    for pos in stat_data
                    if (ch_i, ch_j) in stat_data.get(pos, {})
                ]
                ref = float(np.nanmean(all_refs)) if all_refs else 1.0
                _plot_curves(ax, 'mark_variogram', (ch_i, ch_j), groups, colors, ref=ref)
                ax.set_title(_pair_label(ch_i, ch_j), fontsize=9)
                ax.set_xlabel('r (µm)')
                ax.set_ylabel('γ̃(r)')
                if n_groups > 1:
                    ax.legend(fontsize=7)
            for idx in range(len(ch_pairs), nrows * ncols):
                axes[idx // ncols][idx % ncols].set_visible(False)
            fig.tight_layout()
            figs.append(fig)

        return figs

    def save(self, fname="results.pickle"):
        """
        save results
        """
        # save individual positions and clear data
        for pos in self.PosLbls.keys():
            self.PosLbls[pos].save()
            self.PosLbls[pos].framelabels = []
        # save results object without position data
        with open(join(self.pth, fname), "wb") as dbfile:
            cloudpickle.dump(self, dbfile)
            print("\nsaved results.")
        # replace position data
        for pos in self.PosLbls.keys():
            self.PosLbls[pos].load()

    @classmethod
    def load(cls, pth, fname="results.pickle"):
        """
        load results
        """
        with open(join(pth, fname), "rb") as dbfile:
            r = dill.load(dbfile)
        # replace position data
        for pos in r.PosLbls.keys():
            if r.PosLbls[pos].framelabels == []:
                r.PosLbls[pos].load()
        return r

    @alias(
        {
            "pos": "Position",
            "Pos": "Position",
            "position": "Position",
            "p": "Position",
            "Channel": "channel",
            "ch": "channel",
            "c": "channel",
            "Ch": "channel",
        }
    )
    def property_matrix(
        self, Position=None, prop="area", channel=None, periring=False, keep_only=False
    ):
        """
        Parameters
        ----------
        pos : str - Position
        prop : str - Property to return
        channel : str - for intensity based properties, channel name.
        periring : For intensity based features only. Perinuclear ring values.
        keep_only : {[False], True}

        wrapper for PosLbls.property_matrix property prop for all tracks in csv form with coma delimiter [N tracks x M timepoints x L dimensions of property]
        """
        return self.PosLbls[Position].property_matrix(
            prop=prop, channel=channel, periring=periring, keep_only=keep_only
        )

    @alias(
        {
            "pos": "Position",
            "Pos": "Position",
            "position": "Position",
            "p": "Position",
            "Channel": "channel",
            "ch": "channel",
            "c": "channel",
        }
    )
    def prop_to_csv(
        self, Position=None, prop="area", channel=None, periring=False, keep_only=False
    ):
        """
        Parameters
        ----------
        pos : str - Position
        prop : str - Property to return
        channel : str - for intensity based properties, channel name.
        periring : For intensity based features only. Perinuclear ring values.
        keep_only : {[False], True}

        saves property prop for all tracks in csv form with coma delimiter [[N tracks*L dimensions of property] x M timepoints ]
        """
        import os

        from numpy import savetxt

        csvfolder = os.path.join(self.pth, "csvs" + os.path.sep)
        if not os.path.exists(csvfolder):
            os.makedirs(csvfolder)
        if channel == None:
            filename = os.path.join(
                csvfolder, "prop_" + prop + "_pos_" + Position + ".csv"
            )
        else:
            filename = os.path.join(
                csvfolder,
                "prop_" + prop + "_ch_" + channel + "_pos_" + Position + ".csv",
            )

        A = self.property_matrix(
            Position=Position,
            prop=prop,
            channel=channel,
            periring=periring,
            keep_only=keep_only,
        )
        A = np.reshape(A, newshape=(-1, A.shape[1]))
        savetxt(filename, A, delimiter=",")

    def track_explorer(R, keep_only=False):
        """
        Track explorer app. Written using magicgui (Thanks @tlambert03!)

        Allows one to easily browse through tracks, plot the data and see the corresponding movies. Can also be used for curation and quality control.

        Parameters:
        keep_only : [False] Bool - If true, only tracks that are in PosLbl.track_to_use will be loaded in a given position. This can be used to filter unwanted tracks before examining for quality with the explorer.
        """
        from typing import List

        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np
        from magicgui import magicgui
        from magicgui.widgets import Checkbox, Container, PushButton
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from natsort import natsorted
        from scipy import stats

        from oyLabImaging.Processing.imvisutils import get_or_create_viewer

        cmaps = ["cyan", "magenta", "yellow", "red", "green", "blue"]
        viewer = get_or_create_viewer()

        matplotlib.use("Agg")
        mpl_fig = plt.figure()
        ax = mpl_fig.add_subplot(111)

        fc = FigureCanvasQTAgg(mpl_fig)

        # attr_list = ['area', 'convex_area','centroid','perimeter','eccentricity','solidity','inertia_tensor_eigvals', 'orientation'] #todo: derive list from F regioprops

        position = list(natsorted(R.PosLbls.keys()))[0]
        PosLbl0 = R.PosLbls[position]

        attr_list = [
            f
            for f in list(PosLbl0.framelabels[0].regionprops)
            if not f.startswith(("mean", "median", "max", "min", "90th", "slice"))
        ]
        attr_cmap = plt.cm.get_cmap("tab20b", len(attr_list)).colors

        @magicgui(
            auto_call=True,
            position={"choices": natsorted([a for a in R.PosLbls.keys()])},
            track_id={
                "choices": range(
                    R.PosLbls[natsorted([a for a in R.PosLbls.keys()])[0]].numtracks
                )
            },
            channels={"widget_type": "Select", "choices": list(R.channels)},
            features={"widget_type": "Select", "choices": attr_list},
        )
        def widget(
            position: List[str], track_id: int, channels: List[str], features: List
        ):
            # preserving these parameters for things that the graphing function
            # needs... so that anytime this is called we have to graph.
            ...
            # do your graphing here
            PosLbl = R.PosLbls[position]
            if PosLbl.numtracks:
                if type(track_id) != int:
                    if keep_only:
                        J = PosLbl.track_to_use
                    else:
                        J = range(PosLbl.numtracks)
                    track_id = J[0]
                t0 = PosLbl.get_track(track_id)
            ax.cla()
            ax.set_xlabel("Timepoint")
            ax.set_ylabel("kAU")
            ch_choices = widget.channels.choices
            if PosLbl.numtracks:
                for ch in channels:
                    ax.plot(
                        t0.T,
                        stats.zscore(t0.mean(ch)),
                        color=cmaps[ch_choices.index(ch)],
                    )

                f_choices = widget.features.choices
                for ch in features:
                    feat_to_plot = eval("t0.prop('" + ch + "')")
                    if np.ndim(feat_to_plot) == 1:
                        ax.plot(
                            t0.T,
                            stats.zscore(feat_to_plot, nan_policy="omit"),
                            "--",
                            color=attr_cmap[f_choices.index(ch)],
                            alpha=0.33,
                        )
                    else:
                        mini_cmap = plt.cm.get_cmap("jet", np.shape(feat_to_plot)[1])
                        for dim in np.arange(np.shape(feat_to_plot)[1]):
                            ax.plot(
                                t0.T,
                                stats.zscore(feat_to_plot[:, dim], nan_policy="omit"),
                                "--",
                                color=mini_cmap(dim),
                                alpha=0.33,
                            )
                            # ax.plot(t0.T, feat_to_plot[:,dim],'--', color=mini_cmap(dim), alpha=0.25)

            ax.legend(channels + features)
            fc.draw()

        @widget.position.changed.connect
        def _on_position_changed():
            PosLbl = R.PosLbls[widget.position.value]
            try:
                PosLbl.track_to_use
            except:
                PosLbl.track_to_use = []
            viewer.layers.clear()
            # update track_id choices - bug in choices:
            if keep_only:
                J = PosLbl.track_to_use
            else:
                J = range(PosLbl.numtracks)

            widget.track_id.choices = J
            if PosLbl.numtracks:
                widget.track_id.value = J[0]
            # update keep_btn value
            # keep_btn.value= widget.track_id.value in PosLbl.track_to_use

        @widget.track_id.changed.connect
        def _on_track_changed(new_track: int):
            PosLbl = R.PosLbls[widget.position.value]
            viewer.layers.clear()
            keep_btn.value = widget.track_id.value in PosLbl.track_to_use
            # print("you cahnged to ", new_track)

        movie_btn = PushButton(text="Movie")
        widget.insert(1, movie_btn)

        @movie_btn.clicked.connect
        def _on_movie_clicked():
            PosLbl = R.PosLbls[widget.position.value]
            channels = widget.channels.get_value()
            track_id = widget.track_id.get_value()
            t0 = PosLbl.get_track(track_id)
            viewer.layers.clear()
            ch_choices = widget.channels.choices
            t0.show_movie(
                Channel=channels, cmaps=[cmaps[ch_choices.index(ch)] for ch in channels]
            )

        btn = PushButton(text="NEXT")
        widget.insert(-1, btn)

        @btn.clicked.connect
        def _on_next_clicked():
            choices = widget.track_id.choices
            current_index = choices.index(widget.track_id.value)
            widget.track_id.value = choices[(current_index + 1) % (len(choices))]

        PosLbl = R.PosLbls[widget.position.value]
        try:
            PosLbl.track_to_use
        except:
            PosLbl.track_to_use = []

        keep_btn = Checkbox(text="Keep")
        keep_btn.value = widget.track_id.value in PosLbl.track_to_use
        widget.append(keep_btn)

        @keep_btn.clicked.connect
        def _on_keep_btn_clicked(value: bool):
            # print("keep is now", value)
            PosLbl = R.PosLbls[widget.position.value]
            if value == True:
                if widget.track_id.value not in PosLbl.track_to_use:
                    PosLbl.track_to_use.append(widget.track_id.value)
            if value == False:
                if widget.track_id.value in PosLbl.track_to_use:
                    PosLbl.track_to_use.remove(widget.track_id.value)
            R.PosLbls[widget.position.value] = PosLbl

        # widget.native
        # ... points to the underlying backend widget

        container = Container(layout="horizontal")

        # magicgui container expect magicgui objects
        # but we can access and modify the underlying QLayout
        # https://doc.qt.io/qt-5/qlayout.html#addWidget

        layout = container.native.layout()

        layout.addWidget(fc)
        layout.addWidget(widget.native)  # adding native, because we're in Qt

        # container.show(run=True)
        # OR

        viewer.window.add_dock_widget(container)
        # run()
        matplotlib.use("Qt5Agg")


class frameData(object):
    @alias(
        {
            "pos": "Position",
            "Pos": "Position",
            "position": "Position",
            "p": "Position",
            "frames": "frame",
            "Frame": "frame",
            "f": "frame",
        }
    )
    def __init__(self, outer, frame=0, Position=None, label=None):
        assert frame in outer.frames, "Available frames are " + ", ".join(
            [str(f) for f in outer.frames]
        )
        if Position is None:
            if label:
                self.Position = [pn for pn in outer.PosNames if pn.startswith(label)]
            else:
                self.Position = outer.PosNames
        else:
            self.Position = Position
        self.frame = frame
        self._outer = outer

    @property
    def centroid_um(self):
        frame = self.frame
        a = [
            self._outer.PosLbls[pn].centroid_um[frame]
            for pn in self.Position
            if self._outer.PosLbls[pn].centroid_um[frame].ndim == 2
        ]
        a = np.concatenate(a)
        return a

    @property
    def weighted_centroid_um(self):
        frame = self.frame
        a = [
            self._outer.PosLbls[pn].weighted_centroid_um[frame]
            for pn in self.Position
            if self._outer.PosLbls[pn].weighted_centroid_um[frame].ndim == 2
        ]
        a = np.concatenate(a)
        return a

    @property
    def area(self):
        frame = self.frame
        a = [self._outer.PosLbls[pn].area[frame] for pn in self.Position]
        a = np.concatenate(a)
        return a

    @property
    def cellposition(self):
        frame = self.frame
        return np.concatenate(
            [[pn] * self._outer.PosLbls[pn].num[frame] for pn in self.Position]
        )

    @property
    def cellperposition(self):
        frame = self.frame
        return [self._outer.PosLbls[pn].num[frame] for pn in self.Position]

    def mean(self, ch, periring=False):
        ch = ch if isinstance(ch, (list, np.ndarray)) else [ch]
        frame = self.frame
        a = [
            self._outer.PosLbls[pn]
            .framelabels[frame]
            .regionprops[["".join(["mean_", c, "_periring" * periring]) for c in ch]]
            for pn in self.Position
            if self._outer.PosLbls[pn].num
        ]
        a = np.concatenate(a)
        return a

    def median(self, ch, periring=False):
        ch = ch if isinstance(ch, (list, np.ndarray)) else [ch]
        frame = self.frame
        a = [
            self._outer.PosLbls[pn]
            .framelabels[frame]
            .regionprops[["".join(["median_", c, "_periring" * periring]) for c in ch]]
            for pn in self.Position
            if self._outer.PosLbls[pn].num
        ]
        a = np.concatenate(a)
        return a

    def minint(self, ch, periring=False):
        ch = ch if isinstance(ch, (list, np.ndarray)) else [ch]
        frame = self.frame
        a = [
            self._outer.PosLbls[pn]
            .framelabels[frame]
            .regionprops[["".join(["min_", c, "_periring" * periring]) for c in ch]]
            for pn in self.Position
            if self._outer.PosLbls[pn].num
        ]
        a = np.concatenate(a)
        return a

    def maxint(self, ch, periring=False):
        ch = ch if isinstance(ch, (list, np.ndarray)) else [ch]
        frame = self.frame
        a = [
            self._outer.PosLbls[pn]
            .framelabels[frame]
            .regionprops[["".join(["max_", c, "_periring" * periring]) for c in ch]]
            for pn in self.Position
            if self._outer.PosLbls[pn].num
        ]
        a = np.concatenate(a)
        return a

    def ninetyint(self, ch, periring=False):
        ch = ch if isinstance(ch, (list, np.ndarray)) else [ch]
        frame = self.frame
        a = [
            self._outer.PosLbls[pn]
            .framelabels[frame]
            .regionprops[["".join(["90th_", c, "_periring" * periring]) for c in ch]]
            for pn in self.Position
            if self._outer.PosLbls[pn].num
        ]
        a = np.concatenate(a)
        return a

    def _calculate_pointmat_worldunits(self):
        """
        helper function, calculate points in a napari-friendly way
        """
        a = []
        [
            a.append((np.pad(cen, ((0, 0), (1, 0)), constant_values=i)))
            for i, cen in enumerate(self.centroid_um)
            if np.any(cen)
        ]
        try:
            _pointmatrix = np.concatenate(a)
        except:
            _pointmatrix = []
        return _pointmatrix
