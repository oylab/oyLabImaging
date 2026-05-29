# PosLbl module for microscopy data.
# Contains all single TP FrameLbls for a specific well/position
# Deals with tracking
# AOY

import sys
from functools import partial
import warnings
import os

import lap
import multiprocess as mp  # import Pool, set_start_method

import cloudpickle
import dill

mp.set_start_method("spawn", force=True)

import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

from oyLabImaging import Metadata
from oyLabImaging.Processing import FrameLbl

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Spatial results cache
# ─────────────────────────────────────────────────────────────────────────────

def _hint_kw(**params_with_defaults) -> str:
    """Return ', key=val, ...' for params whose value differs from their default."""
    parts = [f"{k}={repr(v)}" for k, (v, d) in params_with_defaults.items() if v != d]
    return (', ' + ', '.join(parts)) if parts else ''


def _frame_key(frame) -> str:
    """Normalise a frame argument into a cache-key fragment."""
    if frame is None:
        return "f=all"
    if isinstance(frame, (int, np.integer)):
        return f"f={int(frame)}"
    return "f=" + ",".join(str(int(f)) for f in frame)


class CachedResult:
    """A cached spatial result with human-readable retrieval / plot hints.

    Usage
    -----
    P.spatial['key']          # shows a one-line summary
    P.spatial['key']()        # prints retrieve / plot call strings, returns data
    P.spatial['key'].data     # raw result dict (or list of dicts)
    """

    def __init__(self, data, method_str: str, plot_str: str = None):
        self.data = data
        self._method_str = method_str
        self._plot_str = plot_str

    def __call__(self):
        print(f"Retrieve : P{self._method_str}")
        if self._plot_str:
            print(f"Plot     : P{self._plot_str}")
        return self.data

    def __repr__(self):
        return "CachedResult — call () for hints, .data for raw result"


class SpatialResults(dict):
    """Per-position cache of spatial analysis results.

    Populated automatically whenever a spatial method is called on a PosLbl.
    Persists across sessions because it is pickled with the PosLbl object.

    Usage
    -----
    P.spatial                      # show what is cached
    P.spatial['key']()             # print retrieve / plot hints, return data
    P.spatial['key'].data          # raw result dict
    P.spatial.clear()              # wipe the entire cache
    """

    def __repr__(self):
        if not self:
            return "SpatialResults (empty) — run any spatial method to populate."
        header = f"  {'Key':<58}  Type"
        sep    = "  " + "-" * 68
        rows   = [f"  {k[:58]:<58}  [{k.split('|')[0]}]" for k in self]
        footer = "\nCall P.spatial['key']() for retrieval / plot hints."
        return "\n".join([header, sep] + rows + [footer])


class PosLbl(object):
    """
    Class for data from a single position (multi timepoint, position, single experiment, multi channel). Handles image tracking.
    Parameters
    ----------
    MD : relevant metadata OR
    pth : str path to relevant metadata

    Attributes
    ----------
    Pos : position name
    acq : acquisition name
    Zindex : Zindex

    These must specify a unique frame

    Segmentation parameters
    -----------------------
    NucChannel : ['DeepBlue'] str name of nuclear channel
    **kwargs : specific args for segmentation function, anything that goes into FrameLbl
    Threads : how many threads to use for parallel execution. Limited to ~6 for GPU based segmentation and 128 for CPU (but don't use all 128)

    Returns
    -------
    PosLbl instance with segmented FrameLbls

    Class properties
    ----------------
     'acq',
     'area',
     'area_um2',
     'centroid',
     'centroid_um',
     'channels',
     'density',
     'framelabels',
     'frames',
     'maxint',
     'mean',
     'median',
     'minint',
     'ninetyint',
     'num',
     'PixelSize',
     'posname',
     'pth',
     'trackinds',
     'weighted_centroid',
     'weighted_centroid_um'
     'tracks_to_use'

    Class methods
    -------------
     'trackcells',
     'get_track',
     'img',
     'plot_images',
     'plot_points',
     'plot_tracks',

    """

    def __init__(
        self,
        Pos=None,
        MD=None,
        pth=None,
        acq=None,
        frames=None,
        NucChannel=None,
        threads=10,
        register=False,
        ffield=False,
        calculate=True,
        override=False,
        **kwargs,
    ):
        self._registerflag = register
        self._ffieldflag = ffield
        self._tracked = False
        self._splitflag = False
        if not hasattr(self, 'spatial'):
            self.spatial = SpatialResults()

        if pth is None:
            if MD is not None:
                self.pth = MD.base_pth
        else:
            self.pth = pth

        if any([Pos is None]):
            raise ValueError("Please provide position")

        if Pos not in MD.posnames:
            raise AssertionError("Position does not exist in dataset")
        self.posname = Pos

        fname = "PosLbls"
        foldername = os.path.join(self.pth, fname + os.path.sep)
        filename = os.path.join(foldername, Pos + ".pkl")
        if not override:
            if os.path.exists(filename):
                self.load(Pos=self.posname, pth=self.pth, fname=fname)
                calculate = False

        if MD is None:
            MD = Metadata(pth)

        if MD().empty:
            raise AssertionError("No metadata found in supplied path")

        if acq is None:
            acqs = MD.unique("acq", Position=Pos)
            if len(acqs) > 1:
                raise ValueError(f"Multiple acquisitions found for Position='{Pos}': {acqs}. Please specify acq=.")
            self.acq = acqs[0]
        else:
            self.acq = acq

        self.channels = MD.unique("Channel", Position=Pos, acq=self.acq)

        if frames is None:
            self.frames = MD.unique("frame", Position=Pos)
        elif type(frames) is not list:
            self.frames = [frames]
        else:
            self.frames = frames

        # self.PixelSize = MD.unique('PixelSize')[0]

        threads = np.min((threads, len(self.frames)))
        # Create all framelabels for the different TPs. This will segment and measure stuff.
        if calculate:
            with mp.Pool(threads) as ppool:
                frames = list(
                    tqdm(
                        ppool.imap(
                            partial(
                                FrameLbl,
                                MD=MD,
                                pth=pth,
                                Pos=Pos,
                                acq=self.acq,
                                NucChannel=NucChannel,
                                register=self._registerflag,
                                ffield=self._ffieldflag,
                                **kwargs,
                            ),
                            self.frames,
                        ),
                        total=len(self.frames),
                    )
                )

            self.framelabels = np.array(frames)
            self._calculate_pointmat()
            self.save()
            print("\nFinished loading and segmenting position " + str(Pos))

    def __call__(self):
        print("PosLbl object for position " + str(self.posname) + ", acquisition " + str(self.acq) + ".")
        print("\nThe path to the experiment is: \n " + self.pth)
        print("\n " + str(len(self.frames)) + " frames processed.")

        print("\nAvailable channels are : " + ", ".join(list(self.channels)) + ".")

    def __getattr__(self, name):
        if name == 'spatial':
            self.spatial = SpatialResults()
            return self.spatial
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def _cache_spatial(self, key: str, data, method_str: str, plot_str: str = None):
        """Store a spatial result in self.spatial and persist self to disk."""
        self.spatial[key] = CachedResult(data, method_str, plot_str)
        self.save()

    # np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    def save(self, fname="PosLbls"):
        """
        save poslbl
        """
        import os

        foldername = os.path.join(self.pth, fname + os.path.sep)
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        fname = os.path.join(foldername, str(self.posname) + ".pkl")
        with open(fname, "wb") as dbfile:
            cloudpickle.dump(self, dbfile)
            # print("saved Position " + self.posname)
            sys.stdout.write("\r" + "saved Position " + str(self.posname))
            sys.stdout.flush()

    def load(self, Pos=None, pth=None, fname="PosLbls"):
        if Pos is None:
            if self.posname is not None:
                Pos = self.posname
        if pth is None:
            if self.pth is not None:
                pth = self.pth
        p = PosLbl._load(Pos=Pos, pth=pth, fname=fname)
        self.__dict__.update(p.__dict__)
        self.__class__ = PosLbl

    @classmethod
    def _load(self, Pos=None, pth=None, fname="PosLbls"):
        """
        load poslbl
        """
        import os

        foldername = os.path.join(pth, fname + os.path.sep)
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        fname = os.path.join(foldername, Pos + ".pkl")
        with open(fname, "rb") as dbfile:
            p = dill.load(dbfile)
            sys.stdout.write(
                "\r" + "Loaded position " + str(Pos) + " from pickle file."
            )
            sys.stdout.flush()
        # Rebind to the current class so methods always reflect live code,
        # not the version that was baked in by cloudpickle at save time.
        p.__class__ = PosLbl
        return p

    @property
    def PixelSize(self):
        return self.framelabels[0]._pixelsize

    @property
    def imagedims(self):
        return self.framelabels[0].imagedims

    @property
    def XY(self):
        return self.framelabels[0].XY

    @property
    def num(self):
        return np.array([r.num for r in self.framelabels], dtype=object)

    @property
    def centroid(self):
        return np.array([r.centroid for r in self.framelabels], dtype=object)

    @property
    def weighted_centroid(self):
        return np.array([r.weighted_centroid for r in self.framelabels], dtype=object)

    @property
    def area(self):
        return np.array([r.area for r in self.framelabels], dtype=object)

    @property
    def index(self):
        return np.array([r.index for r in self.framelabels], dtype=object)

    @property
    def centroid_um(self):
        return np.array([r.centroid_um for r in self.framelabels], dtype=object)

    @property
    def weighted_centroid_um(self):
        return np.array(
            [r.weighted_centroid_um for r in self.framelabels], dtype=object
        )

    @property
    def area_um2(self):
        return np.array([r.area_um2 for r in self.framelabels])

    def mean(self, ch, periring=False):
        return np.array(
            [r.mean(ch, periring=periring) for r in self.framelabels], dtype=object
        )

    def median(self, ch, periring=False):
        return np.array(
            [r.median(ch, periring=periring) for r in self.framelabels], dtype=object
        )

    def minint(self, ch, periring=False):
        return np.array(
            [r.minint(ch, periring=periring) for r in self.framelabels], dtype=object
        )

    def maxint(self, ch, periring=False):
        return np.array(
            [r.maxint(ch, periring=periring) for r in self.framelabels], dtype=object
        )

    def ninetyint(self, ch, periring=False):
        return np.array(
            [r.ninetyint(ch, periring=periring) for r in self.framelabels], dtype=object
        )

    @property
    def density(self):
        return np.array([r.density for r in self.framelabels])

    @property
    def prop_list(self):
        return [
            f
            for f in list(self.framelabels[0].regionprops)
            if not f.startswith(("mean", "median", "max", "min", "90th", "slice"))
        ]

    @property
    def numtracks(self):
        try:
            return self.trackinds.shape[0]
        except ValueError:
            print("Position not tracked")

    def get_track(self, i=0):
        """
        function to return specific track. Best use as a first class citizen.
        Parameters
        ----------
        i : ind track index

        Returns
        -------
        _onetrack object
        """
        assert i <= len(self.trackinds), "track index must be < %i" % len(
            self.trackinds
        )
        return self._onetrack(self, i=i)

    # class for a single track. return everything we care about
    class _onetrack(object):
        """
        Class that manages a single track.

        Class Properties
        ----------------
         'T',
         'area',
         'area_um2',
         'centroid',
         'centroid_um',
         'maxint',
         'mean',
         'median',
         'minint',
         'ninetyint',
         'numtracks',
         'trackinds',
         'weighted_centroid',
         'weighted_centroid_um'

        Class Methods
        -------------
         'show_movie'

        """

        def __init__(self, outer, i=0):
            self.trackinds = outer.trackinds[i]
            if outer._splitflag:
                self.relatives = outer.relatives[i]
            self.numtracks = len(outer.trackinds)
            self._outer = outer

        @property
        def T(self):
            return np.nonzero(~np.isnan(self.trackinds.astype("float")))[0]

        @property
        def centroid(self):
            return self.prop("centroid")

        @property
        def weighted_centroid(self):
            return self.prop("weighted_centroid")

        @property
        def centroid_um(self):
            return self._outer.XY + self._outer.PixelSize * self.centroid

        @property
        def weighted_centroid_um(self):
            return self._outer.XY + self._outer.PixelSize * self.weighted_centroid

        @property
        def area(self):
            return self.prop("area")

        @property
        def area_um2(self):
            return self.prop("area") * (self._outer.PixelSize**2)

        def mean(self, ch, periring=False):
            peritext = ""
            if periring:
                peritext = "_periring"
            return self.prop("mean_" + ch + peritext)

        def median(self, ch, periring=False):
            peritext = ""
            if periring:
                peritext = "_periring"
            return self.prop("median_" + ch + peritext)

        def minint(self, ch, periring=False):
            peritext = ""
            if periring:
                peritext = "_periring"
            return self.prop("min_" + ch + peritext)

        def maxint(self, ch, periring=False):
            peritext = ""
            if periring:
                peritext = "_periring"
            return self.prop("max_" + ch + peritext)

        def ninetyint(self, ch, periring=False):
            peritext = ""
            if periring:
                peritext = "_periring"
            return self.prop("90th_" + ch + peritext)

        def prop(self, prop="area"):
            return np.array(
                [
                    self._outer.framelabels[j].regionprops[prop][self.trackinds[j]]
                    for j in self.T
                ]
            )

        @property
        def frame(self):
            return np.array([int(self._outer.framelabels[j].frame) for j in self.T])

        def get_movie(self, Channel=["DeepBlue"], boxsize=75, frame=None, **kwargs):
            """
            Function to display a close up movie of a cell being tracked.
            Parameters
            ----------
            Channel : ['DeepBlue'] str or list of strs
            boxsize : [50] num size of box around the cell
            cmaps : order of colormaps for each channel
            """
            if frame is None:
                frame = self.frame
            else:
                if not isinstance(frame, (list, np.ndarray)):
                    frame = np.array([frame])

            f_ind = [np.where(self.frame == f)[0][0] for f in frame]
            cents = np.fliplr(self.centroid[f_ind])
            crp = list(
                map(
                    tuple,
                    np.ceil(
                        np.concatenate((cents - boxsize, cents + boxsize), axis=1)
                    ).astype(int),
                )
            )

            stk = self._outer.img(
                Channel, frame=frame, crop=crp, verbose=False, groupby="Channel"
            )
            return stk

        def show_movie(
            self,
            Channel=["DeepBlue"],
            boxsize=75,
            cmaps=["red", "green", "blue", "cyan", "magenta", "yellow"],
            **kwargs,
        ):
            """
            Function to display a close up movie of a cell being tracked.
            Parameters
            ----------
            Channel : ['DeepBlue'] str or list of strs
            boxsize : [50] num size of box around the cell
            cmaps : order of colormaps for each channel
            """
            if type(Channel) == str:
                cmaps = ["gray"]

            from oyLabImaging.Processing.imvisutils import get_or_create_viewer

            viewer = get_or_create_viewer()
            viewer.scale_bar.unit = "um"
            cents = np.fliplr(self.centroid)
            # cents = self.centroid

            crp = list(
                map(
                    tuple,
                    np.ceil(
                        np.concatenate((cents - boxsize, cents + boxsize), axis=1)
                    ).astype(int),
                )
            )

            for ind, ch in enumerate(Channel):
                # imgs = self._outer.img(ch, frame=list(self.T),verbose=False)
                # stk = np.array([np.pad(im1, boxsize)[crp1[0]+boxsize:crp1[2]+boxsize, crp1[1]+boxsize:crp1[3]+boxsize] for im1, crp1 in zip(imgs, crp)])

                stk = self._outer.img(
                    ch, frame=list(self.frame), crop=crp, verbose=False
                )
                stksmp = stk.flatten()  # sample_stack(stk,int(stk.size/100))
                stksmp = stksmp[stksmp != 0]
                viewer.add_image(
                    stk,
                    blending="additive",
                    contrast_limits=[
                        np.percentile(stksmp, 1),
                        np.percentile(stksmp, 99.9),
                    ],
                    name=ch,
                    colormap=cmaps[ind % len(cmaps)],
                    scale=[self._outer.PixelSize, self._outer.PixelSize],
                )

    def trackcells(self, split=True, **kwargs):
        """
        all tracking is done using the Jonker Volgenant lap algorithm:
        R. Jonker and A. Volgenant, "A Shortest Augmenting Path Algorithm for Dense and Sparse Linear Assignment Problems", Computing 38, 325-340 (1987)

        TODO: try Viterbi algo for tracking
        TODO: add mitosis detector: https://academic.oup.com/bioinformatics/article/35/15/2644/5259190

        Parameters
        ----------
        kwargs that go into tracking helper functions: search_radius, params (list of tuples, (channel, weight)), maxStep for skip ,maxAmpRatio for skip, mintracklength


        """
        self._link(**kwargs)
        self._closegaps(**kwargs)
        self._tracked = True
        if split:
            self._split(**kwargs)
            self._splitflag = True
        else:
            self._splitflag = False
        self._calculate_trackmat()
        self.track_to_use = []
        self.save()

    def _link(self, search_radius=15, params=[], **kwargs):
        """
        Helper function : link adjecent frames using JV lap.

        TODO: extend for a general cost function. make class of cost functions that returns shape, cc, ii, jj
        Parameters
        ----------
        params : [(channel,weight)] list of tuples
        search_radius : [25]
        """

        cents = self.centroid_um
        nums = self.num
        if "adaptive_radius" in kwargs:
            adaptive = True
            sr_factor = np.sqrt(self.num / self.num[0])
        else:
            adaptive = False

        if params:
            Cp = [p[0] for p in params if p[0] in self.channels]
            Wp = [p[1] for p in params if p[0] in self.channels]
            ints = {}
        else:
            Cp = []
            Wp = []

        for i in np.arange(nums.shape[0] - 1):
            sys.stdout.write("\r" + "linking frame " + str(i))
            sys.stdout.flush()
            if nums[i + 1] > 0 and nums[i] > 0:
                T = KDTree(cents[i + 1])
                # We calculate points in centroid(n+1) that are less than distance_upper_bound from points in centroid(n)
                if adaptive:
                    sr = search_radius / np.sqrt(self.num[i] / self.num[0])
                else:
                    sr = search_radius

                dists, idx = T.query(cents[i], k=12, distance_upper_bound=sr)

                dists = [r[r < 1e308] for r in dists]

                idx = [r[r < cents[i + 1].shape[0]] for r in idx]

                # possible matches in n+1
                jj = np.concatenate(idx)

                # possible correspondence in n
                j = 0
                ii = []
                for r in idx:
                    ii.append(j * np.ones_like(r))
                    j += 1
                ii = np.concatenate(ii)

                # ii jj cc are now sparse matrix in COO format
                ampRatio = 1
                eps = 10**-72
                for cp, wp in zip(Cp, Wp):
                    ints = self.ninetyint(cp)
                    ampRatio = ampRatio + wp * (
                        np.array(
                            eps + np.maximum(list(ints[i][ii]), list(ints[i + 1][jj]))
                        )
                        / np.array(
                            eps + np.minimum(list(ints[i][ii]), list(ints[i + 1][jj]))
                        )
                        - 1
                    )

                # costs of match
                cc = np.concatenate(dists) * ampRatio
                cc[cc > 1000000] = 999999

                shape = (nums[i], nums[i + 1])

                cc, ii, kk = prepare_sparse_cost(shape, cc, ii, jj, cost_limit=300)
                ind1, ind0 = lap.lapmod(len(ii) - 1, cc, ii, kk, return_cost=False)
                ind1[ind1 >= shape[1]] = -1
                ind0[ind0 >= shape[0]] = -1
                # inds in n+1 that match inds (1:N) in n
                ind1 = ind1[: shape[0]]
                # inds in n that match inds (1:N) in n+1
                ind0 = ind0[: shape[1]]
                self.framelabels[i].link1in2 = ind1
            else:
                self.framelabels[i].link1in2 = np.array([])
        self.framelabels[nums.shape[0] - 1].link1in2 = np.array([])

    def _getTrackLinks(self, i=0, l=0, **kwargs):
        """
        Helper function recursive function that gets an initial frame i and starting cell label
        l and returns all labels
        """

        if i + 1 < len(self.frames):
            if self.num[i + 1] > 0:
                if l > -1:
                    return np.append(
                        l, self._getTrackLinks(i + 1, self.framelabels[i].link1in2[l])
                    )
                else:
                    pass
            else:
                return np.array([])
        elif i + 1 == len(self.frames):
            if l > -1:
                return l
            else:
                pass

    def _getAllContinuousTrackSegs(self, minseglength=4, **kwargs):
        """
        Helper function that returns the frame indexing of all contiuous tracks (chained links) longer than minseglength.

        Assumes links have been calculated

        Parameters
        ----------
        minseglength : [5] smallest stub that doesnt get discarded
        """
        trackbits = []
        for i in np.arange(len(self.frames) - 1):
            for l in np.nonzero(
                np.isin(
                    np.arange(self.num[i]),
                    np.array([r[i] for r in trackbits]),
                    invert=True,
                )
            )[0]:
                trkl = self._getTrackLinks(i=i, l=l)
                trackbits.append(
                    np.pad(
                        trkl.astype("object"),
                        (i, len(self.frames) - trkl.size - i),
                        "constant",
                        constant_values=(None, None),
                    )
                )
        trackbits = np.array(trackbits)

        # return tracks segments that have more than minseglength frames
        return trackbits[
            (np.array([np.sum(r != None) for r in trackbits]) >= minseglength)
        ]

    def _closegaps(
        self,
        maxStep=10,
        params=[],
        maxAmpRatio=5,
        maxTimeJump=4,
        mintracklength=30,
        **kwargs,
    ):
        """
        Helper function : close gaps between open stubs using JV lap.

        todo: split. When a stub that starts in the middle has a plausible link, make a compound track
        ----------
        #NucChannel : ['DeepBlue']
        params : [(channel,weight)] list of tuples
        maxAmpRatio : [2] max allowd ratio of amplitudes for linking
        mintracklength : [30] final minimum length of a track
        """

        if params:
            Cp = [p[0] for p in params if p[0] in self.channels]
            Wp = [p[1] for p in params if p[0] in self.channels]
            ints = {}
            for cp in Cp:
                ints[cp] = self.ninetyint(cp)
        else:
            Cp = []
            Wp = []

        trackbits = self._getAllContinuousTrackSegs(**kwargs)

        cents = self.centroid_um
        notdoneflag = 1

        while notdoneflag:
            trackstarts = np.array(
                [np.where(~np.isnan(r.astype("float")))[0][0] for r in trackbits],
                dtype=object,
            )
            trackends = np.array(
                [np.where(~np.isnan(r.astype("float")))[0][-1] for r in trackbits],
                dtype=object,
            )

            dtmat = np.expand_dims(trackstarts, 1) - np.expand_dims(trackends, 0)
            possiblelinks = np.transpose(
                np.nonzero((dtmat > 0) * (dtmat < maxTimeJump))
            )
            ii = []
            jj = []
            cc = []
            for i in np.arange(len(possiblelinks)):
                # frame1 - end of possible to link
                frame1 = trackends[possiblelinks[i][1]]
                # frame2 - beginning of possible link
                frame2 = trackstarts[possiblelinks[i][0]]
                # cell label in frame 1 to link
                ind1 = trackbits[possiblelinks[i][1]][frame1]
                # cell label in frame 2 to link
                ind2 = trackbits[possiblelinks[i][0]][frame2]
                dt = frame2 - frame1
                dr = np.linalg.norm(cents[frame1][ind1] - cents[frame2][ind2])

                da = 1
                eps = 10**-72
                for cp, wp in zip(Cp, Wp):
                    da = da + wp * (
                        (
                            eps
                            + np.maximum(ints[cp][frame1][ind1], ints[cp][frame2][ind2])
                        )
                        / (
                            eps
                            + np.minimum(ints[cp][frame1][ind1], ints[cp][frame2][ind2])
                        )
                        - 1
                    )

                if dr <= (np.sqrt(dt) * maxStep):
                    if da <= maxAmpRatio:
                        ii.append(possiblelinks[i][1])
                        jj.append(possiblelinks[i][0])

                        # maybe one day we'll change this somehow. Not sure how rn
                        cost = dr * da * dt

                        cc.append(cost)
            ii = np.array(ii)
            jj = np.array(jj)
            cc = np.array(cc)

            if len(ii) == 0:
                print("\nFinished connecting tracks")
                notdoneflag = 0
                break

            shape = (len(trackbits), len(trackbits))
            cc, ii, kk = prepare_sparse_cost(shape, cc, ii, jj, 1000)
            match1, _ = lap.lapmod(len(ii) - 1, cc, ii, kk, return_cost=False)
            match1[match1 >= shape[1]] = -1
            # inds in n+1 that match inds (1:N) in n
            match1 = np.array(match1[: shape[0]])

            trackindstofill = np.nonzero(match1 + 1)[0]
            trackindstoadd = match1[np.nonzero(match1 + 1)]

            fa = {
                trackindstofill[i]: trackindstoadd[i]
                for i in range(len(trackindstoadd))
            }

            for i in fa:
                sf = trackstarts[fa[i]]
                ef = trackends[fa[i]] + 1
                trackbits[i][np.arange(sf, ef)] = trackbits[fa[i]][np.arange(sf, ef)]
                trackbits[fa[i]][np.arange(sf, ef)] = None

                # add nans in gaps
                # eef = trackends[i]+1
                # trackbits[i][np.arange(eef,sf)]=np.nan

            # remove lines that are all Nones
            trackbits = trackbits[
                [any(~np.isnan(r.astype("float"))) for r in trackbits]
            ]

        trackbits = trackbits[
            np.array([sum(~np.isnan(r.astype("float"))) for r in trackbits])
            >= mintracklength
        ]

        sortind = np.lexsort(
            (
                np.arange(len(trackbits)),
                [np.sum(np.isnan(r.astype("float"))) for r in trackbits],
                [np.sum(r == None) for r in trackbits],
            )
        )
        self.trackinds = trackbits[sortind]

    def _split(self, search_radius=20, params=[], maxAmpRatio=5, **kwargs):
        """
        Helper function : find splits using JV lap.

        ----------
        params : [(channel,weight)] list of tuples
        maxAmpRatio : [5] max allowd ratio of amplitudes for linking
        search_radius : [40] search radius
        """
        # ch=NucChannel
        # if ch is None:
        #    ch = self.channels[0]

        if params:
            Cp = [p[0] for p in params if p[0] in self.channels]
            Wp = [p[1] for p in params if p[0] in self.channels]
            ints = {}
            for cp in Cp:
                ints[cp] = self.ninetyint(cp)
        else:
            Cp = []
            Wp = []

        trackbits = self.trackinds

        cents = self.centroid_um
        relatives = [[]] * len(trackbits)
        notdoneflag = 1

        while notdoneflag:
            trackstarts = np.array(
                [np.where(~np.isnan(r.astype("float")))[0][0] for r in trackbits],
                dtype=object,
            )
            trackends = np.array(
                [np.where(~np.isnan(r.astype("float")))[0] for r in trackbits],
                dtype=object,
            )

            possiblelinks = np.empty((0, 2), int)
            for J in np.unique(trackstarts):
                link_a = np.where(trackstarts == J)
                link_b = np.where(list(map(lambda x: np.any(x == J - 1), trackends)))
                if np.any(link_a):
                    f = lambda x: np.pad(link_b, ((1, 0), (0, 0)), constant_values=x)
                    d = np.transpose(np.hstack(list(map(f, link_a[0]))))
                    possiblelinks = np.concatenate((possiblelinks, d))

            ii = []
            jj = []
            cc = []
            for i in np.arange(len(possiblelinks)):
                # frame1 - end of possible to link
                frame2 = trackstarts[possiblelinks[i][0]]
                # frame1 - end of possible to link
                frame1 = frame2 - 1
                # cell label in frame 1 to link
                ind1 = trackbits[possiblelinks[i][1]][frame1]
                # cell label in frame 2 to link
                ind2 = trackbits[possiblelinks[i][0]][frame2]
                dr = np.linalg.norm(cents[frame1][int(ind1)] - cents[frame2][int(ind2)])

                if dr <= search_radius:
                    da = 1
                    for cp, wp in zip(Cp, Wp):
                        da = da + wp * (
                            np.maximum(ints[cp][frame1][ind1], ints[cp][frame2][ind2])
                            / np.minimum(ints[cp][frame1][ind1], ints[cp][frame2][ind2])
                            - 1
                        )
                    if da <= maxAmpRatio:
                        ii.append(possiblelinks[i][1])
                        jj.append(possiblelinks[i][0])

                        # maybe one day we'll change this somehow. Not sure how rn
                        cost = dr * da

                        cc.append(cost)
            ii = np.array(ii)
            jj = np.array(jj)
            cc = np.array(cc)

            # print(ii)
            if len(ii) == 0:
                print("\nFinished finding splits")
                notdoneflag = 0
                break

            shape = (len(trackbits), len(trackbits))
            cc, ii, kk = prepare_sparse_cost(shape, cc, ii, jj, 1000)
            match1, _ = lap.lapmod(len(ii) - 1, cc, ii, kk, return_cost=False)
            match1[match1 >= shape[1]] = -1
            # inds in n+1 that match inds (1:N) in n
            match1 = np.array(match1[: shape[0]])

            trackindstofill = np.nonzero(match1 + 1)[0]
            trackindstoadd = match1[np.nonzero(match1 + 1)]

            fa = {
                trackindstofill[i]: trackindstoadd[i]
                for i in range(len(trackindstoadd))
            }

            for i in fa:
                sf = trackstarts[i]
                ef = trackstarts[fa[i]]
                trackbits[fa[i]][np.arange(sf, ef)] = trackbits[i][np.arange(sf, ef)]
                relatives[i] = relatives[i] + [fa[i]]
                relatives[fa[i]] = relatives[fa[i]] + [i]

        self.relatives = relatives
        self.trackinds = trackbits

    def img(self, Channel=None, Zindex=None, ffield=None, **kwargs):
        """
        Parameters
        ------
        Channel : [DeepBlue] str or list of strings
        register : {[True], False}
        Zindex=[0]
        ffield : {[None], True, False} — defaults to the ffield flag the
                 position was segmented with (self._ffieldflag)

        Returns
        -------
        Image stack of given channels
        """
        from oyLabImaging import Metadata

        if Channel is None:
            Channel = self.channels[0]
            print("loading " + Channel)

        if ffield is None:
            ffield = self._ffieldflag

        pth = self.pth
        MD = Metadata(pth, verbose=False)
        if Zindex is None:
            Zindex = MD.Zindexes[0]

        return MD.stkread(
            Channel=Channel,
            Position=self.posname,
            register=self._registerflag,
            ffield=ffield,
            Zindex=Zindex,
            **kwargs,
        )

    def _calculate_pointmat(self):
        """
        helper function, calculate points in a napari-friendly way
        """
        a = []
        [
            a.append((np.pad(cen, ((0, 0), (1, 0)), constant_values=i)))
            for i, cen in enumerate(self.centroid)
            if np.any(cen)
        ]
        try:
            self._pointmatrix = np.concatenate(a)
        except:
            self._pointmatrix = []

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

    def _pointmat(self, frames=None):
        """
        helper function, return points in a napari-friendly way for given frames
        Parameters
        ----------
        frames
        """
        if frames is None:
            frames = self.frames
        else:
            if not isinstance(frames, (list, np.ndarray)):
                frames = [frames]
        frames = [j for j in frames if j in self.frames]
        return self._pointmatrix[np.in1d(self._pointmatrix[:, 0], frames)]

    def _calculate_trackmat(self):
        """
        helper function, calculate tracks in a napari-friendly way
        """
        if self.numtracks:
            t0 = self.get_track
            J = np.arange(self.numtracks)
            self._trackmatrix = np.concatenate(
                [
                    [np.append([i, ind], x) for ind, x in zip(t0(i).T, t0(i).centroid)]
                    for i in tqdm(J)
                ]
            )
        else:
            self._trackmatrix = np.array([])

    def _tracksmat(self, J=None):
        """
        helper function, return tracks in a napari-friendly way for given tracks
        Parameters
        ----------
        J - track indices to return
        """
        if self.numtracks:
            t0 = self.get_track
            if J is None:
                J = np.arange(self.numtracks)
            else:
                if not (isinstance(J, list) or isinstance(J, np.ndarray)):
                    J = [J]
                J = [j for j in J if j in np.arange(t0().numtracks)]
            return self._trackmatrix[np.in1d(self._trackmatrix[:, 0], J)], J
        else:
            return np.array([]), []

    def radial_corr(self, ch_i, ch_j=None, frame=None,
                    img=False, ffield=True,
                    max_r=None, dr=None,
                    intensity='mean', periring=False, n_max=200_000, seed=42,
                    recompute=False):
        """Radial cross/auto-correlation g(r).

        img=False (default): cell-level Pearson correlation binned by distance.
        img=True: pixel-level FFT correlation (Moisan 2011 boundary conditions).

        ffield checks whether FrameLbls were segmented with flat-field
        correction (img=False) or applies correction when loading images (img=True).

        Returns
        -------
        dict with keys r, g, sem, n, ch_i, ch_j, max_r, dr, img
        """
        _chj = ch_j or ch_i
        key = f"radial_corr|{ch_i}|{_chj}|{_frame_key(frame)}|img={img}|max_r={max_r}|dr={dr}|{intensity}|periring={periring}|ffield={ffield}"
        if not recompute and key in self.spatial:
            return self.spatial[key].data
        from oyLabImaging.Processing.spatial import radial_corr
        result = radial_corr(self, ch_i=ch_i, ch_j=ch_j, frame=frame,
                             img=img, ffield=ffield, max_r=max_r, dr=dr,
                             intensity=intensity, periring=periring,
                             n_max=n_max, seed=seed)
        _kw = _hint_kw(ch_j=(ch_j, None), frame=(frame, None), img=(img, False),
                       max_r=(max_r, None), dr=(dr, None), ffield=(ffield, True),
                       intensity=(intensity, 'mean'), periring=(periring, False))
        self._cache_spatial(
            key, result,
            f".radial_corr('{ch_i}'{_kw})",
            f".plot_radial_corr('{ch_i}'{_kw})",
        )
        return result

    def plot_radial_corr(self, ch_i, ch_j=None, frame=None,
                         img=False, ffield=True,
                         max_r=None, dr=None,
                         intensity='mean', periring=False,
                         recompute=False, ax=None, **plot_kwargs):
        """Compute and plot radial correlation in one call."""
        from oyLabImaging.Processing.spatial import plot_radial
        result = self.radial_corr(ch_i=ch_i, ch_j=ch_j, frame=frame,
                                  img=img, ffield=ffield, max_r=max_r, dr=dr,
                                  intensity=intensity, periring=periring,
                                  recompute=recompute)
        return plot_radial(result, ax=ax, **plot_kwargs)

    def radial_density(self, frame=None, max_r=200.0, dr=5.0, recompute=False):
        """Pair correlation function g(r) for cell positions.

        g(r) = 1 for a random distribution; >1 clustering; <1 repulsion.

        Returns
        -------
        dict with keys r, g, n, max_r, dr
        """
        key = f"radial_density|{_frame_key(frame)}|max_r={max_r}|dr={dr}"
        if not recompute and key in self.spatial:
            return self.spatial[key].data
        from oyLabImaging.Processing.spatial import radial_density
        result = radial_density(self, frame=frame, max_r=max_r, dr=dr)
        _kw = _hint_kw(frame=(frame, None), max_r=(max_r, 200.0), dr=(dr, 5.0))
        self._cache_spatial(
            key, result,
            f".radial_density({_kw.lstrip(', ')})",
            f".plot_radial_density({_kw.lstrip(', ')})",
        )
        return result

    def plot_radial_density(self, frame=None, max_r=200.0, dr=5.0,
                            recompute=False, ax=None, **plot_kwargs):
        """Compute and plot pair correlation function in one call."""
        from oyLabImaging.Processing.spatial import plot_radial
        result = self.radial_density(frame=frame, max_r=max_r, dr=dr,
                                     recompute=recompute)
        return plot_radial(result, ax=ax, **plot_kwargs)

    def local_moran_I(self, ch, frame=None, radius=50.0, n_permutations=999,
                      intensity='mean', periring=False, seed=42, ffield=True,
                      recompute=False):
        """Local Moran's I (LISA) for detecting collective activity hotspots.

        ffield=True (default) warns if FrameLbls were segmented without
        flat-field correction, since illumination bias affects I scores.

        Returns a dict (single frame) or list of dicts (multiple frames) with
        per-cell I scores, p-values, edge flags, and standardized intensities.
        """
        key = f"local_moran_I|{ch}|{_frame_key(frame)}|r={radius}|np={n_permutations}|{intensity}|periring={periring}|ffield={ffield}"
        if not recompute and key in self.spatial:
            return self.spatial[key].data
        from oyLabImaging.Processing.spatial import local_moran_I
        result = local_moran_I(self, ch=ch, frame=frame, radius=radius,
                               n_permutations=n_permutations, intensity=intensity,
                               periring=periring, seed=seed, ffield=ffield)
        _kw = _hint_kw(frame=(frame, None), radius=(radius, 50.0),
                       n_permutations=(n_permutations, 999), ffield=(ffield, True),
                       intensity=(intensity, 'mean'), periring=(periring, False))
        self._cache_spatial(
            key, result,
            f".local_moran_I('{ch}'{_kw})",
            f".plot_lisa('{ch}'{_kw})",
        )
        return result

    def find_activity_clusters(self, lisa_result, min_I=0.5, max_pvalue=0.05,
                               min_cells=10, eps=None):
        """DBSCAN clustering on LISA-significant cells.

        Pass the output of local_moran_I (single frame) plus threshold params.
        """
        from oyLabImaging.Processing.spatial import find_activity_clusters
        return find_activity_clusters(lisa_result, min_I=min_I,
                                      max_pvalue=max_pvalue,
                                      min_cells=min_cells, eps=eps)

    def activity_clusters(self, ch, frame=None, radius=50.0, n_permutations=999,
                          intensity='mean', periring=False, seed=42, ffield=True,
                          min_I=0.5, max_pvalue=0.05, min_cells=3, eps=None):
        """Compute LISA then find activity clusters — no plotting.

        Returns
        -------
        list of dicts (one per frame) with keys:
            frame_index, n_clusters, cluster_sizes, coords, labels, I, pvalue
        """
        from oyLabImaging.Processing.spatial import local_moran_I, find_activity_clusters

        lisa_list = local_moran_I(self, ch=ch, frame=frame, radius=radius,
                                  n_permutations=n_permutations,
                                  intensity=intensity, periring=periring,
                                  seed=seed, ffield=ffield)
        if isinstance(lisa_list, dict):
            lisa_list = [lisa_list]

        results = []
        for lisa_res in lisa_list:
            cl = find_activity_clusters(lisa_res, min_I=min_I,
                                        max_pvalue=max_pvalue,
                                        min_cells=min_cells, eps=eps)
            cl['frame_index'] = lisa_res['frame_index']
            results.append(cl)
        return results

    def plot_lisa(self, ch, frame=None, radius=50.0, n_permutations=999,
                  intensity='mean', periring=False, seed=42, ffield=True,
                  min_I=0.5, max_pvalue=0.005, min_cells=10,
                  show_clusters=True, overlay=True, size=8,
                  colormap='coolwarm', vmax=None, recompute=False):
        """Compute LISA and visualise in napari.

        Parameters
        ----------
        ch : str
        frame : int, list of int, or None
        radius : float  neighborhood radius in µm
        overlay : bool  add raw channel image behind LISA points (default True)
        show_clusters : bool  add convex-hull outlines for significant clusters
        min_I, max_pvalue, min_cells : cluster thresholds
        colormap : diverging colormap for Moran's I
        vmax : color scale maximum (default: 99th percentile of |I|)

        Returns
        -------
        layer : napari Points layer
        cluster_results : list of dicts  (only when show_clusters=True)
        """
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        from scipy.spatial import ConvexHull
        from oyLabImaging.Processing.spatial import find_activity_clusters
        from oyLabImaging.Processing.imvisutils import get_or_create_viewer

        # ── LISA computation (cached) ─────────────────────────────────────────
        lisa_list = self.local_moran_I(ch=ch, frame=frame, radius=radius,
                                       n_permutations=n_permutations,
                                       intensity=intensity, periring=periring,
                                       seed=seed, ffield=ffield,
                                       recompute=recompute)
        if isinstance(lisa_list, dict):
            lisa_list = [lisa_list]

        frames = [r['frame_index'] for r in lisa_list]

        ps = float(self.PixelSize)
        viewer = get_or_create_viewer()

        #── overlay image ─────────────────────────────────────────────────────
        if overlay:
            for t in frames:
                # load one frame at a time — always results in a 2-D (H, W) array
                raw = self.img(Channel=ch, verbose=False, frames=[self.frames[t]])
                raw = np.squeeze(raw)
                # collapse accidental colour dim (e.g. grayscale stored as RGB)
                if raw.ndim == 3:
                    raw = raw.mean(axis=-1)
                lo, hi = np.percentile(raw, [1, 99.9])
                viewer.add_image(
                    raw,
                    blending='additive',
                    contrast_limits=[lo, hi],
                    scale=[ps, ps],
                    colormap='gray',
                    name=f'{ch} t={t}',
                )

        # ── collect points ────────────────────────────────────────────────────
        all_pts, all_I, all_pv = [], [], []

        for lisa_res in lisa_list:
            t = lisa_res['frame_index']
            fl = self.framelabels[t]
            cen = np.asarray(fl.centroid, dtype=np.float64)   # (N, 2) row/col
            if len(cen) == 0:
                continue
            I        = lisa_res['I']
            pvalue   = lisa_res['pvalue']
            is_edge  = lisa_res.get('is_edge', np.zeros(len(I), dtype=bool))
            interior = ~is_edge & ~np.isnan(I)
            all_pts.append(cen[interior])
            all_I.append(I[interior])
            all_pv.append(pvalue[interior])

        if not all_pts:
            raise ValueError("No valid interior cells found.")

        pts_int   = np.concatenate(all_pts)           # (M, 2) row/col pixels
        moran_int = np.concatenate(all_I).astype(np.float64)
        pval_int  = np.concatenate(all_pv).astype(np.float64)

        # ── colour map ────────────────────────────────────────────────────────
        vmax_use = vmax if vmax is not None else float(
            np.nanpercentile(np.abs(moran_int), 99))
        norm     = mcolors.TwoSlopeNorm(vmin=-vmax_use, vcenter=0.0, vmax=vmax_use)
        rgba     = cm.get_cmap(colormap)(norm(moran_int))
        rgba[pval_int > max_pvalue, 3] = 0.2

        # ── LISA points layer ─────────────────────────────────────────────────
        layer = viewer.add_points(
            pts_int,
            face_color=rgba,
            edge_width=0,
            size=size,
            blending='translucent',
            scale=[ps, ps],
            name=f'LISA {ch}',
            properties={'moran_I': moran_int, 'pvalue': pval_int},
        )

        # ── cluster outlines ──────────────────────────────────────────────────
        cluster_results = []
        if show_clusters:
            hulls, centroid_pts, centroid_ids = [], [], []

            for lisa_res in lisa_list:
                t = lisa_res['frame_index']
                cl = find_activity_clusters(lisa_res, min_I=min_I,
                                            max_pvalue=max_pvalue,
                                            min_cells=min_cells)
                cl['frame_index'] = t
                cluster_results.append(cl)
                if cl['n_clusters'] == 0:
                    continue

                fl  = self.framelabels[t]
                xy  = np.asarray(fl.XY, dtype=np.float64)

                for k in range(cl['n_clusters']):
                    pts_um = cl['coords'][cl['labels'] == k]   # (M, 2) µm
                    pts_px = (pts_um - xy) / ps                # (M, 2) pixels

                    if len(pts_px) >= 3:
                        try:
                            verts = pts_px[ConvexHull(pts_px).vertices]
                        except Exception:
                            verts = pts_px
                    else:
                        verts = pts_px

                    if len(verts) < 2:
                        continue
                    hulls.append(verts)
                    centroid_pts.append(pts_px.mean(axis=0))
                    centroid_ids.append(k + 1)

            if hulls:
                viewer.add_shapes(
                    hulls,
                    shape_type='polygon',
                    face_color=[1, 1, 1, 0.0],
                    edge_color='#ff1493',
                    edge_width=10,
                    scale=[ps, ps],
                    name=f'LISA {ch} clusters',
                )
                viewer.add_points(
                    np.array(centroid_pts),
                    properties={'n': np.array(centroid_ids, dtype=int)},
                    text='n',
                    face_color=[0, 0, 0, 0],
                    edge_width=0,
                    size=size * 1.5,
                    blending='translucent',
                    scale=[ps, ps],
                    name=f'LISA {ch} cluster labels',
                )

        if show_clusters:
            return layer, cluster_results
        return layer

    def gistar(self, ch, frame=None, radius=50.0, intensity='mean',
               periring=False, seed=42, ffield=True, recompute=False):
        """Getis-Ord Gi* hot-spot statistic. See spatial.gistar for details."""
        key = f"gistar|{ch}|{_frame_key(frame)}|r={radius}|{intensity}|periring={periring}|ffield={ffield}"
        if not recompute and key in self.spatial:
            return self.spatial[key].data
        from oyLabImaging.Processing.spatial import gistar
        result = gistar(self, ch=ch, frame=frame, radius=radius,
                        intensity=intensity, periring=periring,
                        seed=seed, ffield=ffield)
        _kw = _hint_kw(frame=(frame, None), radius=(radius, 50.0), ffield=(ffield, True),
                       intensity=(intensity, 'mean'), periring=(periring, False))
        self._cache_spatial(
            key, result,
            f".gistar('{ch}'{_kw})",
            f".plot_gistar('{ch}'{_kw})",
        )
        return result

    def plot_gistar(self, ch, frame=None, radius=50.0, intensity='mean',
                    periring=False, seed=42, ffield=True,
                    max_pvalue=0.05, min_z=1.96, min_cells=10,
                    show_clusters=True, vmax=None, colormap='coolwarm',
                    overlay=True, size=8, viewer=None, recompute=False):
        """Compute Gi* and visualise hot/cold spots in napari.

        Parameters
        ----------
        ch : str
        frame : int, list of int, or None
        radius : float   Neighbourhood radius in µm.
        max_pvalue : float
            p-value threshold: cells above this are rendered at 20% opacity
            and excluded from cluster detection.
        min_z : float
            Minimum |z-score| for a cell to be included in cluster detection
            (default 1.96, equivalent to p < 0.05 under normality).
        min_cells : int
            Minimum number of cells to form a cluster (default 10).
        show_clusters : bool
            Draw convex-hull outlines around hot and cold spot clusters.
        vmax : float
            Colour-scale maximum (default: 99th percentile of |z|).
        colormap : str
            Diverging colormap. Blue = cold spots, red = hot spots.
        overlay : bool   Add raw channel image behind the points.
        size : int       Point size in pixels.

        Returns
        -------
        layer : napari Points layer
        cluster_results : list of dicts (only when show_clusters=True)
        """
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        from scipy.spatial import ConvexHull
        from sklearn.cluster import DBSCAN
        from oyLabImaging.Processing.imvisutils import get_or_create_viewer

        results = self.gistar(ch=ch, frame=frame, radius=radius,
                              intensity=intensity, periring=periring,
                              seed=seed, ffield=ffield, recompute=recompute)
        if isinstance(results, dict):
            results = [results]

        frames = [r['frame_index'] for r in results]

        ps = float(self.PixelSize)
        if viewer is None:
            viewer = get_or_create_viewer()

        if overlay:
            for t in frames:
                raw = self.img(Channel=ch, verbose=False, frames=[self.frames[t]])
                raw = np.squeeze(raw)
                if raw.ndim == 3:
                    raw = raw.mean(axis=-1)
                lo, hi = np.percentile(raw, [1, 99.9])
                viewer.add_image(raw, blending='additive',
                                 contrast_limits=[lo, hi], scale=[ps, ps],
                                 colormap='gray', name=f'{ch} t={t}')

        all_pts, all_z, all_pv = [], [], []
        for res in results:
            t = res['frame_index']
            fl = self.framelabels[t]
            cen = np.asarray(fl.centroid, dtype=np.float64)
            if len(cen) == 0:
                continue
            interior = ~res['is_edge'] & np.isfinite(res['z_score'])
            all_pts.append(cen[interior])
            all_z.append(res['z_score'][interior])
            all_pv.append(res['pvalue'][interior])

        if not all_pts:
            raise ValueError("No valid interior cells found.")

        pts   = np.concatenate(all_pts)
        z_all = np.concatenate(all_z).astype(np.float64)
        pv_all = np.concatenate(all_pv).astype(np.float64)

        vmax_use = vmax if vmax is not None else float(np.nanpercentile(np.abs(z_all), 99))
        norm = mcolors.TwoSlopeNorm(vmin=-vmax_use, vcenter=0.0, vmax=vmax_use)
        rgba = cm.get_cmap(colormap)(norm(z_all))
        rgba[pv_all > max_pvalue, 3] = 0.2

        layer = viewer.add_points(
            pts,
            face_color=rgba,
            edge_width=0,
            size=size,
            blending='translucent',
            scale=[ps, ps],
            name=f'Gi* {ch}',
            properties={'Gi_star': z_all, 'pvalue': pv_all},
        )

        # ── cluster outlines ──────────────────────────────────────────────────
        cluster_results = []
        if show_clusters:
            hulls_hot, hulls_cold = [], []
            centroid_pts, centroid_ids = [], []

            for res in results:
                t = res['frame_index']
                fl  = self.framelabels[t]
                xy  = np.asarray(fl.XY, dtype=np.float64)
                coords_um = res['coords']
                z     = res['z_score']
                pv    = res['pvalue']
                is_edge = res.get('is_edge', np.zeros(len(z), dtype=bool))

                sig = ~is_edge & np.isfinite(z) & (np.abs(z) >= min_z) & (pv <= max_pvalue)
                frame_clusters = {'frame_index': t, 'hot': {}, 'cold': {}}

                for sign, label, hulls_list, edge_col in [
                    (1,  'hot',  hulls_hot,  '#ff1493'),
                    (-1, 'cold', hulls_cold, '#00bfff'),
                ]:
                    mask = sig & (np.sign(z) == sign)
                    if mask.sum() < min_cells:
                        frame_clusters[label] = {'n_clusters': 0, 'cluster_sizes': []}
                        continue

                    sig_coords = coords_um[mask]
                    eps = radius
                    db = DBSCAN(eps=eps, min_samples=min_cells).fit(sig_coords)
                    db_labels = db.labels_
                    n_cl = int((db_labels >= 0).any() and db_labels.max() + 1 or 0)
                    sizes = [(db_labels == k).sum() for k in range(n_cl)]
                    frame_clusters[label] = {'n_clusters': n_cl, 'cluster_sizes': sizes}

                    for k in range(n_cl):
                        pts_um = sig_coords[db_labels == k]
                        pts_px = (pts_um - xy) / ps
                        if len(pts_px) >= 3:
                            try:
                                verts = pts_px[ConvexHull(pts_px).vertices]
                            except Exception:
                                verts = pts_px
                        else:
                            verts = pts_px
                        if len(verts) < 2:
                            continue
                        hulls_list.append(verts)
                        centroid_pts.append(pts_px.mean(axis=0))
                        centroid_ids.append(f'{"H" if sign == 1 else "C"}{k+1}')

                cluster_results.append(frame_clusters)

            for hulls_list, edge_col, name_suffix in [
                (hulls_hot,  '#ff1493', 'hot clusters'),
                (hulls_cold, '#00bfff', 'cold clusters'),
            ]:
                if hulls_list:
                    viewer.add_shapes(
                        hulls_list,
                        shape_type='polygon',
                        face_color=[1, 1, 1, 0.0],
                        edge_color=edge_col,
                        edge_width=10,
                        scale=[ps, ps],
                        name=f'Gi* {ch} {name_suffix}',
                    )

            if centroid_pts:
                viewer.add_points(
                    np.array(centroid_pts),
                    properties={'label': np.array(centroid_ids)},
                    text='label',
                    face_color=[0, 0, 0, 0],
                    edge_width=0,
                    size=size * 1.5,
                    blending='translucent',
                    scale=[ps, ps],
                    name=f'Gi* {ch} cluster labels',
                )

        return layer, cluster_results

    def mark_variogram(self, ch_i, ch_j=None, frame=None, max_r=200.0, dr=5.0,
                       intensity='mean', periring=False, seed=42, ffield=True,
                       img=False, recompute=False):
        """Normalized mark variogram / cross-variogram γ̃(r). See spatial.mark_variogram."""
        _chj = ch_j or ch_i
        key = f"mark_variogram|{ch_i}|{_chj}|{_frame_key(frame)}|max_r={max_r}|dr={dr}|{intensity}|periring={periring}|ffield={ffield}|img={img}"
        if not recompute and key in self.spatial:
            return self.spatial[key].data
        from oyLabImaging.Processing.spatial import mark_variogram
        result = mark_variogram(self, ch_i=ch_i, ch_j=ch_j, frame=frame,
                                max_r=max_r, dr=dr, intensity=intensity,
                                periring=periring, seed=seed, ffield=ffield, img=img)
        _kw = _hint_kw(ch_j=(ch_j, None), frame=(frame, None), max_r=(max_r, 200.0),
                       dr=(dr, 5.0), ffield=(ffield, True), intensity=(intensity, 'mean'),
                       periring=(periring, False), img=(img, False))
        self._cache_spatial(
            key, result,
            f".mark_variogram('{ch_i}'{_kw})",
            f".plot_mark_variogram('{ch_i}'{_kw})",
        )
        return result

    def plot_mark_variogram(self, ch_i, ch_j=None, frame=None, max_r=200.0, dr=5.0,
                            intensity='mean', periring=False, seed=42,
                            ffield=True, img=False, recompute=False, ax=None, **kwargs):
        """Compute and plot the mark variogram / cross-variogram γ̃(r)."""
        from oyLabImaging.Processing.spatial import plot_radial
        result = self.mark_variogram(ch_i=ch_i, ch_j=ch_j, frame=frame,
                                     max_r=max_r, dr=dr, intensity=intensity,
                                     periring=periring, seed=seed, ffield=ffield,
                                     img=img, recompute=recompute)
        return plot_radial(result, ax=ax, **kwargs)

    def spatial_regions(self, channels, frame=None, radius=50.0, n_regions=None,
                        method='kmeans', intensity='mean', periring=False,
                        seed=42, ffield=True, max_k=10, recompute=False):
        """Multivariate spatial regionalization. See spatial.spatial_regions for details."""
        _chs = ",".join(sorted(channels))
        key = f"spatial_regions|{_chs}|{_frame_key(frame)}|r={radius}|k={n_regions}|{method}|{intensity}|periring={periring}|ffield={ffield}"
        if not recompute and key in self.spatial:
            return self.spatial[key].data
        from oyLabImaging.Processing.spatial import spatial_regions as _sr
        result = _sr(self, channels=channels, frame=frame, radius=radius,
                     n_regions=n_regions, method=method, intensity=intensity,
                     periring=periring, seed=seed, ffield=ffield, max_k=max_k)
        # Strip the large z-scored niche matrix before caching — easy to recompute
        def _strip(r):
            s = dict(r)
            s.pop('niche', None)
            return s
        cached = [_strip(r) for r in result] if isinstance(result, list) else _strip(result)
        _chs_repr = repr(list(channels))
        _kw = _hint_kw(frame=(frame, None), radius=(radius, 50.0), n_regions=(n_regions, None),
                       method=(method, 'kmeans'), ffield=(ffield, True),
                       intensity=(intensity, 'mean'), periring=(periring, False))
        self._cache_spatial(
            key, cached,
            f".spatial_regions({_chs_repr}{_kw})",
            f".plot_spatial_regions({_chs_repr}{_kw})",
        )
        return result

    def plot_spatial_regions(self, channels, frame=None, radius=50.0, n_regions=None,
                             method='kmeans', intensity='mean', periring=False,
                             seed=42, ffield=True, max_k=10,
                             overlay=True, size=8, viewer=None, recompute=False):
        """Compute spatial regions and visualise as a colour-coded Points layer in napari.

        Each region gets a distinct colour (tab10/tab20).  The layer name reports
        the number of regions and silhouette score.  The *marker_profiles* table
        (mean niche expression per region) is printed to stdout for interpretation.

        Parameters
        ----------
        channels : list of str   Marker channels used for regionalization.
        frame : int or None
        radius : float           Neighborhood radius in µm.
        n_regions : int or None  Number of regions (None = auto via silhouette).
        overlay : bool           Add raw images for each channel behind the points.
        size : int               Point size in pixels.
        viewer : napari.Viewer or None

        Returns
        -------
        list of result dicts (same as spatial_regions)
        """
        import matplotlib.cm as cm
        from oyLabImaging.Processing.imvisutils import get_or_create_viewer

        results = self.spatial_regions(channels=channels, frame=frame, radius=radius,
                                       n_regions=n_regions, method=method,
                                       intensity=intensity, periring=periring,
                                       seed=seed, ffield=ffield, max_k=max_k,
                                       recompute=recompute)
        if isinstance(results, dict):
            results = [results]

        frames = [r['frame_index'] for r in results]

        ps = float(self.PixelSize)
        if viewer is None:
            viewer = get_or_create_viewer()

        if overlay:
            for ch in channels:
                for t in frames:
                    raw = self.img(Channel=ch, verbose=False, frames=[self.frames[t]])
                    raw = np.squeeze(raw)
                    if raw.ndim == 3:
                        raw = raw.mean(axis=-1)
                    lo, hi = np.percentile(raw, [1, 99.9])
                    viewer.add_image(raw, blending='additive',
                                     contrast_limits=[lo, hi], scale=[ps, ps],
                                     name=f'{ch} t={t}')

        for res in results:
            t = res['frame_index']
            fl = self.framelabels[t]
            cen = np.asarray(fl.centroid, dtype=np.float64)
            labels = res['labels']
            k = res['n_regions']
            sil = res['silhouette']

            cmap = cm.get_cmap('tab10' if k <= 10 else 'tab20')
            face_colors = np.array([cmap(int(lbl) % cmap.N) for lbl in labels])

            viewer.add_points(
                cen,
                face_color=face_colors,
                edge_width=0,
                size=size,
                blending='translucent',
                scale=[ps, ps],
                name=f'regions t={t} (k={k}, sil={sil:.2f})',
                properties={'region': labels},
            )

            # Print region profiles for interpretation
            import pandas as pd
            df = pd.DataFrame(res['marker_profiles'],
                               columns=res['channels'],
                               index=[f'region {i}' for i in range(k)])
            print(f"\nFrame {t} — {k} regions (silhouette={sil:.3f})")
            print(df.round(3).to_string())

        return results

    def gwr(self, ch_y, ch_x, frame=None, bandwidth=50.0, kernel='gaussian',
            intensity='mean', periring=False, seed=42, ffield=True, recompute=False):
        """Geographically Weighted Regression. See spatial.gwr for details."""
        _chx = [ch_x] if isinstance(ch_x, str) else list(ch_x)
        key = f"gwr|{ch_y}|{','.join(_chx)}|{_frame_key(frame)}|bw={bandwidth}|{kernel}|{intensity}|periring={periring}|ffield={ffield}"
        if not recompute and key in self.spatial:
            return self.spatial[key].data
        from oyLabImaging.Processing.spatial import gwr as _gwr
        result = _gwr(self, ch_y=ch_y, ch_x=ch_x, frame=frame,
                      bandwidth=bandwidth, kernel=kernel, intensity=intensity,
                      periring=periring, seed=seed, ffield=ffield)
        _chx_repr = repr(ch_x)
        _kw = _hint_kw(frame=(frame, None), bandwidth=(bandwidth, 50.0),
                       kernel=(kernel, 'gaussian'), ffield=(ffield, True),
                       intensity=(intensity, 'mean'), periring=(periring, False))
        self._cache_spatial(
            key, result,
            f".gwr('{ch_y}', {_chx_repr}{_kw})",
            f".plot_gwr('{ch_y}', {_chx_repr}{_kw})",
        )
        return result

    def plot_gwr(self, ch_y, ch_x, frame=None, bandwidth=50.0, kernel='gaussian',
                 intensity='mean', periring=False, ffield=True,
                 show='slope', predictor_idx=0, max_pvalue=0.05,
                 vmax=None, overlay=True, size=8, viewer=None, recompute=False):
        """Compute GWR and visualise spatially-varying regression coefficients in napari.

        Parameters
        ----------
        ch_y, ch_x  : response and predictor channel(s)
        bandwidth   : kernel bandwidth in µm
        kernel      : 'gaussian' or 'bisquare'
        show        : what to colour points by —
                      'slope'     local slope for predictor_idx (coolwarm, diverging at 0)
                      'r_squared' local R² (viridis, 0–1)
                      'intercept' local intercept (coolwarm)
                      't_stat'    t-statistic for predictor_idx (coolwarm)
        predictor_idx : which predictor's slope/t_stat to show (0 = first ch_x)
        max_pvalue  : cells with pvalue > this are shown at 20% opacity
                      (applied to the chosen predictor; ignored for r_squared)
        vmax        : colour scale maximum (default: 99th percentile)
        overlay     : add raw ch_y image behind the points
        """
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        from oyLabImaging.Processing.imvisutils import get_or_create_viewer

        results = self.gwr(ch_y=ch_y, ch_x=ch_x, frame=frame,
                           bandwidth=bandwidth, kernel=kernel,
                           intensity=intensity, periring=periring,
                           ffield=ffield, recompute=recompute)
        if isinstance(results, dict):
            results = [results]

        ps = float(self.PixelSize)
        if viewer is None:
            viewer = get_or_create_viewer()

        if overlay:
            for res in results:
                t = res['frame_index']
                raw = self.img(Channel=ch_y, verbose=False, frames=[self.frames[t]])
                raw = np.squeeze(raw)
                if raw.ndim == 3:
                    raw = raw.mean(axis=-1)
                lo, hi = np.percentile(raw, [1, 99.9])
                viewer.add_image(raw, blending='additive',
                                 contrast_limits=[lo, hi], scale=[ps, ps],
                                 colormap='gray', name=f'{ch_y} t={t}')

        layers = []
        for res in results:
            t   = res['frame_index']
            fl  = self.framelabels[t]
            cen = np.asarray(fl.centroid, dtype=np.float64)

            beta   = res['beta']        # (N, p+1)
            r_sq   = res['r_squared']   # (N,)
            t_st   = res['t_stat']      # (N, p+1)
            pval   = res['pvalue']      # (N, p+1)
            col_i  = predictor_idx + 1  # +1 for intercept column

            if show == 'r_squared':
                vals   = r_sq
                cmap   = 'viridis'
                diverge = False
                pv_dim  = None          # don't dim by p-value for R²
                label  = f'GWR R² {ch_y}~{ch_x} t={t}'
            elif show == 'intercept':
                vals   = beta[:, 0]
                cmap   = 'coolwarm'
                diverge = True
                pv_dim  = pval[:, 0]
                label  = f'GWR intercept {ch_y} t={t}'
            elif show == 't_stat':
                vals   = t_st[:, col_i]
                cmap   = 'coolwarm'
                diverge = True
                pv_dim  = pval[:, col_i]
                _chx_l = res['ch_x'][predictor_idx]
                label  = f'GWR t-stat {ch_y}~{_chx_l} t={t}'
            else:  # 'slope' (default)
                vals   = beta[:, col_i]
                cmap   = 'coolwarm'
                diverge = True
                pv_dim  = pval[:, col_i]
                _chx_l = res['ch_x'][predictor_idx]
                label  = f'GWR slope {ch_y}~{_chx_l} t={t}'

            finite = np.isfinite(vals)
            vmax_use = vmax if vmax is not None else float(
                np.nanpercentile(np.abs(vals[finite]), 99) if diverge
                else np.nanpercentile(vals[finite], 99)
            )

            if diverge:
                norm = mcolors.TwoSlopeNorm(vmin=-vmax_use, vcenter=0.0, vmax=vmax_use)
            else:
                norm = mcolors.Normalize(vmin=0.0, vmax=vmax_use)

            rgba = cm.get_cmap(cmap)(norm(np.where(finite, vals, 0.0)))
            rgba[~finite, 3] = 0.0          # hide cells with no fit

            if pv_dim is not None:
                rgba[np.isfinite(pv_dim) & (pv_dim > max_pvalue), 3] = 0.2

            layer = viewer.add_points(
                cen,
                face_color=rgba,
                edge_width=0,
                size=size,
                blending='translucent',
                scale=[ps, ps],
                name=label,
                properties={show: np.where(finite, vals, np.nan)},
            )
            layers.append(layer)

        return layers[0] if len(layers) == 1 else layers

    def fit_corr_lengthscale(self, ch, ch_j=None, frame=None,
                              img=False, ffield=True,
                              r_min=10.0, max_r=None, dr=None,
                              plot=False, **fit_kwargs):
        """Fit g(r) to A·exp(−r/λ) + C to extract a spatial length scale.

        Defaults to img=False (cell-level correlation).  Set img=True for
        pixel-level FFT correlation (sub-cell resolution, Oyler-Yaniv et al. 2017).

        The fit starts at r_min (~one cell diameter) to exclude within-nucleus
        autocorrelation.

        Parameters
        ----------
        ch : str
            Source channel.
        ch_j : str, optional
            Target channel.  Defaults to ch (autocorrelation).
        frame : int, list of int, or None
        img : bool
            True = pixel-level correlation (default); False = cell-level.
        ffield : bool
            Flat-field flag forwarded to radial_corr.
        r_min : float
            Minimum radius (µm) for the exponential fit.
        max_r : float, optional
            Maximum radius (µm).  Defaults: 250 (img=True), 200 (img=False).
        dr : float, optional
            Bin width (µm).  Defaults: 0.5 (img=True), 5 (img=False).
        plot : bool
            If True, overlay fit on g(r) plot and return (fit, ax).
        **fit_kwargs
            Forwarded to fit_lengthscale (e.g. min_pairs).

        Returns
        -------
        fit : dict  (or (fit, ax) when plot=True)
        """
        from oyLabImaging.Processing.spatial import (
            radial_corr, fit_lengthscale, plot_radial,
        )

        result = radial_corr(self, ch_i=ch, ch_j=ch_j, frame=frame,
                             img=img, ffield=ffield, max_r=max_r, dr=dr)
        fit = fit_lengthscale(result, r_min=r_min, **fit_kwargs)

        if plot:
            ax = plot_radial(result, fit_results=fit)
            return fit, ax
        return fit

    def spectral_basis(self, frame=None, n_modes=None, alpha=0.05,
                       recompute=False):
        """Build (or retrieve cached) cotangent Laplacian spectral basis.

        Uses the alpha-shape triangulation + gpytoolbox cotangent Laplacian,
        following Jerison et al. 2025 (PNAS) exactly.  The basis is
        geometry-only and cached so multiple channels share it.

        Parameters
        ----------
        frame   : int, list of int, or None
        n_modes : int or None
            None (default) → full dense eigendecomposition via scipy.linalg.eigh
            (all N-1 oscillatory modes; practical for N ≲ 5 000).
            Pass an integer for large datasets.
        alpha   : float
            Alpha-shape parameter (~1/max_circumradius, µm^{-1}).
            Default 0.05 (max circumradius ≈ 20 µm).

        Returns
        -------
        SpectralBasis (single frame) or list of SpectralBasis
        """
        key = f"spectral_basis|{_frame_key(frame)}|nm={n_modes}|a={alpha}"
        if not recompute and key in self.spatial:
            return self.spatial[key].data
        from oyLabImaging.Processing.spatial import build_spectral_basis
        result = build_spectral_basis(self, frame=frame, n_modes=n_modes,
                                      alpha=alpha)
        _kw = _hint_kw(frame=(frame, None), n_modes=(n_modes, None),
                       alpha=(alpha, 0.05))
        self._cache_spatial(
            key, result,
            f".spectral_basis({_kw.lstrip(', ')})",
            f".plot_spatial_power_spectrum('<ch>'{_kw})",
        )
        return result

    def spectral_power_spectrum(self, ch, frame=None, n_modes=None, alpha=0.05,
                                intensity='mean', periring=False, ffield=True,
                                n_permutations=100, seed=42, recompute=False):
        """Spatial power spectrum via cotangent Laplacian spectral decomposition.

        Implements Jerison et al. 2025 (PNAS).  Builds the spectral basis once
        (cached), then projects z-scored expression onto the eigenmodes and
        computes fractional variance per spatial length scale.  A permutation
        null identifies which length scales carry biologically organised signal.

        Parameters
        ----------
        ch            : str or list of str
        frame         : int, list of int, or None
        n_modes       : int or None  (see spectral_basis)
        alpha         : float        (see spectral_basis)
        intensity     : str  'mean', 'median', 'max', 'min', 'ninety'
        periring      : bool
        ffield        : bool
        n_permutations: int
        seed          : int

        Returns
        -------
        dict or list of dicts with keys:
            power            (k, n_ch)  fractional variance per mode
            power_null_mean  (k, n_ch)
            power_null_std   (k, n_ch)
            signal_to_null   (k, n_ch)
            char_lengthscale (n_ch,)    power-weighted mean (paper's estimator)
            peak_lengthscale (n_ch,)    length scale of highest-SNR mode
            lengthscales_um  (k,)       2/sqrt(λ) per mode
            eigenvalues      (k,)
            channels         list of str
            frame_index      int
        """
        channels = [ch] if isinstance(ch, str) else list(ch)
        ch_key   = '|'.join(channels)
        key = (f"spectral_ps|{ch_key}|{_frame_key(frame)}|nm={n_modes}"
               f"|a={alpha}|np={n_permutations}|{intensity}"
               f"|periring={periring}|ffield={ffield}")
        if not recompute and key in self.spatial:
            return self.spatial[key].data

        import warnings
        from oyLabImaging.Processing.spatial import _frame_data_multi, _resolve_frames

        frames = _resolve_frames(self, frame)
        if ffield:
            uncorrected = [t for t in frames
                           if not getattr(self.framelabels[t], '_ffield', False)]
            if uncorrected:
                warnings.warn(
                    f"ffield=True but {len(uncorrected)} frame(s) were segmented "
                    "without flat-field correction.", stacklevel=2)

        bases = self.spectral_basis(frame=frame, n_modes=n_modes, alpha=alpha,
                                    recompute=recompute)
        if not isinstance(bases, list):
            bases = [bases]

        per_frame = []
        for basis in bases:
            t  = basis.frame_index
            fl = self.framelabels[t]
            coords, vals = _frame_data_multi(fl, channels, intensity, periring)
            if coords is None:
                continue
            mu    = vals.mean(axis=0)
            sigma = vals.std(axis=0)
            sigma[sigma < 1e-12] = 1.0
            z = (vals - mu) / sigma
            res = basis.power_spectrum(z, n_permutations=n_permutations, seed=seed)
            res['channels'] = channels
            per_frame.append(res)

        if not per_frame:
            raise ValueError("No valid frames.")

        result = (per_frame[0] if isinstance(frame, (int, np.integer)) or
                  (isinstance(frame, list) and len(frame) == 1)
                  else per_frame)

        _kw = _hint_kw(frame=(frame, None), n_modes=(n_modes, None),
                       alpha=(alpha, 0.05),
                       n_permutations=(n_permutations, 100), ffield=(ffield, True),
                       intensity=(intensity, 'mean'), periring=(periring, False))
        self._cache_spatial(
            key, result,
            f".spectral_power_spectrum('{ch}'{_kw})",
            f".plot_spatial_power_spectrum('{ch}'{_kw})",
        )
        return result

    def plot_spatial_power_spectrum(self, ch, frame=None, n_modes=None,
                                    alpha=0.05, intensity='mean',
                                    periring=False, ffield=True,
                                    n_permutations=100, seed=42,
                                    recompute=False, ax=None, **plot_kwargs):
        """Compute and plot spatial power spectrum in one call.

        See spectral_power_spectrum for parameter documentation.
        Returns matplotlib.axes.Axes.
        """
        from oyLabImaging.Processing.spatial import plot_spatial_power_spectrum
        result = self.spectral_power_spectrum(
            ch=ch, frame=frame, n_modes=n_modes, alpha=alpha,
            intensity=intensity, periring=periring, ffield=ffield,
            n_permutations=n_permutations, seed=seed, recompute=recompute,
        )
        return plot_spatial_power_spectrum(result, ax=ax, **plot_kwargs)

    def plot_wavenumber_power_spectrum(self, ch, frame=None, n_modes=None,
                                       alpha=0.05, intensity='mean',
                                       periring=False, ffield=True,
                                       n_permutations=100, seed=42,
                                       recompute=False, ax=None, **plot_kwargs):
        """Compute and plot fraction of power vs wavenumber k (µm⁻¹) — Figure 6 style.

        Same parameters as spectral_power_spectrum.
        Returns matplotlib.axes.Axes.
        """
        from oyLabImaging.Processing.spatial import plot_wavenumber_power_spectrum
        result = self.spectral_power_spectrum(
            ch=ch, frame=frame, n_modes=n_modes, alpha=alpha,
            intensity=intensity, periring=periring, ffield=ffield,
            n_permutations=n_permutations, seed=seed, recompute=recompute,
        )
        return plot_wavenumber_power_spectrum(result, ax=ax, **plot_kwargs)

    def plot_spatial_reconstruction(self, ch, frame=None, n_modes=None,
                                    alpha=0.05, intensity='mean',
                                    periring=False, ffield=True,
                                    n_modes_reconstruct=50,
                                    recompute=False, **plot_kwargs):
        """Figure 4C–style: raw data vs N-mode spatial reconstruction.

        For each channel, shows two scatter maps side by side: the centred
        expression values and the reconstruction using only the first
        *n_modes_reconstruct* low-frequency eigenmodes (spatial low-pass filter).

        Parameters
        ----------
        ch                   : str or list of str
        frame                : int or None
        n_modes              : int or None  (spectral basis truncation)
        alpha                : float        (alpha-shape parameter)
        intensity            : str
        periring             : bool
        ffield               : bool
        n_modes_reconstruct  : int  modes kept in reconstruction (default 50)
        recompute            : bool

        Returns matplotlib.figure.Figure.
        """
        from oyLabImaging.Processing.spatial import (
            plot_spatial_reconstruction, _frame_data_multi, _resolve_frames
        )
        channels = [ch] if isinstance(ch, str) else list(ch)
        frames   = _resolve_frames(self, frame)
        t        = frames[0]
        fl       = self.framelabels[t]

        basis = self.spectral_basis(frame=t, n_modes=n_modes, alpha=alpha,
                                    recompute=recompute)
        if isinstance(basis, list):
            basis = basis[0]

        coords, vals = _frame_data_multi(fl, channels, intensity, periring)
        if coords is None:
            raise ValueError(f"No cells in frame {t}.")

        return plot_spatial_reconstruction(
            coords, vals, basis,
            channels=channels,
            n_modes_reconstruct=n_modes_reconstruct,
            **plot_kwargs,
        )

    def plot_images(
        self,
        Channel="DeepBlue",
        Zindex=[0],
        frames=None,
        cmaps=["red", "green", "blue", "cyan", "magenta", "yellow"],
        **kwargs,
    ):
        """
        Parameters
        ----------
        Channel : [DeepBlue] str or list of strings
        Zindex : [0]

        Draws image stks in current napari viewer
        """

        Channel = (
            Channel
            if isinstance(Channel, list) or isinstance(Channel, np.ndarray)
            else [Channel]
        )

        if len(Channel) == 1:
            cmaps = ["gray"]

        if frames is None:
            frames = self.frames
        else:
            if not isinstance(frames, (list, np.ndarray)):
                frames = [frames]

        from oyLabImaging.Processing.improcutils import sample_stack
        from oyLabImaging.Processing.imvisutils import get_or_create_viewer

        viewer = get_or_create_viewer()
        viewer.scale_bar.unit = "um"
        viewer.scale_bar.font_size = 16

        ffield = kwargs.pop('ffield', None)
        for ind, ch in enumerate(Channel):
            stk = self.img(
                Channel=ch, verbose=True, Zindex=Zindex, frames=frames, ffield=ffield, **kwargs
            )
            stksmp = sample_stack(stk, int(stk.size / 1000))
            viewer.add_image(
                stk,
                blending="additive",
                contrast_limits=[np.percentile(stksmp, 1), np.percentile(stksmp, 99.9)],
                scale=[self.PixelSize, self.PixelSize],
                colormap=cmaps[ind % len(cmaps)],
                name=ch,
                **kwargs,
            )

    def plot_tracks(self, J=None, **kwargs):
        """
        Parameters
        ----------
        J : track indices - plots all tracks if not provided


        Draws overlaying tracks in current napari viewer
        """
        assert self._tracked, str(self.posname) + " not tracked yet"

        from oyLabImaging.Processing.imvisutils import get_or_create_viewer

        viewer = get_or_create_viewer()
        trackmat, J = self._tracksmat(J=J)
        # inds_to_include = self.trackinds[J]!=None
        # inds_to_include = np.array([self.trackinds[j] != None for j in J])
        # track_props = {'cell_id' :list( self.trackinds[J][np.where(inds_to_include)]), 'cell_T' : np.where(inds_to_include)[1],}

        tracklayer = viewer.add_tracks(
            trackmat, blending="additive", scale=[self.PixelSize, self.PixelSize]
        )
        # tracklayer.display_id=True
        return tracklayer

    def plot_points(
        self,
        Channel="DeepBlue",
        periring=False,
        colormap="plasma",
        func=lambda x: x,
        size=8,
        face_color="mean",
        **kwargs,
    ):
        """
        Parameters
        ----------
        Channel : [DeepBlue] str or list of strings
        periring : {[False], True} will error on Trueif periring wasnt calculated
        colormap : colormap to use for points
        func : transformation to color, i.e. np.log, logicle, or whatever

        Draws overlaying points colorcoded by intensity in current napari viewer
        """
        # assert self._tracked, str(pos) +' not tracked yet'

        from oyLabImaging.Processing.imvisutils import get_or_create_viewer

        viewer = get_or_create_viewer()
        try:
            pointsmat = self._pointmatrix
        except:
            self._calculate_pointmat()
            pointsmat = self._pointmatrix

        point_props = {
            "mean": func(np.concatenate(self.mean(Channel, periring=periring))).astype(np.float64),
            "ind": np.concatenate(self.index).astype(np.float64),
        }

        text = {
            "string": "{ind:.2f}",
            "size": 10,
            "color": "white",
            "translation": np.array([0, 0]),
        }
        pointlayer = viewer.add_points(
            pointsmat,
            properties=point_props,
            text="ind",  # [str(a) for a in np.concatenate(self.index)],
            face_color=face_color,
            edge_width=0,
            face_colormap=colormap,
            size=size,
            blending="translucent",
            scale=[1, self.PixelSize, self.PixelSize],
        )
        return pointlayer

    def property_matrix(
        self, prop="mean", channel=None, periring=False, keep_only=False
    ):
        """
        Parameters
        ----------
        prop : str - Property to return
        channel : str - for intensity based properties, channel name.
        periring : For intensity based features only. Perinuclear ring values.
        keep_only : {[False], True}

        returns property prop for all tracks in matrix form [N tracks x M timepoints x L dimensions of property]

        """
        return self.prop_mat(
            prop=prop, channel=channel, periring=periring, keep_only=keep_only
        )

    def prop_mat(self, prop="mean", channel=None, periring=False, keep_only=False):
        """
        Parameters
        ----------
        prop : str - Property to return
        channel : str - for intensity based properties, channel name.
        periring : For intensity based features only. Perinuclear ring values.
        keep_only : {[False], True}

        returns property prop for all tracks in matrix form [N tracks x M timepoints x L dimensions of property]


        """
        ch_props = ["mean", "median", "90th", "min", "max"]
        # props = ['area', 'convex_area','perimeter','eccentricity','solidity','inertia_tensor_eigvals', 'orientation']

        props = self.prop_list

        assert prop in props + ch_props, (
            "unsupported property, try: "
            + " ,".join(props)
            + ", OR, "
            + " ,".join(ch_props)
            + " and a channel name."
        )

        if prop in ch_props:
            assert channel in self.channels, (
                "requested channel does not exist, try " + " ,".join(self.channels)
            )
        peritext = ""
        if periring:
            peritext = "_periring"

        ts = self.get_track

        if keep_only:
            J = self.track_to_use
        else:
            J = range(ts(0).numtracks)

        if prop in ch_props:
            if ts(0).prop(prop + "_" + channel).ndim == 1:
                track_mat = np.empty((len(J), len(self.frames)))
            elif ts(0).prop(prop + "_" + channel).ndim == 2:
                track_mat = np.empty(
                    (
                        len(J),
                        len(self.frames),
                        ts(0).prop(prop + "_" + channel).shape[1],
                    )
                )

            track_mat[:] = np.NaN

            assert channel in self.channels
            for i in J:
                track_mat[i, ts(i).T] = ts(i).prop(prop + "_" + channel + peritext)
            return track_mat
        elif prop in props:
            if ts(0).prop(prop).ndim == 1:
                track_mat = np.empty((len(J), len(self.frames)))
            elif ts(0).prop(prop).ndim == 2:
                track_mat = np.empty(
                    (len(J), len(self.frames), ts(0).prop(prop).shape[1])
                )

            track_mat[:] = np.NaN
            for i, j in enumerate(J):
                track_mat[i, ts(j).T] = ts(j).prop(prop)
            return track_mat


def prepare_sparse_cost(shape, cc, ii, jj, cost_limit):
    """
    Transform the given sparse matrix extending it to a square sparse matrix.

    Parameters
    ==========
    shape: tuple
       - cost matrix shape
    (cc, ii, jj): tuple of floats, ints, ints)
        - cost matrix in COO format, see [1].
    cost_limit: float

    Returns
    =======
    cc, ii, kk
      - extended square cost matrix in CSR format

    Notes
    =====
    WARNING: Avoid using scipy.sparse.coo_matrix(cost) as it will not return the correct (cc, ii, jj).
    `coo_matrix` leaves out any zero values which are the most salient parts of the cost matrix.
    (cc, ii, jj) should include zero costs (if any) and skip all costs that are too large (infinite).

    1. https://en.wikipedia.org/wiki/Sparse_matrix
    """
    assert cost_limit < np.inf
    n, m = shape
    cc_ = np.r_[cc, [cost_limit] * n, [cost_limit] * m, [0] * len(cc)]
    ii_ = np.r_[
        ii,
        np.arange(0, n, dtype=np.uint32),
        np.arange(n, n + m, dtype=np.uint32),
        n + jj,
    ]
    jj_ = np.r_[
        jj,
        np.arange(m, n + m, dtype=np.uint32),
        np.arange(0, m, dtype=np.uint32),
        m + ii,
    ]
    order = np.lexsort((jj_, ii_))
    cc_ = cc_[order]
    kk_ = jj_[order]
    ii_ = np.bincount(ii_, minlength=shape[0] - 1)
    ii_ = np.r_[[0], np.cumsum(ii_)]
    ii_ = ii_.astype(np.uint32)
    assert ii_[-1] == 2 * len(cc) + n + m
    return cc_, ii_, kk_