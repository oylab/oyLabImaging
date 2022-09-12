# PosLbl module for microscopy data.
# Contains all single TP FrameLbls for a specific well/position
# Deals with tracking
# AOY

import sys
from functools import partial

import lap
import multiprocess as mp  # import Pool, set_start_method

mp.set_start_method("spawn", force=True)

import numpy as np
from scipy.spatial import KDTree
from tqdm import tqdm

from oyLabImaging import Metadata
from oyLabImaging.Processing import FrameLbl


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
        register=True,
        **kwargs
    ):

        if any([Pos is None]):
            raise ValueError("Please provide position")

        if pth is None:
            if MD is not None:
                self.pth = MD.base_pth
        else:
            self.pth = pth

        if MD is None:
            MD = Metadata(pth)

        if MD().empty:
            raise AssertionError("No metadata found in supplied path")

        if Pos not in MD.posnames:
            raise AssertionError("Position does not exist in dataset")
        self.posname = Pos

        self.channels = MD.unique("Channel", Position=Pos)
        self.acq = MD.unique("acq", Position=Pos)

        if frames is None:
            self.frames = MD.unique("frame", Position=Pos)
        elif type(frames) is not list:
            self.frames = [frames]
        else:
            self.frames = frames

        if len(self.frames) == 1:
            register = False
        self._registerflag = register
        self._tracked = False
        self._splitflag = False
        # self.PixelSize = MD.unique('PixelSize')[0]

        # Create all framelabels for the different TPs. This will segment and measure stuff.

        with mp.Pool(threads) as ppool:
            frames = list(
                tqdm(
                    ppool.imap(
                        partial(
                            FrameLbl,
                            MD=MD,
                            pth=pth,
                            Pos=Pos,
                            NucChannel=NucChannel,
                            register=self._registerflag,
                            **kwargs
                        ),
                        self.frames,
                    ),
                    total=len(self.frames),
                )
            )
            # ppool.close()
            # ppool.join()
        self.framelabels = np.array(frames)
        self._calculate_pointmat()
        print("\nFinished loading and segmenting position " + str(Pos))

    def __call__(self):
        print("PosLbl object for position " + str(self.posname) + ".")
        print("\nThe path to the experiment is: \n " + self.pth)
        print("\n " + str(len(self.frames)) + " frames processed.")

        print("\nAvailable channels are : " + ", ".join(list(self.channels)) + ".")

    np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

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
        return np.array([r.num for r in self.framelabels])

    @property
    def centroid(self):
        return np.array([r.centroid for r in self.framelabels])

    @property
    def weighted_centroid(self):
        return np.array([r.weighted_centroid for r in self.framelabels])

    @property
    def area(self):
        return np.array([r.area for r in self.framelabels])

    @property
    def index(self):
        return np.array([r.index for r in self.framelabels])

    @property
    def centroid_um(self):
        return np.array([r.centroid_um for r in self.framelabels])

    @property
    def weighted_centroid_um(self):
        return np.array([r.weighted_centroid_um for r in self.framelabels])

    @property
    def area_um2(self):
        return np.array([r.area_um2 for r in self.framelabels])

    def mean(self, ch, periring=False):
        return np.array([r.mean(ch, periring=periring) for r in self.framelabels])

    def median(self, ch, periring=False):
        return np.array([r.median(ch, periring=periring) for r in self.framelabels])

    def minint(self, ch, periring=False):
        return np.array([r.minint(ch, periring=periring) for r in self.framelabels])

    def maxint(self, ch, periring=False):
        return np.array([r.maxint(ch, periring=periring) for r in self.framelabels])

    def ninetyint(self, ch, periring=False):
        return np.array([r.ninetyint(ch, periring=periring) for r in self.framelabels])

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

        def show_movie(
            self,
            Channel=["DeepBlue"],
            boxsize=75,
            cmaps=["red", "green", "blue", "cyan", "magenta", "yellow"],
            **kwargs
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

            from oyLabImaging.Processing.improcutils import sample_stack
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
        TODO: add support for splitting cells

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
                dists, idx = T.query(cents[i], k=12, distance_upper_bound=search_radius)

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
        if self.num[i + 1] > 0:
            if i + 1 < len(self.frames) - 1:
                if l > -1:
                    return np.append(
                        l, self._getTrackLinks(i + 1, self.framelabels[i].link1in2[l])
                    )
                else:
                    pass
            elif i + 1 == len(self.frames) - 1:
                if l > -1:
                    return l
                else:
                    pass
        else:
            return np.array([])

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
        **kwargs
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
                [np.where(~np.isnan(r.astype("float")))[0][0] for r in trackbits]
            )
            trackends = np.array(
                [np.where(~np.isnan(r.astype("float")))[0][-1] for r in trackbits]
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
                [np.where(~np.isnan(r.astype("float")))[0][0] for r in trackbits]
            )
            trackends = np.array(
                [np.where(~np.isnan(r.astype("float")))[0] for r in trackbits]
            )

            #             dtmat = np.expand_dims(trackstarts,1)-np.expand_dims(trackends,0)
            #             dt1array = np.zeros_like(dtmat)
            #             for (x, y), element in np.ndenumerate(dtmat):
            #                 try:
            #                     dt1array[x,y] = np.nonzero(element==1)[0][0]
            #                 except:
            #                     dt1array[x,y] = None

            #             possiblelinks = np.transpose(np.where(dt1array!=None))

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

    def img(self, Channel=None, Zindex=[0], **kwargs):
        """
        Parameters
        ------
        Channel : [DeepBlue] str or list of strings
        register : {[True], False}
        Zindex=[0]

        Returns
        -------
        Image stack of given channels
        """
        from oyLabImaging import Metadata

        if Channel is None:
            Channel = self.channels[0]
            print("loading " + Channel)

        pth = self.pth
        MD = Metadata(pth, verbose=False)
        return MD.stkread(
            Channel=Channel,
            Position=self.posname,
            register=self._registerflag,
            Zindex=Zindex,
            **kwargs
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

    def plot_images(
        self,
        Channel="DeepBlue",
        Zindex=[0],
        frames=None,
        cmaps=["red", "green", "blue", "cyan", "magenta", "yellow"],
        **kwargs
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

        for ind, ch in enumerate(Channel):
            stk = self.img(
                Channel=ch, verbose=True, Zindex=Zindex, frames=frames, **kwargs
            )
            stksmp = sample_stack(stk, int(stk.size / 1000))
            viewer.add_image(
                stk,
                blending="additive",
                contrast_limits=[np.percentile(stksmp, 1), np.percentile(stksmp, 99.9)],
                scale=[self.PixelSize, self.PixelSize],
                colormap=cmaps[ind % len(cmaps)],
                **kwargs
            )

    def plot_tracks(self, J=None, **kwargs):
        """
        Parameters
        ----------
        J : track indices - plots all tracks if not provided


        Draws overlaying tracks in current napari viewer
        """
        assert self._tracked, str(pos) + " not tracked yet"

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
        **kwargs
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
            "mean": func(np.concatenate(self.mean(Channel, periring=periring))),
            "ind": np.concatenate(self.index),
        }
        pointlayer = viewer.add_points(
            pointsmat,
            properties=point_props,
            text="ind",
            face_color="mean",
            edge_width=0,
            face_colormap=colormap,
            size=20,
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
            assert (
                channel in self.channels
            ), "requested channel does not exist, try " + " ,".join(self.channels)
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
