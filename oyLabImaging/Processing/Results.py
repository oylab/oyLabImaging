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

    def __init__(self, MD=None, pth=None, threads=10, **kwargs):

        if pth is None:
            if MD is not None:
                self.pth = MD.base_pth
        else:
            self.pth = pth

        pth = self.pth

        if "results.pickle" in listdir(pth):
            r = results.load(pth, fname="results.pickle")
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
            + ", ".join(sorted([str(a) for a in self.PosLbls.keys()]))
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
    
    def setPosLbls(self, MD=None, groups=None, Position=None, **kwargs):
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
            assert np.all(
                np.isin(groups, self.groups)
            ), "some provided groups don't exist, try %s" % ", ".join(list(self.groups))
            Position = MD.unique("Position", group=groups)
        if Position is None:
            Position = self.PosNames

        elif type(Position) is not list:
            Position = [Position]

        for p in Position:
            print("\nProcessing position " + str(p))
            self.PosLbls.update({p: PosLbl(MD=MD, Pos=p, pth=MD.base_pth, **kwargs)})
        self.save()

    @alias(
        {
            "pos": "Position",
            "Pos": "Position",
            "position": "Position",
            "p": "Position",
        }
    )

    def segment_and_extract_features(self, MD=None, groups=None, Position=None, **kwargs):
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
        return self.setPosLbls(MD=MD, groups=groups, Position=Position, **kwargs)
    
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

    def save(self, fname="results.pickle"):
        """
        save results
        """
        with open(join(self.pth, fname), "wb") as dbfile:
            cloudpickle.dump(self, dbfile)
            print("saved results")

    @classmethod
    def load(cls, pth, fname="results.pickle"):
        """
        load results
        """
        with open(join(pth, fname), "rb") as dbfile:
            r = dill.load(dbfile)
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
        return np.concatenate([[pn]*self._outer.PosLbls[pn].num[frame] for pn in self.Position])
    
    @property
    def cellperposition(self):
        frame = self.frame
        return [self._outer.PosLbls[pn].num[frame] for pn in self.Position]

    def mean(self, ch, periring=False):
        ch = ch if isinstance(ch, (list, np.ndarray)) else [ch]
        frame = self.frame
        a = [
            self._outer.PosLbls[pn].framelabels[frame].regionprops[[''.join(['mean_',c, '_periring'* periring]) for c in ch]]
            for pn in self.Position if self._outer.PosLbls[pn].num
        ]   
        a = np.concatenate(a)
        return a

    def median(self, ch, periring=False):
        ch = ch if isinstance(ch, (list, np.ndarray)) else [ch]
        frame = self.frame
        a = [
            self._outer.PosLbls[pn].framelabels[frame].regionprops[[''.join(['median_',c, '_periring'* periring]) for c in ch]]
            for pn in self.Position if self._outer.PosLbls[pn].num
        ]   
        a = np.concatenate(a)
        return a


    def minint(self, ch, periring=False):
        ch = ch if isinstance(ch, (list, np.ndarray)) else [ch]
        frame = self.frame
        a = [
            self._outer.PosLbls[pn].framelabels[frame].regionprops[[''.join(['min_',c, '_periring'* periring]) for c in ch]]
            for pn in self.Position if self._outer.PosLbls[pn].num
        ]   
        a = np.concatenate(a)
        return a

    def maxint(self, ch, periring=False):
        ch = ch if isinstance(ch, (list, np.ndarray)) else [ch]
        frame = self.frame
        a = [
            self._outer.PosLbls[pn].framelabels[frame].regionprops[[''.join(['max_',c, '_periring'* periring]) for c in ch]]
            for pn in self.Position if self._outer.PosLbls[pn].num
        ]   
        a = np.concatenate(a)
        return a

    def ninetyint(self, ch, periring=False):
        ch = ch if isinstance(ch, (list, np.ndarray)) else [ch]
        frame = self.frame
        a = [
            self._outer.PosLbls[pn].framelabels[frame].regionprops[[''.join(['90th_',c, '_periring'* periring]) for c in ch]]
            for pn in self.Position if self._outer.PosLbls[pn].num
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
