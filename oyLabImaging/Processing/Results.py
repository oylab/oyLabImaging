# Results module for microscopy data.
# Aggregates all Pos labels for a specific experiment

from os import listdir
from os.path import join
import os

import cloudpickle
import dill
import numpy as np

# AOY
from oyLabImaging import Metadata
from oyLabImaging.Processing import PosLbl
from oyLabImaging.Processing.generalutils import alias
from natsort import natsorted

# Spatial statistics imports 
import pandas as pd             
import anndata as ad            
import squidpy as sq            
import warnings                 
from itertools import combinations  
from esda.moran import Moran_BV, Moran_Local, Moran_Local_BV     
from libpysal.weights import W      

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
            "Position": "pos",
            "Pos": "pos",
            "position": "pos",
            "p": "pos",
            "channel": "Channel",
            "ch": "Channel",
            "c": "Channel",
        }
    )
    def show_spatial_map_napari(self, pos, Channel, metric='local_morans_i', frame_idx='all', size=15, load_images=True, clear_viewer=False, **kwargs):
        """
        Dedicated function to view spatial hotspots and expression in Napari.
        
        Parameters
        ----------
        pos : str
            Position name
        Channel : str
            Channel or Bivariate pair (e.g. 'Red' or 'Red vs FarRed')
        metric : str
            'local_morans_i', 'local_bivariate_moran', or 'expression'.
        frame_idx : int or 'all'
            Index of the frame to display, or 'all' to load the full time-lapse movie.
        size : int
            Size of the points in Napari
        load_images : bool
            If True, loads the TIFFs. If False, opens instantly with just the spatial dots.
        clear_viewer : bool
            If True, wipes the Napari viewer before loading. Critical if switching between 1-frame and 'all' frames.
        """
        from oyLabImaging.Processing.imvisutils import get_or_create_viewer
        from oyLabImaging.Processing.improcutils import sample_stack
        import matplotlib.colors as mc
        import numpy as np

        if pos not in self.PosLbls:
            print(f"Error: Position {pos} not found.")
            return

        # --- LOGICAL VALIDATION ---
        if metric == 'local_bivariate_moran' and ' vs ' not in Channel:
            print(f"Error: '{metric}' requires a pair of channels separated by ' vs ' (e.g., 'Green vs Red').")
            return
        if metric in ['local_morans_i', 'expression'] and ' vs ' in Channel:
            print(f"Error: '{metric}' requires a single channel (e.g., 'Green'), not '{Channel}'.")
            return

        P = self.PosLbls[pos]

        # 1. Determine if we are loading 1 frame or ALL frames (movie)
        if str(frame_idx).lower() == 'all':
            time_indices = range(len(self.frames))
            frames_to_load = self.frames
            is_3d = True
            print(f"Preparing Napari movie layer for {Channel} ({metric})...")
        else:
            if frame_idx >= len(self.frames):
                print(f"Error: frame_idx {frame_idx} out of bounds.")
                return
            time_indices = [frame_idx]
            frames_to_load = [self.frames[frame_idx]]
            is_3d = False

        # Base Image Channel
        col_channel = Channel.replace(' vs ', '_vs_')
        img_ch = Channel.split(' vs ')[0] if ' vs ' in Channel else Channel

        # 2. Gather Coordinates and Metrics (ULTRA FAST O(N) LOOP)
        all_points = []
        layer_q_vals = []
        layer_expr = []

        for idx, t in enumerate(time_indices):
            fl = P.framelabels[t]  # <--- THIS BYPASSES THE O(N^2) PERFORMANCE TRAP!
            if fl.num == 0: continue
            
            df = fl.regionprops
            xy = fl.centroid
            
            if is_3d:
                pts = np.pad(xy, ((0, 0), (1, 0)), constant_values=idx)
            else:
                pts = xy
                
            if metric == 'expression':
                mean_col = f'mean_{img_ch}'
                if mean_col in df.columns:
                    vals = df[mean_col].values
                    v_min, v_max = np.percentile(vals, 1), np.percentile(vals, 99)
                    norm_vals = np.clip((vals - v_min) / (v_max - v_min + 1e-9), 0, 1)
                    layer_expr.extend(norm_vals)
                    all_points.append(pts)
            else:
                q_col = f'local_moran_q_{col_channel}' if metric == 'local_morans_i' else f'local_biv_q_{col_channel}'
                p_col = f'local_moran_p_{col_channel}' if metric == 'local_morans_i' else f'local_biv_p_{col_channel}'
                
                if q_col in df.columns and p_col in df.columns:
                    q_vals = df[q_col].values.copy()
                    p_vals = df[p_col].values
                    q_vals[p_vals > 0.05] = 0
                    layer_q_vals.extend(q_vals)
                    all_points.append(pts)

        if not all_points:
            print(f"Error: No spatial stats found for {Channel}. Did you run calculate_spatial_stats()?")
            return

        pointsmat = np.concatenate(all_points)
        
        # 3. Setup Napari Viewer
        viewer = get_or_create_viewer()
        if clear_viewer:
            viewer.layers.clear()
        viewer.scale_bar.unit = "um"

        def get_napari_cmap(ch_name):
            ch_low = str(ch_name).lower()
            if 'far' in ch_low and 'red' in ch_low: return 'magenta' 
            if 'red' in ch_low: return 'red'
            if 'green' in ch_low: return 'green'
            if 'cyan' in ch_low: return 'cyan'
            if 'magenta' in ch_low: return 'magenta'
            if 'yellow' in ch_low: return 'yellow'
            if 'blue' in ch_low or 'dapi' in ch_low: return 'blue'
            return 'gray'

        # Load Underlying Image Layer
        if load_images:
            try:
                stk = P.img(Channel=img_ch, frames=frames_to_load)
                if stk.max() > 0:
                    clim = [np.percentile(stk, 50), np.percentile(stk, 99.9)]
                else:
                    clim = [0, 1]

                layer_name = f"{img_ch} (Image)"
                if layer_name not in viewer.layers:
                    viewer.add_image(
                        stk, name=layer_name, blending="additive", colormap=get_napari_cmap(img_ch),
                        contrast_limits=clim, scale=[1, P.PixelSize, P.PixelSize] if is_3d else [P.PixelSize, P.PixelSize]
                    )
            except Exception as e:
                print(f"Warning: Could not load underlying image for {img_ch}.")

        # 4. Generate Point Layers
        layer_name_pts = f"{metric} ({Channel})"
        
        if metric == 'expression':
            viewer.add_points(
                pointsmat, properties={'expr': layer_expr}, face_color='expr', face_colormap='magma',
                edge_width=0, size=size, name=layer_name_pts,
                scale=[1, P.PixelSize, P.PixelSize] if is_3d else [P.PixelSize, P.PixelSize]
            )
        else:
            lisa_colors = {1: 'red', 2: 'cyan', 3: 'blue', 4: 'orange', 0: 'gray'}
            lisa_rgba = {k: mc.to_rgba(v) for k, v in lisa_colors.items()}
            q_arr = np.array(layer_q_vals)
            rgba_matrix = np.zeros((len(q_arr), 4))
            for q_cat, rgba in lisa_rgba.items():
                rgba_matrix[q_arr == q_cat] = rgba

            viewer.add_points(
                pointsmat, face_color=rgba_matrix,
                edge_width=0, size=size, name=layer_name_pts,
                scale=[1, P.PixelSize, P.PixelSize] if is_3d else [P.PixelSize, P.PixelSize]
            )
            
        print(f"Loaded {layer_name_pts} into Napari.")
        return viewer

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
            if len(r.PosLbls[pos].framelabels) == 0:
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

    def calculate_spatial_stats(self, Position=None, metrics=['morans_i', 'local_morans_i'], channels=None, 
                                bivariate_pairs=None, cluster_key=None, n_neighs=6, radius=None, 
                                nhood_frames=None, export_h5ad=False, save=True, **kwargs):
        """
        Incorporates squidpy/scverse spatial statistics into the oyLabImaging pipeline.
        Calculates global and local spatial autocorrelation, as well as categorical neighborhood enrichment.

        Parameters:
        -----------
        Position : list or str, optional
            Position(s) to analyze (e.g., ['B6-Site_0']). If None, defaults to all positions in the experiment.
        metrics : list of str, optional
            List of spatial metrics to compute. 
            Options: 'morans_i', 'gearys_c', 'bivariate_moran', 'local_morans_i', 'local_bivariate_moran', 'neighborhood_enrichment'.
        channels : list of str, optional
            Specific channels to use for univariate stats. If None, automatically ignores DAPI/DIC and uses the rest.
        bivariate_pairs : list of tuples, optional
            Specific channel pairs to test for Bivariate Moran's I and Local Bivariate Moran's I. 
            Note: Bivariate Moran's is directional. The first element is the focal channel, the second is the spatial lag channel.
            Example: [('Red', 'FarRed')]. If None, auto-generates combinations from `channels`.
        cluster_key : str, optional
            Name of an existing column in `P.framelabels[t].regionprops` containing true categorical cell type labels.
            If provided, enables categorical 'neighborhood_enrichment'.
        n_neighs : int, optional
            Number of spatial neighbors to connect per cell in the spatial graph. Default is 6.
        radius : float, optional
            Distance radius (in microns) to connect cells in the spatial graph. Overrides n_neighs if provided.
        nhood_frames : list of int, optional
            Specific frames to compute Neighborhood Enrichment for (to save time). If None, computes for all frames.
        export_h5ad : bool, optional
            If True, exports the constructed AnnData object for each frame to a .h5ad file in the data folder.
        save : bool, optional
            If True, permanently updates the original PosLbl .pkl files on the hard drive with the new stats.
        """
        
        warnings.filterwarnings("ignore", message=".*numba.*")

        if Position is None:
            Position = list(self.PosNames)
        elif not isinstance(Position, list) and not isinstance(Position, np.ndarray):
            Position = [Position]

        # --- DYNAMIC CHANNEL SELECTION ---
        if channels is None:
            target_channels = [str(ch) for ch in self.channels if 'blue' not in str(ch).lower() 
                               and 'dapi' not in str(ch).lower() and 'cyan' not in str(ch).lower() 
                               and 'dic' not in str(ch).lower()]
            if not target_channels: target_channels = [str(ch) for ch in self.channels]
        else:
            target_channels = [str(ch) for ch in channels if ch in self.channels]

        # Auto-generate bivariate pairs if requested but not provided
        if ('bivariate_moran' in metrics or 'local_bivariate_moran' in metrics) and bivariate_pairs is None:
            bivariate_pairs = list(combinations(target_channels, 2))

        for pos in Position:
            print(f"\nCalculating spatial stats for position: {pos}")
            if pos not in self.PosLbls:
                print(f"  Warning: {pos} not segmented yet. Skipping.")
                continue
                
            P = self.PosLbls[pos]
            if not hasattr(P, 'spatial_stats'): P.spatial_stats = {}
            
            for t, frame in enumerate(self.frames):
                if P.num[t] == 0: continue
                
                print(f"  Processing Frame {frame}...", end="\r")
                frame_str = str(frame)
                if frame_str not in P.spatial_stats: P.spatial_stats[frame_str] = {}
                
                # 1. Extract Data (Force Float64 to prevent Numba typing errors)
                intensity_dict = {}
                for ch in target_channels:
                    intensity_dict[ch] = np.array(P.mean(ch)[t], dtype=np.float64)
                
                df_expr = pd.DataFrame(intensity_dict)
                adata = ad.AnnData(X=df_expr.values.astype(np.float64))
                adata.var_names = list(intensity_dict.keys())
                adata.obs_names = [str(i) for i in range(df_expr.shape[0])]
                adata.obsm['spatial'] = np.array(P.centroid_um[t], dtype=np.float64)
                adata.obs['Area_um2'] = np.array(P.framelabels[t].area_um2, dtype=np.float64)

                active_cluster_key = None
                if cluster_key and cluster_key in P.framelabels[t].regionprops.columns:
                    adata.obs[cluster_key] = pd.Categorical(P.framelabels[t].regionprops[cluster_key])
                    active_cluster_key = cluster_key

                # 2. Build Spatial Graph
                sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=n_neighs, radius=radius)
                
                # Setup spatial weights object for PySAL/ESDA calculations
                conn = adata.obsp['spatial_connectivities']
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    w = W(dict(enumerate(conn.tolil().rows)))

                # 3. Global Univariate Autocorrelation
                if 'morans_i' in metrics:
                    sq.gr.spatial_autocorr(adata, mode='moran', genes=adata.var_names)
                    res_key = 'moranI' if 'moranI' in adata.uns else 'moran'
                    if res_key in adata.uns: P.spatial_stats[frame_str]['morans_i'] = adata.uns[res_key]
                        
                if 'gearys_c' in metrics:
                    sq.gr.spatial_autocorr(adata, mode='geary', genes=adata.var_names)
                    res_key = 'gearyC' if 'gearyC' in adata.uns else 'geary'
                    if res_key in adata.uns: P.spatial_stats[frame_str]['gearys_c'] = adata.uns[res_key]
                
                # 4. Global Bivariate Autocorrelation
                if 'bivariate_moran' in metrics and bivariate_pairs:
                    biv_dict = {}
                    for ch1, ch2 in bivariate_pairs:
                        if ch1 in adata.var_names and ch2 in adata.var_names:
                            x = adata[:, ch1].X.flatten().astype(np.float64)
                            y = adata[:, ch2].X.flatten().astype(np.float64)
                            mbv = Moran_BV(x, y, w)
                            biv_dict[f"{ch1} vs {ch2}"] = mbv.I
                    if biv_dict: P.spatial_stats[frame_str]['bivariate_moran'] = biv_dict

                # 5. Local Univariate Autocorrelation (Local Moran's I per cell)
                if 'local_morans_i' in metrics:
                    for ch in target_channels:
                        y = adata[:, ch].X.flatten().astype(np.float64)
                        ml = Moran_Local(y, w)
                        # ml.q mapping: 1=HH, 2=LH, 3=LL, 4=HL
                        # Append directly into tracking matrix (regionprops)
                        P.framelabels[t].regionprops[f'local_moran_I_{ch}'] = ml.Is
                        P.framelabels[t].regionprops[f'local_moran_p_{ch}'] = ml.p_sim
                        P.framelabels[t].regionprops[f'local_moran_q_{ch}'] = ml.q

                # 6. Local Bivariate Autocorrelation (Per cell)
                if 'local_bivariate_moran' in metrics and bivariate_pairs:
                    for ch1, ch2 in bivariate_pairs:
                        if ch1 in adata.var_names and ch2 in adata.var_names:
                            x = adata[:, ch1].X.flatten().astype(np.float64)
                            y = adata[:, ch2].X.flatten().astype(np.float64)
                            ml_bv = Moran_Local_BV(x, y, w)
                            P.framelabels[t].regionprops[f'local_biv_I_{ch1}_vs_{ch2}'] = ml_bv.Is
                            P.framelabels[t].regionprops[f'local_biv_p_{ch1}_vs_{ch2}'] = ml_bv.p_sim
                            P.framelabels[t].regionprops[f'local_biv_q_{ch1}_vs_{ch2}'] = ml_bv.q

                # 7. Categorical Neighborhood Enrichment
                if 'neighborhood_enrichment' in metrics and active_cluster_key:
                    if nhood_frames is None or frame in nhood_frames:
                        if len(adata.obs[active_cluster_key].unique()) > 1:
                            sq.gr.nhood_enrichment(adata, cluster_key=active_cluster_key)
                            P.spatial_stats[frame_str]['neighborhood_enrichment'] = adata.uns[f'{active_cluster_key}_nhood_enrichment']['zscore']
                            P.spatial_stats[frame_str]['neighborhood_categories'] = list(adata.obs[active_cluster_key].cat.categories)

                # 8. Export h5ad
                if export_h5ad:
                    export_dir = os.path.join(self.pth, "AnnData")
                    os.makedirs(export_dir, exist_ok=True)
                    adata.write(os.path.join(export_dir, f"{pos}_frame{frame}.h5ad"))
            
            print(f"  ✓ Finished {pos}                        ")
            if save: P.save()

    def plot_spatial_stats(self, Position, metric='morans_i', channels=None, nhood_pairs=None, custom_colors=None, plot_type='auto', frame_idx=0):
        """
        Plots spatial statistic scores dynamically.

        Parameters:
        -----------
        Position : list or str
            Position(s) to analyze (e.g., ['B6-Site_0']).
        metric : str, optional
            Metric to plot. Options: 'morans_i', 'gearys_c', 'bivariate_moran', 'local_morans_i', 'local_bivariate_moran', 'neighborhood_enrichment', 'expression'.
        channels : list of str, optional
            Specific channels to plot. If None, plots all available non-DAPI channels.
        nhood_pairs : list of tuples, optional
            Specific pairs to plot for neighborhood enrichment. Example: [('Tumor', 'Immune')].
        custom_colors : dict, optional
            Dictionary mapping a channel or interaction pair to a specific color (hex or string).
        plot_type : str, optional
            'auto', 'time_series', 'summary' (bar/line), or 'spatial_map'.
        frame_idx : int, optional
            If plot_type is 'spatial_map', specifies which timepoint frame to draw the map for.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if not isinstance(Position, list) and not isinstance(Position, np.ndarray):
            Position = [Position]

        if channels is None:
            plot_channels = [str(ch) for ch in self.channels if 'blue' not in str(ch).lower() 
                             and 'dapi' not in str(ch).lower() and 'dic' not in str(ch).lower()]
            if not plot_channels: plot_channels = [str(ch) for ch in self.channels]
        else:
            plot_channels = [str(ch) for ch in channels if ch in self.channels]

        # Smart Color Matcher
        def get_color(ch, idx, total_colors):
            ch_low = str(ch).lower()
            if 'far' in ch_low and 'red' in ch_low: return '#800080' 
            if 'red' in ch_low: return '#d62728'
            if 'green' in ch_low: return '#2ca25f'
            if 'cyan' in ch_low: return '#17becf'
            if 'magenta' in ch_low: return '#e377c2'
            if 'yellow' in ch_low: return '#bcbd22'
            if 'blue' in ch_low: return '#1f77b4'
            palette = sns.color_palette("husl", max(8, total_colors))
            return palette[idx]

        try: plt.style.use('seaborn-v0_8-whitegrid')
        except: sns.set_style("whitegrid")

        valid_positions = [p for p in Position if p in self.PosLbls]
        if not valid_positions: 
            print("No valid spatial stats data found for plotting.")
            return

        # ________________________________________
        # SECTION 1: LOCAL METRICS & EXPRESSION
        # ________________________________________

        if metric in ['local_morans_i', 'local_bivariate_moran', 'expression']:
            
            # --- 1. Spatial X/Y Map Visualization ---
            if plot_type == 'spatial_map' or (plot_type == 'auto' and len(self.frames) == 1):
                t = frame_idx if frame_idx < len(self.frames) else 0
                
                if metric in ['local_morans_i', 'expression']:
                    targets = plot_channels
                else:
                    targets = []
                    valid_P = self.PosLbls[valid_positions[0]]
                    if valid_P.num[t] > 0:
                        cols = valid_P.framelabels[t].regionprops.columns
                        targets = [c.replace('local_biv_q_', '') for c in cols if c.startswith('local_biv_q_')]
                
                if not targets: return
                
                fig, axes = plt.subplots(len(targets), len(valid_positions), 
                                         figsize=(6 * len(valid_positions), 6 * len(targets)), squeeze=False)
                
                lisa_colors = {1: '#d7191c', 2: '#abd9e9', 3: '#2c7bb6', 4: '#fdae61', 0: '#e0e0e0'}
                lisa_labels = {1: 'High-High', 2: 'Low-High', 3: 'Low-Low', 4: 'High-Low', 0: 'Not Significant'}
                
                for j, target in enumerate(targets):
                    for i, pos in enumerate(valid_positions):
                        ax = axes[j, i]
                        P = self.PosLbls[pos]
                        
                        if P.num[t] == 0:
                            ax.axis('off')
                            continue
                            
                        df = P.framelabels[t].regionprops
                        xy = P.centroid_um[t]
                        
                        if metric == 'expression':
                            mean_col = f'mean_{target}'
                            if mean_col in df.columns:
                                vals = df[mean_col].values
                                v_min, v_max = np.percentile(vals, 1), np.percentile(vals, 99)
                                norm_vals = np.clip((vals - v_min) / (v_max - v_min + 1e-9), 0, 1)
                                sc = ax.scatter(xy[:, 0], xy[:, 1], c=norm_vals, cmap='magma', s=10, alpha=0.9)
                                if i == len(valid_positions) - 1:
                                    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Norm Expression")
                        else:
                            q_col = f'local_moran_q_{target}' if metric == 'local_morans_i' else f'local_biv_q_{target}'
                            p_col = f'local_moran_p_{target}' if metric == 'local_morans_i' else f'local_biv_p_{target}'
                                
                            if q_col in df.columns and p_col in df.columns:
                                q_vals = df[q_col].values
                                p_vals = df[p_col].values
                                plot_q = q_vals.copy()
                                plot_q[p_vals > 0.05] = 0
                                
                                for q_cat in [0, 3, 2, 4, 1]: 
                                    mask = (plot_q == q_cat)
                                    if mask.sum() > 0:
                                        ax.scatter(xy[mask, 0], xy[mask, 1], c=lisa_colors[q_cat], 
                                                   s=5 if q_cat==0 else 15, 
                                                   alpha=0.6 if q_cat==0 else 1.0, 
                                                   label=lisa_labels[q_cat] if (i==0 and j==0) else "")
                        
                        ax.invert_yaxis() 
                        ax.set_aspect('equal')
                        ax.set_xticks([])
                        ax.set_yticks([])
                        if j == 0: ax.set_title(f"Position: {pos}", fontsize=14, fontweight='bold')
                        if i == 0: 
                            label_name = target.replace('_vs_', ' \u2194 ') if metric == 'local_bivariate_moran' else target
                            ax.set_ylabel(f"{label_name}", fontsize=14, fontweight='bold')
                
                if metric != 'expression':
                    handles, labels = axes[0, 0].get_legend_handles_labels()
                    if handles:
                        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
                                   ncol=5, frameon=True, edgecolor='black', fontsize=12)
                
                title_map = {'local_bivariate_moran': "Local Bivariate", 'local_morans_i': "Local", 'expression': "Expression"}
                plt.suptitle(f"{title_map[metric]} Spatial Map (Frame {self.frames[t]})", fontsize=16, fontweight='bold', y=1.02)
                plt.tight_layout()
                plt.show()
                plt.close(fig)
                return

            # --- 2. Summary / Time-Series Visualization ---
            elif plot_type in ['summary', 'time_series'] or (plot_type == 'auto' and len(self.frames) > 1):
                
                if metric in ['local_morans_i', 'expression']:
                    targets = plot_channels
                else:
                    targets = []
                    valid_P = self.PosLbls[valid_positions[0]]
                    if valid_P.num[0] > 0:
                        cols = valid_P.framelabels[0].regionprops.columns
                        targets = [c.replace('local_biv_q_', '') for c in cols if c.startswith('local_biv_q_')]
                
                if not targets: return

                if len(self.frames) == 1:
                    # GROUPED BAR CHART FOR 1 TIMEPOINT
                    fig_width = max(4.0, len(valid_positions) * 2.5)
                    fig, ax = plt.subplots(figsize=(fig_width, 6))
                    bar_width = 0.8 / len(targets)
                    x_indices = np.arange(len(valid_positions))
                    max_y_val = 0
                    
                    for j, target in enumerate(targets):
                        scores = []
                        for pos in valid_positions:
                            P = self.PosLbls[pos]
                            if P.num[0] == 0: 
                                scores.append(0)
                                continue
                            df = P.framelabels[0].regionprops
                            
                            if metric == 'expression':
                                mean_col = f'mean_{target}'
                                scores.append(df[mean_col].mean() if mean_col in df.columns else 0)
                                if scores[-1] > max_y_val: max_y_val = scores[-1]
                            else:
                                q_col = f'local_moran_q_{target}' if metric == 'local_morans_i' else f'local_biv_q_{target}'
                                p_col = f'local_moran_p_{target}' if metric == 'local_morans_i' else f'local_biv_p_{target}'
                                    
                                if q_col in df.columns and p_col in df.columns:
                                    hh_mask = (df[q_col] == 1) & (df[p_col] <= 0.05)
                                    pct_hh = (hh_mask.sum() / P.num[0]) * 100
                                    scores.append(pct_hh)
                                    if pct_hh > max_y_val: max_y_val = pct_hh
                                else:
                                    scores.append(0)
                                
                        c = custom_colors[target] if custom_colors and target in custom_colors else get_color(target, j, len(targets))
                        label_name = target.replace('_vs_', ' \u2194 ') if metric == 'local_bivariate_moran' else target
                        
                        offset = (j - len(targets)/2 + 0.5) * bar_width
                        bars = ax.bar(x_indices + offset, scores, bar_width, label=label_name, color=c, edgecolor='black')
                        
                        for bar in bars:
                            y = bar.get_height()
                            # FIXED: Tiny threshold so raw intensities aren't skipped
                            if y > 0.000001: 
                                # FIXED: 4 decimal places for expression
                                label_fmt = f'{y:.4f}' if metric == 'expression' else f'{y:.1f}%'
                                # FIXED: Dynamic offset so text hugs the bar and never floats away
                                text_y = y + (max_y_val * 0.03)
                                ax.text(bar.get_x() + bar.get_width()/2, text_y, label_fmt, ha='center', va='bottom', fontweight='bold', fontsize=9)

                    # Dynamic headroom
                    ax.set_ylim(0, max(max_y_val * 1.20, 5) if metric != 'expression' else max_y_val * 1.20) 
                    ax.set_xticks(x_indices)
                    ax.set_xticklabels(valid_positions, fontweight='bold', fontsize=12)
                    title_map = {'local_bivariate_moran': "Local Bivariate", 'local_morans_i': "Local", 'expression': "Expression"}
                    y_label = "Mean Intensity" if metric == 'expression' else "% of Cells in High-High"
                    ax.set_title(f"{title_map[metric]} Summary", fontweight='bold', fontsize=15, pad=15)
                    ax.set_ylabel(y_label, fontweight='bold', fontsize=12)
                    ax.xaxis.grid(False)
                    sns.despine(bottom=True)
                    plt.legend(title="Marker / Pair", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
                    plt.tight_layout()
                    plt.show()
                    plt.close(fig)
                    return

                else:
                    # LINE PLOT FOR >1 TIMEPOINTS
                    fig, axes = plt.subplots(1, len(valid_positions), figsize=(max(6, len(valid_positions)*6), 5), squeeze=False)
                    axes = axes.flatten()
                    
                    for i, pos in enumerate(valid_positions):
                        ax = axes[i]
                        P = self.PosLbls[pos]
                        
                        for j, target in enumerate(targets):
                            frames_plotted, values = [], []
                            
                            for t, frame in enumerate(self.frames):
                                if P.num[t] == 0: continue
                                df = P.framelabels[t].regionprops
                                
                                if metric == 'expression':
                                    mean_col = f'mean_{target}'
                                    if mean_col in df.columns:
                                        frames_plotted.append(frame)
                                        values.append(df[mean_col].mean())
                                else:
                                    q_col = f'local_moran_q_{target}' if metric == 'local_morans_i' else f'local_biv_q_{target}'
                                    p_col = f'local_moran_p_{target}' if metric == 'local_morans_i' else f'local_biv_p_{target}'
                                        
                                    if q_col in df.columns and p_col in df.columns:
                                        hh_mask = (df[q_col] == 1) & (df[p_col] <= 0.05)
                                        pct_hh = (hh_mask.sum() / P.num[t]) * 100
                                        frames_plotted.append(frame)
                                        values.append(pct_hh)
                                    
                            if frames_plotted:
                                c = custom_colors[target] if custom_colors and target in custom_colors else get_color(target, j, len(targets))
                                label_name = target.replace('_vs_', ' \u2194 ') if metric == 'local_bivariate_moran' else target
                                ax.plot(frames_plotted, values, marker='o', markersize=5, linestyle='-', linewidth=2.0, label=label_name, color=c, markeredgecolor='white', markeredgewidth=1)
                        
                        ax.set_ylim(bottom=0)
                        ax.set_xlabel("Timepoint (Frame)", fontsize=12, fontweight='bold')
                        ax.set_title(f"Position: {pos}", fontsize=14, fontweight='bold')
                        ax.xaxis.grid(False)
                        y_label = "Mean Intensity" if metric == 'expression' else "% of Cells in High-High"
                        if i == 0: ax.set_ylabel(y_label, fontsize=12, fontweight='bold')

                    handles, labels = axes[-1].get_legend_handles_labels()
                    if handles: 
                        fig.legend(handles, labels, title='Marker / Pair' if metric == 'local_bivariate_moran' else 'Marker', loc='lower center', 
                                   bbox_to_anchor=(0.5, -0.12), ncol=min(4, len(targets)), frameon=True, edgecolor='black')
                    
                    title_map = {'local_bivariate_moran': "Local Bivariate", 'local_morans_i': "Local", 'expression': "Expression"}
                    plt.suptitle(f"{title_map[metric]} Dynamics", fontsize=16, fontweight='bold', y=1.02)
                    sns.despine()
                    plt.tight_layout()
                    fig.subplots_adjust(bottom=0.20)
                    plt.show()
                    plt.close(fig)
                    return

        frame_str = str(self.frames[0])

        # _____________________________________________________
        # SECTION 2: GLOBAL BIVARIATE SPATIAL AUTOCORRELATION
        # _____________________________________________________

        if metric == 'bivariate_moran':
            if len(self.frames) == 1:
                pairs = list(self.PosLbls[valid_positions[0]].spatial_stats[frame_str].get(metric, {}).keys())
                if not pairs: return
                fig_width = max(5.0, len(valid_positions) * len(pairs) * 1.5)
                fig, ax = plt.subplots(figsize=(fig_width, 6))
                bar_width = 0.8 / len(pairs)
                x_indices = np.arange(len(valid_positions))
                
                default_palette = sns.color_palette("Set2", len(pairs))
                pair_colors = [custom_colors[p] if custom_colors and p in custom_colors else default_palette[i] for i, p in enumerate(pairs)]
                
                for i, pair in enumerate(pairs):
                    scores = [self.PosLbls[pos].spatial_stats[frame_str][metric].get(pair, 0) for pos in valid_positions]
                    offset = (i - len(pairs)/2 + 0.5) * bar_width
                    bars = ax.bar(x_indices + offset, scores, bar_width, label=pair, color=pair_colors[i], edgecolor='black')
                    for bar in bars: 
                        y = bar.get_height()
                        if abs(y) > 0.001: ax.text(bar.get_x() + bar.get_width()/2, y + (0.01 if y>0 else -0.02), f'{y:.3f}', ha='center', va='bottom' if y>0 else 'top', fontweight='bold', fontsize=9)
                
                ax.axhline(0, color='black', linewidth=1)
                ax.set_xticks(x_indices)
                ax.set_xticklabels(valid_positions, fontweight='bold', fontsize=12)
                ax.set_title("Bivariate Spatial Autocorrelation", fontweight='bold', fontsize=15)
                ax.set_ylabel("Bivariate Moran's I", fontweight='bold', fontsize=12)
                ax.xaxis.grid(False)
                sns.despine(bottom=True)
                plt.legend(title="Channel Pairs", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
                plt.tight_layout()
                plt.show()
                plt.close(fig)
            else:
                # Time-series Bivariate
                fig, axes = plt.subplots(1, len(valid_positions), figsize=(max(6, len(valid_positions)*6), 5), squeeze=False)
                axes = axes.flatten()
                
                pairs = []
                for f in self.frames:
                    if str(f) in self.PosLbls[valid_positions[0]].spatial_stats:
                        pairs = list(self.PosLbls[valid_positions[0]].spatial_stats[str(f)].get(metric, {}).keys())
                        if pairs: break
                
                default_palette = sns.color_palette("Set2", len(pairs))
                pair_colors = [custom_colors[p] if custom_colors and p in custom_colors else default_palette[i] for i, p in enumerate(pairs)]
                
                for i, pos in enumerate(valid_positions):
                    ax = axes[i]
                    P = self.PosLbls[pos]
                    
                    for j, pair in enumerate(pairs):
                        frames_plotted, scores = [], []
                        for frame in self.frames:
                            f_str = str(frame)
                            if f_str in P.spatial_stats and metric in P.spatial_stats[f_str]:
                                stat_dict = P.spatial_stats[f_str][metric]
                                if pair in stat_dict:
                                    frames_plotted.append(frame)
                                    scores.append(stat_dict[pair])
                        
                        if frames_plotted:
                            ax.plot(frames_plotted, scores, marker='o', markersize=5, linestyle='-', linewidth=2.0, label=pair, color=pair_colors[j], markeredgecolor='white', markeredgewidth=1)
                    
                    ax.axhline(0, color='black', linewidth=1.5, linestyle='--')
                    ax.set_xlabel("Timepoint (Frame)", fontsize=12, fontweight='bold')
                    ax.set_title(f"Position: {pos}", fontsize=14, fontweight='bold')
                    ax.xaxis.grid(False)
                    if i == 0: ax.set_ylabel("Bivariate Moran's I", fontsize=12, fontweight='bold')
                
                handles, labels = axes[-1].get_legend_handles_labels()
                if handles: 
                    fig.legend(handles, labels, title='Channel Pairs', loc='lower center', 
                               bbox_to_anchor=(0.5, -0.12), ncol=len(pairs), frameon=True, edgecolor='black')
                plt.suptitle("Bivariate Spatial Autocorrelation Over Time", fontsize=16, fontweight='bold', y=1.02)
                sns.despine()
                plt.tight_layout()
                fig.subplots_adjust(bottom=0.20)
                plt.show()
                plt.close(fig)

        # ________________________________________________
        # SECTION 3: CATEGORICAL NEIGHBORHOOD ENRICHMENT
        # ________________________________________________

        elif metric == 'neighborhood_enrichment':
            valid_frames = [f for f in self.frames if str(f) in self.PosLbls[valid_positions[0]].spatial_stats and metric in self.PosLbls[valid_positions[0]].spatial_stats[str(f)]]
            if not valid_frames:
                print("No Neighborhood Enrichment data found to plot. Ensure you provided a 'cluster_key' during calculation.")
                return
            
            all_cats = set()
            for f in valid_frames:
                cats = self.PosLbls[valid_positions[0]].spatial_stats[str(f)].get('neighborhood_categories', [])
                all_cats.update(cats)
            cat_names_global = sorted(list(all_cats))

            if nhood_pairs is not None:
                pairs = nhood_pairs
            else:
                pairs = [(cat_names_global[i], cat_names_global[j]) for i in range(len(cat_names_global)) for j in range(i, len(cat_names_global))]
                if len(pairs) > 6:
                    print(f"Warning: Plotting {len(pairs)} interaction lines. Consider using `nhood_pairs` argument.")
            
            default_palette = sns.color_palette("Set2", len(pairs))
            pair_colors = [custom_colors[p] if custom_colors and p in custom_colors else default_palette[i] for i, p in enumerate(pairs)]
            
            if len(self.frames) == 1:
                first_valid_str = str(valid_frames[0])
                fig_width = max(5.0, len(valid_positions) * len(pairs) * 1.5)
                fig, ax = plt.subplots(figsize=(fig_width, 6))
                bar_width = 0.8 / len(pairs)
                x_indices = np.arange(len(valid_positions))
                
                # Significance Lines for Bar Chart
                ax.axhline(1.96, color='gray', linestyle='--', linewidth=1.5, zorder=0, label='Sig. Threshold (p<0.05)')
                ax.axhline(-1.96, color='gray', linestyle='--', linewidth=1.5, zorder=0)
                
                for k, (c1, c2) in enumerate(pairs):
                    scores = []
                    for pos in valid_positions:
                        z_matrix = self.PosLbls[pos].spatial_stats[first_valid_str][metric]
                        curr_cats = self.PosLbls[pos].spatial_stats[first_valid_str].get('neighborhood_categories', [])
                        if c1 in curr_cats and c2 in curr_cats:
                            i_idx, j_idx = curr_cats.index(c1), curr_cats.index(c2)
                            scores.append(z_matrix[i_idx, j_idx])
                        else:
                            scores.append(0)
                    
                    label = f"{c1} - {c2}"
                    offset = (k - len(pairs)/2 + 0.5) * bar_width
                    bars = ax.bar(x_indices + offset, scores, bar_width, label=label, color=pair_colors[k], edgecolor='black', zorder=3)
                    for bar in bars: 
                        y = bar.get_height()
                        if abs(y) > 0.001: ax.text(bar.get_x() + bar.get_width()/2, y + (0.5 if y>0 else -1.5), f'{y:.1f}', ha='center', va='bottom' if y>0 else 'top', fontweight='bold', fontsize=9)
                
                ax.axhline(0, color='black', linewidth=1)
                ax.set_xticks(x_indices)
                ax.set_xticklabels(valid_positions, fontweight='bold', fontsize=12)
                ax.set_title("Neighborhood Enrichment (Z-Score)", fontweight='bold', fontsize=15)
                ax.set_ylabel("Enrichment Z-Score", fontweight='bold', fontsize=12)
                ax.xaxis.grid(False)
                sns.despine(bottom=True)
                plt.legend(title="Interaction Type", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
                plt.tight_layout()
                plt.show()
                plt.close(fig)

            else:
                # Time-series Line Graphs
                fig, axes = plt.subplots(1, len(valid_positions), figsize=(max(6, len(valid_positions)*6), 5), squeeze=False)
                axes = axes.flatten()
                
                for p_idx, pos in enumerate(valid_positions):
                    ax = axes[p_idx]
                    P = self.PosLbls[pos]
                    
                    # Add Significance Threshold Lines
                    ax.axhline(1.96, color='gray', linestyle='--', linewidth=1.5, zorder=0)
                    ax.axhline(-1.96, color='gray', linestyle='--', linewidth=1.5, zorder=0)
                    if p_idx == 0:
                        ax.plot([], [], color='gray', linestyle='--', linewidth=1.5, label='Sig. Threshold (p<0.05)')

                    for k, (c1, c2) in enumerate(pairs):
                        frames_plotted = valid_frames 
                        scores = []
                        
                        for frame in valid_frames:
                            f_str = str(frame)
                            appended = False
                            if f_str in P.spatial_stats and metric in P.spatial_stats[f_str]:
                                z_matrix = P.spatial_stats[f_str][metric]
                                curr_cats = P.spatial_stats[f_str].get('neighborhood_categories', [])
                                
                                if c1 in curr_cats and c2 in curr_cats:
                                    i_idx, j_idx = curr_cats.index(c1), curr_cats.index(c2)
                                    scores.append(z_matrix[i_idx, j_idx])
                                    appended = True
                            
                            if not appended:
                                scores.append(np.nan)
                        
                        if not np.all(np.isnan(scores)):
                            label = f"{c1} \u2194 {c2}"
                            ax.plot(frames_plotted, scores, marker='o', markersize=5, linestyle='-', linewidth=2.0, label=label, color=pair_colors[k], markeredgecolor='white', markeredgewidth=1)
                    
                    ax.axhline(0, color='black', linewidth=1.5, linestyle='-')
                    ax.set_xlabel("Timepoint (Frame)", fontsize=12, fontweight='bold')
                    ax.set_title(f"Position: {pos}", fontsize=14, fontweight='bold')
                    ax.xaxis.grid(False)
                    if p_idx == 0: ax.set_ylabel("Enrichment Z-Score", fontsize=12, fontweight='bold')
                
                handles, labels = axes[-1].get_legend_handles_labels()
                if handles: 
                    if 'Sig. Threshold (p<0.05)' in labels:
                        idx = labels.index('Sig. Threshold (p<0.05)')
                        handles.append(handles.pop(idx))
                        labels.append(labels.pop(idx))
                    fig.legend(handles, labels, title='Interaction Type', loc='lower center', 
                               bbox_to_anchor=(0.5, -0.12), ncol=min(4, len(pairs)+1), frameon=True, edgecolor='black')
                
                plt.suptitle("Neighborhood Enrichment Over Time", fontsize=16, fontweight='bold', y=1.02)
                sns.despine()
                plt.tight_layout()
                fig.subplots_adjust(bottom=0.20)
                plt.show()
                plt.close(fig)

        # ______________________________________________________________
        # SECTION 4: GLOBAL UNIVARIATE AUTOCORRELATION (Moran / Geary)
        # ______________________________________________________________

        else:
            stat_col = 'C' if metric == 'gearys_c' else 'I'
            
            if len(self.frames) == 1:
                frame_str = str(self.frames[0])
                fig_width = max(4.0, len(valid_positions) * 2.5)
                fig, ax = plt.subplots(figsize=(fig_width, 6))
                bar_width = 0.8 / len(plot_channels)
                x_indices = np.arange(len(valid_positions))
                
                # Pre-calculate global min/max for dynamic text offsets and y-limits
                all_scores = []
                for ch in plot_channels:
                    scores = [self.PosLbls[pos].spatial_stats[frame_str].get(metric).loc[ch, stat_col] if (self.PosLbls[pos].spatial_stats[frame_str].get(metric) is not None and ch in self.PosLbls[pos].spatial_stats[frame_str].get(metric).index) else 0 for pos in valid_positions]
                    all_scores.extend(scores)
                    
                max_y_val = max(all_scores) if all_scores else 0
                min_y_val = min(all_scores) if all_scores else 0
                y_range = max_y_val - min_y_val if (max_y_val - min_y_val) > 0 else 0.1
                offset_y = y_range * 0.03
                
                for i, ch in enumerate(plot_channels):
                    scores = [self.PosLbls[pos].spatial_stats[frame_str].get(metric).loc[ch, stat_col] if (self.PosLbls[pos].spatial_stats[frame_str].get(metric) is not None and ch in self.PosLbls[pos].spatial_stats[frame_str].get(metric).index) else 0 for pos in valid_positions]
                    offset = (i - len(plot_channels)/2 + 0.5) * bar_width
                    
                    c = custom_colors[ch] if custom_colors and ch in custom_colors else get_color(ch, i, len(plot_channels))
                    bars = ax.bar(x_indices + offset, scores, bar_width, label=ch, color=c, edgecolor='black')
                    
                    for bar in bars:
                        y = bar.get_height()
                        if abs(y) > 0.001: 
                            ax.text(bar.get_x() + bar.get_width()/2, y + (offset_y if y>0 else -offset_y - 0.02), f'{y:.3f}', ha='center', va='bottom' if y>0 else 'top', fontweight='bold', fontsize=9)
                
                # Plot the baseline (0 for Moran, 1 for Geary)
                baseline = 1 if metric == 'gearys_c' else 0
                ax.axhline(baseline, color='black', linewidth=1.5, linestyle='--')
                
                # Dynamic Y-Limits: Fixes the 'squashed' look for Geary's C
                bot_lim = min(min_y_val - y_range * 0.1, 0)
                if metric == 'gearys_c' and max_y_val < 0.9:
                    top_lim = max_y_val + y_range * 0.25 # Zooms in nicely on the bars!
                else:
                    top_lim = max(max_y_val + y_range * 0.20, baseline + y_range * 0.15)
                ax.set_ylim(bot_lim, top_lim)
                
                ax.set_xticks(x_indices)
                ax.set_xticklabels(valid_positions, fontweight='bold', fontsize=12)
                ax.set_ylabel(f"{metric.replace('_', ' ').title()} Score", fontweight='bold', fontsize=12)
                ax.set_title(f"Spatial Autocorrelation ({metric.replace('_', ' ').title()})", fontweight='bold', pad=15)
                ax.xaxis.grid(False)
                sns.despine(bottom=True)
                plt.legend(title="Channel", bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.show()
                plt.close(fig)

            else:
                # LINE PLOT FOR >1 TIMEPOINTS
                fig, axes = plt.subplots(1, len(valid_positions), figsize=(max(6, len(valid_positions)*6), 5), squeeze=False)
                axes = axes.flatten()
                
                for i, pos in enumerate(valid_positions):
                    ax = axes[i]
                    P = self.PosLbls[pos]
                    
                    for j, ch in enumerate(plot_channels):
                        frames_plotted, scores = [], []
                        for frame in self.frames:
                            f_str = str(frame)
                            if f_str in P.spatial_stats and metric in P.spatial_stats[f_str]:
                                stat_df = P.spatial_stats[f_str][metric]
                                if ch in stat_df.index:
                                    frames_plotted.append(frame)
                                    scores.append(stat_df.loc[ch, stat_col])
                        
                        if frames_plotted:
                            c = custom_colors[ch] if custom_colors and ch in custom_colors else get_color(ch, j, len(plot_channels))
                            ax.plot(frames_plotted, scores, marker='o', markersize=4, linestyle='-', linewidth=2.0, label=ch, color=c, markeredgecolor='white', markeredgewidth=0.5)
                    
                    ax.axhline(1 if metric == 'gearys_c' else 0, color='black', linewidth=1.5, linestyle='--')
                    ax.set_xlabel("Timepoint (Frame)", fontsize=12, fontweight='bold')
                    ax.set_title(f"Position: {pos}", fontsize=14, fontweight='bold')
                    ax.xaxis.grid(False)
                    if i == 0: ax.set_ylabel(f"{metric.replace('_', ' ').title()} Score", fontsize=12, fontweight='bold')
                    
                handles, labels = axes[-1].get_legend_handles_labels()
                if handles: 
                    fig.legend(handles, labels, title='Channel', loc='lower center', 
                               bbox_to_anchor=(0.5, -0.12), ncol=len(plot_channels), frameon=True, edgecolor='black')
                plt.suptitle(f"Spatial Clustering Over Time ({metric.replace('_', ' ').title()})", fontsize=16, fontweight='bold', y=1.02)
                sns.despine()
                plt.tight_layout()
                fig.subplots_adjust(bottom=0.20)
                plt.show()
                plt.close(fig)

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