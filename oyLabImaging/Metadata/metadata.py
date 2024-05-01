# Metadata module for microscopy data
# Currently supports Wollman Lab Scopes data, Micromanager MDA, and nikon ND2 files
# AOY

import logging
import os
import sys
import warnings
from ast import literal_eval
from os import listdir, path, walk
from os.path import join

import dill as pickle
import numpy as np
import pandas as pd
from natsort import natsort_keygen, natsorted
from PIL import Image
from skimage import io
from oyLabImaging.Processing.generalutils import alias

md_logger = logging.getLogger(__name__)
md_logger.setLevel(logging.DEBUG)

usecolslist = [
    "acq",
    "Position",
    "frame",
    "Channel",
    "Marker",
    "Fluorophore",
    "group",
    "XY",
    "Z",
    "Zindex",
    "Exposure",
    "PixelSize",
    "PlateType",
    "TimestampFrame",
    "TimestampImage",
    "filename",
]

# Metadata stores all relevant info about an imaging experiment.


class Metadata(object):
    """
    General class for opening image metadata and for loading images.
    Parameters
    ----------
    pth : str folder where metadata is stored
    verbose : boolean

    Implemented MD types
    --------------------
    MicroManager metadata.txt
    Wollman lab Scope class Metadata.txt
    Nikon *.nd2

    Class methods
    -------------
    'CalculateDriftCorrection',
    'append',
    'pickle',
    'stkread',
    'unique',
    'unpickle'

    Class properties
    ----------------
    'Zindexes',
    'acqnames',
    'base_pth',
    'channels',
    'frames',
    'groups',
    'image_table',
    'posnames',
    'type',

    """

    def __init__(self, pth="", load_type="local", verbose=True):
        # get the base path (directory where it it) to the metadata file.
        # If full path to file is given, separate out the directory
        self._md_name = ""

        if path.isdir(pth):
            self.base_pth = pth
        else:
            self.base_pth, self._md_name = path.split(pth)

        # Init an empty md table
        self.image_table = pd.DataFrame(columns=usecolslist)

        # Determine which type of metadata we're working with
        self.type = self._determine_metadata_type(pth)
        self._md_name = self._determine_metadata_name(pth)

        # If it can't find a supported MD, it exits w/o doing anything
        if self.type == None:
            print("Could not find supported metadata.")
            return

        # How should metadata be read?
        if self.type == "PICKLE":
            self._load_method = self.unpickle
        elif self.type == "TXT":
            self._load_method = self._load_metadata_txt
        elif self.type == "MM":
            self._load_method = self._load_metadata_MM
        elif self.type == "ND2":
            self._load_method = self._load_metadata_nd
        elif self.type == "TIFFS":
            self._load_method = self._load_metadata_TIF_GUI

        # With all we've learned, we can now load the metadata
        try:
            self._load_metadata(verbose=verbose)
        except Exception as e:
            print("could not load metadata, file may be corrupted.")

        # Handle columns that don't import from text well
        try:
            self._convert_data("XY", float)
        except Exception as e:
            self.image_table["XY"] = [literal_eval(i) for i in self.image_table["XY"]]

        # How should files be read?
        if self.type == "ND2":
            self._open_file = self._read_nd2
        elif self.type == "OME":
            self._open_file = self._read_ome
        else:  # Tiffs
            if load_type == "local":
                self._open_file = self._read_local
            elif load_type == "google_cloud":
                raise NotImplementedError("google_cloud loading is not implemented.")
        if np.all(self.unique("Position") == "Default"):
            self.image_table["Position"] = self.image_table["acq"]

    def __call__(self):
        return self.image_table

    # keeping some multiples for backwards compatability
    @property
    def posnames(self):
        """
        Returns all unique position names
        """
        return natsorted(self().Position.unique())

    @property
    def Position(self):
        """
        Returns all unique position names
        """
        return natsorted(self().Position.unique())

    @property
    def frames(self):
        """
        Returns all unique frames
        """
        return list(self().frame.unique())

    @property
    def frame(self):
        """
        Returns all unique frames
        """
        return list(self().frame.unique())

    @property
    def channels(self):
        """
        Returns all unique channel names
        """
        return self().Channel.unique()

    @property
    def Channel(self):
        """
        Returns all unique channel names
        """
        return self().Channel.unique()

    @property
    def Zindexes(self):
        """
        Returns all unique Zindexes
        """
        return self().Zindex.unique()

    @property
    def acq(self):
        """
        Returns all unique acquisition names
        """
        return self().acq.unique()

    @property
    def acqnames(self):
        """
        Returns all unique acquisition names
        """
        return self().acq.unique()

    @property
    def groups(self):
        """
        Returns all unique group names
        """
        return self().group.unique()

    @alias(
        {
            "pos": "Position",
            "Pos": "Position",
            "position": "Position",
            "p": "Position",
            "frames": "frame",
            "Frame": "frame",
            "f": "frame",
            "channel": "Channel",
            "ch": "Channel",
            "c": "Channel",
        }
    )
    def unique(self, Attr=None, sortby="TimestampFrame", **kwargs):
        """
        Parameters
        ----------
        Attr : The desired attribute
        kwargs : Property Value pairs to subset images (see below)

        Returns
        -------
        list of unique values that Attr can have in the subset of images defined by kwargs.
        Default Attr is image size.

        Implemented kwargs
        ------------------
        Position : str, list(str)
        Channel : str, list(str)
        Zindex : int, list(int)
        frames : int, list(int)
        acq : str, list(str)
        """
        # pd.set_option('mode.chained_assignment', None)

        for key, value in kwargs.items():
            if not isinstance(value, (list, np.ndarray)):
                kwargs[key] = [value]
        image_subset_table = self.image_table.copy()
        # Filter images according to some criteria
        for attr in image_subset_table.columns:
            if attr in kwargs:
                image_subset_table = image_subset_table[
                    image_subset_table[attr].isin(kwargs[attr])
                ]
        image_subset_table.sort_values(sortby, inplace=True)

        if Attr is None:
            return image_subset_table.size
        elif Attr in image_subset_table.columns:
            return image_subset_table[Attr].unique()
        elif Attr == "index":
            return image_subset_table.index
        else:
            return None

    def _convert_data(self, column, dtype, isnan=np.nan):
        """
        Helper function to convert text lists to lists of numbers
        """
        converted = []
        arr = self.image_table[column].values
        for i in arr:
            if isinstance(i, str):
                i = np.array(list(map(dtype, i.split())))
                converted.append(i)
            else:
                converted.append(i)
        self.image_table[column] = converted

    def _determine_metadata_type(self, pth):
        """
        Helper function to determine metadata type
        Returns
        -------
        str - MD type
        """
        import os
        from os import path, walk

        fname, fext = path.splitext(pth)
        if path.isdir(pth):
            for subdir, curdir, filez in walk(pth, followlinks=True):
                assert (
                    len([f for f in filez if f.endswith(".nd2")]) < 2
                ), "directory had multiple nd2 files. Either specify a direct path or (preferably) organize your data so that every nd2 file is in a separate folder"

                for f in filez:
                    if f == "metadata.pickle":
                        return "PICKLE"
                for f in filez:
                    fname, fext = path.splitext(f)
                    if fext == ".nd2":
                        return "ND2"
                    if fext == ".txt":
                        if fname == "metadata":
                            return "MM"
                        if fname == "Metadata":
                            return "TXT"
                for f in filez:  # if you couldn't find any others, try tiffs
                    fname, fext = path.splitext(f)
                    if fext == ".tif" or fext == ".TIF":
                        print("Manual loading from tiffs")
                        return "TIFFS"
        else:
            fname, fext = path.splitext(str.split(pth, os.path.sep)[-1])
            if str.split(pth, os.path.sep)[-1] == "metadata.pickle":
                return "PICKLE"
            if fext == ".nd2":
                return "ND2"
            if fext == ".txt":
                if fname == "metadata":
                    return "MM"
                if fname == "Metadata":
                    return "TXT"
            if fext == ".tif" or fext == ".TIF":
                print("Manual loading from tiffs")
                return "TIFFS"
        return None

    def _determine_metadata_name(self, pth):
        """
        Helper function to determine metadata name
        Returns
        -------
        str - MD name
        """

        import os
        from os import path, walk

        fname, fext = path.splitext(pth)
        if path.isdir(pth):
            for subdir, curdir, filez in walk(pth, followlinks=True):
                for f in filez:
                    if f == "metadata.pickle":
                        return f
                for f in filez:
                    fname, fext = path.splitext(f)
                    if fext == ".nd2":
                        return f
                    if fext == ".txt":
                        if fname == "metadata":
                            return f
                        if fname == "Metadata":
                            return f
        else:
            fname, fext = path.splitext(str.split(pth, os.path.sep)[-1])
            if str.split(pth, os.path.sep)[-1] == "metadata.pickle":
                return str.split(pth, os.path.sep)[-1]
            if fext == ".nd2":
                return str.split(pth, os.path.sep)[-1]
            if fext == ".txt":
                if fname == "metadata":
                    return str.split(pth, os.path.sep)[-1]
                if fname == "Metadata":
                    return str.split(pth, os.path.sep)[-1]
        return None

    def _load_metadata(self, verbose=True):
        """
        Helper function to load metadata.

        Parameters
        -------
        verbose - [True] boolean
        """
        if self.type == "TIFFS":
            self._load_method(pth=self.base_pth)
        elif self._md_name in listdir(self.base_pth):
            self.append(self._load_method(pth=self.base_pth, fname=self._md_name))
            if verbose:
                print(
                    "loaded "
                    + self.type
                    + " metadata from "
                    + join(self.base_pth, self._md_name)
                )
        else:
            # if there is no MD in the folder, look at subfolders and append all
            for subdir, curdir, filez in walk(self.base_pth, followlinks=True):
                for f in filez:
                    if f == self._md_name:
                        self.append(self._load_method(pth=join(subdir), fname=f))
                        if verbose:
                            print(
                                "loaded "
                                + self.type
                                + " metadata from"
                                + join(subdir, f)
                            )

    def _load_metadata_nd(self, pth, fname="", delimiter="\t"):
        """
        Helper function to load nikon nd2 metadata.
        Returns
        -------
        image_table - pd dataframe of metadata image table
        """

        import nd2
        import pandas as pd

        data = []
        with nd2.ND2File(join(self.base_pth, fname)) as f:
            pixsize = f.voxel_size().x
            for event in f.events():
                if "Index" not in event:  # non image event
                    continue
                framedata = {
                    "acq": fname,
                    "Position": event.get("Position Name") or str(event.get("P Index", "")) or 'Pos0',
                    "frame": event.get("T Index") or 0,
                    "XY": [event.get("X Coord [µm]"), event.get("Y Coord [µm]")] if event.get("X Coord [µm]") else [0,0],
                    "Z": event.get("Z Coord [µm]"),
                    "Zindex": event.get("Z Index") or 0,
                    "Exposure": event.get("Exposure Time [ms]"),
                    "PixelSize": pixsize or 1,
                    "TimestampFrame": event.get("Time [s]") * 1000,
                    "TimestampImage": event.get("Time [s]") * 1000,
                    "filename": fname.replace(self.base_pth, ""),
                    "root_pth": join(self.base_pth, fname.split("/")[-1]),
                }
                for channel in f.metadata.channels:
                    data.append({**framedata, "Channel": channel.channel.name})

        return pd.DataFrame(data)

    def export_as_text(self, fname="Metadata.txt"):
        from os.path import join
        from pathlib import Path

        filepath = Path(join(self.base_pth, fname))
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.image_table.to_csv(filepath, sep="\t")

    def _load_metadata_MM(self, pth, fname="metadata.txt", delimiter="\t"):
        """
        Helper function to load a micromanager txt metadata file.
        Returns
        -------
        image_table - pd dataframe of metadata image table
        """
        import json

        try:
            with open(join(pth, fname)) as f:
                mddata = json.load(f)
        except:
            with open(join(pth, fname), "a") as f:
                f.write("}")
            with open(join(pth, fname)) as f:
                mddata = json.load(f)

        usecolslist = [
            "acq",
            "Position",
            "frame",
            "Channel",
            "Marker",
            "Fluorophore",
            "group",
            "XY",
            "Z",
            "Zindex",
            "Exposure",
            "PixelSize",
            "PlateType",
            "TimestampFrame",
            "TimestampImage",
            "filename",
        ]
        image_table = pd.DataFrame(columns=usecolslist)
        mdsum = mddata["Summary"]

        mdkeys = [key for key in mddata.keys() if key.startswith("Metadata")]

        for key in mdkeys:
            mdsing = mddata[key]
            framedata = {
                "acq": mdsum["Prefix"],
                "Position": mdsing["PositionName"],
                "frame": mdsing["Frame"],
                "Channel": mdsum["ChNames"][mdsing["ChannelIndex"]],
                "Marker": mdsum["ChNames"][mdsing["ChannelIndex"]],
                "Fluorophore": mdsing["XLIGHT Emission Wheel-Label"],
                "group": mdsing["PositionName"],
                "XY": [mdsing["XPositionUm"], mdsing["YPositionUm"]],
                "Z": mdsing["ZPositionUm"],
                "Zindex": mdsing["SliceIndex"],
                "Exposure": mdsing["Exposure-ms"],
                "PixelSize": mdsing["PixelSizeUm"],
                "PlateType": "NA",
                "TimestampFrame": mdsing["ReceivedTime"],
                "TimestampImage": mdsing["ReceivedTime"],
                "root_pth": mdsing["FileName"],
            }
            # image_table = image_table.append(framedata, sort=False, ignore_index=True)
            image_table = pd.concat(
                [image_table, pd.DataFrame([framedata])],
                sort=False,
                ignore_index=True,
            )

        image_table.root_pth = [
            join(pth, f.split("/")[-1]) for f in image_table.root_pth
        ]
        image_table["filename"] = [
            f.replace(self.base_pth, "") for f in image_table.root_pth
        ]
        return image_table

    def _load_metadata_txt(self, pth, fname="Metadata.txt", delimiter="\t"):
        """
        Helper function to load a text metadata file.
        Returns
        -------
        image_table - pd dataframe of metadata image table
        """
        image_table = pd.read_csv(join(pth, fname), delimiter=delimiter)

        image_table["root_pth"] = image_table.filename

        image_table["filename"] = [
            f.replace(self.base_pth, "") for f in image_table.filename
        ]
        return image_table

    # functions for generic TIF loading!

    def _load_metadata_TIF_GUI(self, pth=""):
        GlobText = self._getPatternsFromPathGUI(pth=pth)
        box1 = self._getMappingFromGUI(GlobText)

    def _getPatternsFromPathGUI(self, pth=""):
        from tkinter import Tk, filedialog

        from IPython.display import display
        from ipywidgets import Button, HBox, Layout, Text, widgets

        from oyLabImaging.Processing.generalutils import (
            extractFieldsByRegex,
            findregexp,
        )

        out1 = widgets.Output()
        out2 = widgets.Output()
        out3 = widgets.Output()

        FolderText = Text(
            value="",
            placeholder="Enter path to image files",
            description="Path:",
            layout=Layout(width="70%", height="30px"),
        )

        GlobText = Text(
            value="",
            placeholder="*",
            description="Regular expression:",
            layout=Layout(width="70%", height="30px"),
            style={"description_width": "initial"},
        )
        GlobText.flag = 0

        def select_files(b):
            with out1:
                out1.clear_output()
                root = Tk()
                root.withdraw()  # Hide the main window.
                root.call(
                    "wm", "attributes", ".", "-topmost", True
                )  # Raise the root to the top of all windows.
                b.dir = (
                    filedialog.askdirectory()
                )  # List of selected folder will be set button's file attribute.
                FolderText.value = b.dir + os.path.sep + "*"
                print(b.dir)  # Print the list of files selected.

        fileselect = Button(description="Browse directory")
        fileselect.on_click(select_files)

        left_box = HBox([fileselect, FolderText])
        display(left_box)

        def on_value_change_path(change):
            with out1:
                out1.clear_output()
                print(change["new"])
            with out2:
                out2.clear_output()
                GlobText.fnames = fnamesFromPath(change["new"])
                if len(GlobText.fnames):
                    GlobText.value = findregexp(GlobText.fnames)
                    GlobText.globExp = GlobText.value
                    GlobText.patterns = extractFieldsByRegex(
                        GlobText.globExp, GlobText.fnames
                    )
                    GlobText.path = FolderText.value
                    GlobText.flag = 1
                else:
                    GlobText.value = "Empty File List!"
                    GlobText.flag = 0
                print(*GlobText.fnames[0 : min(5, len(GlobText.fnames))], sep="\n")

        def fnamesFromPath(pth):
            import glob
            import os

            fnames = glob.glob(
                pth + "**" + os.path.sep + "*.[tT][iI][fF]", recursive=True
            )
            fnames.sort(key=os.path.getmtime)
            return fnames

        def on_value_change_glob(change):
            with out3:
                out3.clear_output()
                if len(GlobText.fnames):
                    GlobText.globExp = GlobText.value
                    GlobText.patterns = extractFieldsByRegex(
                        GlobText.globExp, GlobText.fnames
                    )
                else:
                    GlobText.value = "Empty File List!"

        FolderText.observe(on_value_change_path, names="value")
        FolderText.value = pth

        GlobText.observe(on_value_change_glob, names="value")

        out1.append_stdout(pth)

        print("Files:")
        display(out2)

        display(GlobText)
        return GlobText

    def _getMappingFromGUI(self, GlobText):
        import itertools

        from IPython.display import display
        from ipywidgets import Button, HBox, Layout, widgets

        out1 = widgets.Output()
        out3 = widgets.Output()
        box1 = HBox()
        buttonDone = Button(
            description="Done",
            layout=Layout(width="25%", height="80px"),
            button_style="success",
            style=dict(font_size="48", font_weight="bold"),
        )

        def change_traits(change):
            with out1:
                out1.clear_output()
                ordered_list_of_traits = []
                for i in range(1, len(box1.children), 2):
                    ordered_list_of_traits.append(box1.children[i].value)
                box1.ordered_list_of_traits = ordered_list_of_traits
                print(ordered_list_of_traits)
                print(*GlobText.patterns[0 : min(5, len(GlobText.fnames))], sep="\n")

        def on_change(t):
            with out3:
                out3.clear_output()
                if GlobText.flag:
                    print(
                        "Found regular expression with "
                        + str(len(GlobText.patterns[0]))
                        + " :"
                    )

                    parts = GlobText.globExp.split("*")
                    options = [
                        "Channel",
                        "Position",
                        "frame",
                        "Zindex",
                        "acq",
                        "IGNORE",
                    ]
                    layout = widgets.Layout(
                        width="auto", height="40px"
                    )  # set width and height

                    dddict = {}
                    for i in range(len(parts) - 1):
                        # dynamically create key
                        key = parts[i]
                        # calculate value
                        value = [
                            widgets.Label(parts[i]),
                            widgets.Dropdown(
                                options=options,
                                value=options[i],
                                layout=Layout(width="9%"),
                            ),
                        ]
                        dddict[key] = value
                    key = parts[-1]
                    value = [widgets.Label(parts[-1])]
                    dddict[key] = value

                    ddlist = list(itertools.chain.from_iterable(dddict.values()))
                    box1.children = tuple(ddlist)
                    for dd in box1.children:
                        dd.observe(change_traits, names="value")
                    box1.children[1].value = "frame"
                    box1.children[1].value = "Channel"
                    display(box1)

        def on_click_done(b):
            b.disabled = True
            self._load_imagetable_Tiffs(
                box1.ordered_list_of_traits,
                GlobText.fnames,
                GlobText.patterns,
                GlobText.path,
            )

        display(out3)
        if GlobText.flag:
            on_change(GlobText)

        box2 = HBox([out1, buttonDone])
        display(box2)

        GlobText.observe(on_change, names="value")
        buttonDone.on_click(on_click_done)

        return box1

    def _load_imagetable_Tiffs(self, ordered_list_of_traits, fnames, patterns, pth):
        """
        Helper function to load minimal metadata from tiffs.
        Returns
        -------
        image_table - pd dataframe of metadata image table
        """
        from os.path import join

        import pandas as pd

        traitdict = {}
        for i, trait in enumerate(ordered_list_of_traits):
            if not trait == "IGNORE":
                key = trait
                value = [p[i] for p in patterns]
                traitdict[key] = value

        usecolslist = [
            "acq",
            "Position",
            "frame",
            "Channel",
            "Marker",
            "Fluorophore",
            "group",
            "XY",
            "Z",
            "Zindex",
            "Exposure",
            "PixelSize",
            "PlateType",
            "TimestampFrame",
            "TimestampImage",
            "filename",
        ]
        image_table = pd.DataFrame(columns=usecolslist)

        image_table["root_pth"] = fnames

        # Default values
        image_table["acq"] = pth
        image_table["XY"] = [[0, 0]] * len(fnames)
        image_table["Z"] = 0
        image_table["Zindex"] = 0
        image_table["Channel"] = 'Ch_0'
        image_table["Position"] = 'Pos0'
        image_table["frame"] = 0
        image_table["PixelSize"] = 1

        # update whichever values we got from the names
        for trait in traitdict:
            image_table[trait] = traitdict[trait]

        try:
            image_table["frame"] = [int(f) for f in image_table["frame"]]
        except:
            pass
            # todo : convert numerical strings to numbers

        image_table["TimestampFrame"] = image_table["frame"]

        image_table["filename"] = [
            f.replace(self.base_pth, "") for f in image_table.root_pth
        ]
        self.image_table = image_table

    #  CORE METADATA FUNCTIONS
    #
    # This is how everything gets added to a MD
    def append(self, framedata):
        """
        main method of adding data to metadata object
        Parameters
        ----------
        framedata - pd dataframe of metadata
        """
        # self.image_table = self.image_table.append(framedata, sort=False,ignore_index=True)

        self.image_table = pd.concat(
            [self.image_table, framedata],
            axis=0,
            join="outer",
            sort=False,
            ignore_index=True,
        )

        # framedata can be another MD dataframe
        # framedata can also be a dict of column names and values: This will be handeled in scopex
        # framedata = {}
        # for attr in column_names:
        #     framedata.update(attr=v)

    def save(self):
        """
        save metadata as a pickle file. Saves as 'metadata.pickle' in the metadata root path.
        """
        return self.pickle()

    # Save metadata in pickle format
    def pickle(self):
        """
        save metadata as a pickle file. Saves as 'metadata.pickle' in the metadata root path.
        """
        with open(join(self.base_pth, "metadata.pickle"), "wb") as dbfile:
            tempfn = self.image_table["root_pth"].copy()
            del self.image_table["root_pth"]
            pickle.dump(self, dbfile)
            self.image_table["root_pth"] = tempfn
            md_logger.info("saved metadata")

    def unpickle(self, pth, fname="*.pickle", delimiter="\t"):
        """
        load metadata from pickle file.
        Parameters
        ----------
        pth : str - path to root folder where data is
        Returns
        -------
        image_table - pd dataframe of metadata image table
        """
        from os.path import join

        with open(join(pth, fname), "rb") as dbfile:
            MD = pickle.load(dbfile)

            if path.isdir(pth):
                MD.base_pth = pth
            else:
                MD.base_pth, MD._md_name = path.split(pth)

            MD.image_table["root_pth"] = MD.image_table["filename"].copy()
            
            MD.image_table["root_pth"] = [
                join(MD.base_pth, f) for f in MD.image_table["filename"]
            ]
            self._md_name = "metadata.pickle"
            self.type = MD.type

            exclude_keys = ["image_table", "_load_method", "_md_name"]
            d = MD.__dict__
            new_d = {k: d[k] for k in set(list(d.keys())) - set(exclude_keys)}
            self.__dict__.update(new_d)
            return MD.image_table

    @alias(
        {
            "pos": "Position",
            "Pos": "Position",
            "position": "Position",
            "p": "Position",
            "frames": "frame",
            "Frame": "frame",
            "f": "frame",
            "channel": "Channel",
            "ch": "Channel",
            "c": "Channel",
            "Ch": "Channel",
        }
    )
    def stkread(
        self,
        groupby="Position",
        sortby="TimestampFrame",
        finds_only=False,
        metadata=False,
        **kwargs
    ):
        """
        Main interface of Metadata

        Parameters
        ----------
        groupby : str - all images with the same groupby field with be stacked
        sortby : str, list(str) - images in stks will be ordered by this(these) fields
        finds_only : Bool (default False) - lazy loading
        metadata : Bool (default False) - whether to return metadata of images
        register : Bool [False]
        kwargs : Property Value pairs to subset images (see below)

        Returns
        -------
        stk of images if only one value of the groupby_value
        dictionary (groupby_value : stk) if more than one groupby_value
        stk/dict, metadata table if metadata=True
        finds if finds_only true
        finds, metadata table if finds_only and metadata

        Implemented kwargs
        ------------------
        Position : str, list(str)
        Channel : str, list(str)
        Zindex : int, list(int)
        frames : int, list(int)
        acq : str, list(str)
        """

        # pd.set_option('mode.chained_assignment', None)

        image_subset_table = self.image_table.copy()
        # Filter images according to given criteria
        for attr in image_subset_table.columns:
            if attr in kwargs:
                if not isinstance(kwargs[attr], (list, np.ndarray)):
                    kwargs[attr] = [kwargs[attr]]
                image_subset_table = image_subset_table[
                    image_subset_table[attr].isin(kwargs[attr])
                ]

        # Group images and sort them then extract filenames/indices of sorted images
        image_subset_table_sorted = image_subset_table.sort_values(
            by=sortby, key=natsort_keygen(), inplace=False
        )
        image_groups = image_subset_table_sorted.groupby(groupby)

        finds_output = {}
        mdata = {}
        for posname in natsorted(image_groups.groups.keys()):
            finds_output[posname] = image_subset_table_sorted.loc[
                image_groups.groups[posname]
            ].index.values
            mdata[posname] = image_subset_table_sorted.loc[image_groups.groups[posname]]

        # Clunky block of code below allows getting filenames only, and handles returning
        # dictionary if multiple groups present or ndarray only if single group
        if finds_only:
            if metadata:
                if len(mdata) == 1:
                    mdata = mdata[posname]
                return finds_output, mdata
            else:
                return finds_output
        else:  # Load and organize images
            if metadata:
                if len(mdata) == 1:
                    mdata = mdata[posname]
                    return (
                        np.squeeze(self._open_file(finds_output, **kwargs)[posname]),
                        mdata,
                    )
                else:
                    return self._open_file(finds_output, **kwargs), mdata
            else:
                stk = self._open_file(finds_output, **kwargs)
                if len(list(stk.keys())) == 1:
                    return np.squeeze(stk[posname])
                else:
                    return stk

    #    def save_images(self, images, fname = '/Users/robertf/Downloads/tmp_stk.tif'):
    #        with TiffWriter(fname, bigtiff=False, imagej=True) as t:
    #            if len(images.shape)>2:
    #                for i in range(images.shape[2]):
    #                    t.save(img_as_uint(images[:,:,i]))
    #            else:
    #                t.save(img_as_uint(images))
    #        return fname

    def viewer(MD):
        """
        Napari viewer app for metadata. Lets you easily scroll through the dataset. Takes no parameters, returns no prisoners.
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

        # attr_list = ['area', 'convex_area','centroid','perimeter','eccentricity','solidity','inertia_tensor_eigvals', 'orientation'] #todo: derive list from F regioprops

        Position = list(natsorted(MD.posnames))[0]

        @magicgui(
            auto_call=True,
            Acquisition={"widget_type": "Select", "choices": list(natsorted(MD.acq))},
            Position={
                "choices": list(MD.unique("Position", acq=list(natsorted(MD.acq))[0]))
            },
            Channels={"widget_type": "Select", "choices": list(MD.channels)},
            Z_Index={"choices": list(MD.Zindexes)},
        )
        def widget(
            Acquisition: List[str],
            Position: List[str],
            Channels: List[str],
            Z_Index: List,
        ):
            ch_choices = widget.Channels.choices

        @widget.Position.changed.connect
        def _on_pos_change():
            viewer.layers.clear()

        @widget.Acquisition.changed.connect
        def _on_acq_change():
            viewer.layers.clear()
            widget.Position.choices = MD.unique(
                "Position", acq=widget.Acquisition.value
            )
            widget.Channels.choices = MD.unique("Channel", acq=widget.Acquisition.value)

        movie_btn = PushButton(text="Movie")
        widget.insert(1, movie_btn)

        @movie_btn.clicked.connect
        def _on_movie_clicked():
            channels = widget.Channels.get_value()
            viewer.layers.clear()
            ch_choices = widget.Channels.choices
            pixsize = MD.unique("PixelSize")[0]

            cmaps = ["red", "green", "blue", "cyan", "magenta", "yellow"]

            if len(channels) == 1:
                cmaps = ["gray"]

            for ind, ch in enumerate(channels):
                # imgs = self._outer.img(ch, frame=list(self.T),verbose=False)
                # stk = np.array([np.pad(im1, boxsize)[crp1[0]+boxsize:crp1[2]+boxsize, crp1[1]+boxsize:crp1[3]+boxsize] for im1, crp1 in zip(imgs, crp)])

                stk = MD.stkread(
                    Position=widget.Position.value,
                    Channel=ch,
                    frame=list(MD.frame),
                    acq=widget.Acquisition.value,
                    verbose=True,
                    register=w2.value,
                    Zindex=widget.Z_Index.value,
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
                    scale=[pixsize, pixsize],
                )

        btn = PushButton(text="Next Position")
        widget.append(btn)

        @btn.clicked.connect
        def _on_next_clicked():
            choices = widget.Position.choices
            current_index = choices.index(widget.Position.value)
            widget.Position.value = choices[(current_index + 1) % (len(choices))]

        if "driftTform" in MD().columns:
            w2 = Checkbox(value=False, text="Drift Correction?")
            widget.append(w2)
        else:

            class Object(object):
                pass

            w2 = Object()
            w2.value = False

        container = Container(layout="horizontal")

        layout = container.native.layout()

        layout.addWidget(widget.native)  # adding native, because we're in Qt

        viewer.window.add_dock_widget(container, name="Metadata Viewer")

        matplotlib.use("Qt5Agg")
        return viewer

    def _read_local(
        self, ind_dict, ffield=False, register=False, verbose=True, crop=None, **kwargs
    ):
        """
        Helper function to read list of files given an TIFF type metadata and an filename list
        Load images into dictionary of stks.
        """
        pillow = False

        images_dict = {}
        for key, value in ind_dict.items():
            # key is groupby property value
            # value is list of filenames of images to be loaded as a stk

            imgs = []

            for img_idx, find in enumerate(value):
                fname = self.image_table.at[find, "root_pth"]
                # Weird print style to print on same line
                if verbose:
                    sys.stdout.write("\r" + "opening file " + path.split(fname)[-1])
                    sys.stdout.flush()
                # md_logger.info("\r"+'opening '+path.split(fname)[-1])

                # For speed: use PIL when loading a single image, imread when using stack
                im = Image.open(join(fname))
                # PIL crop seems like a faster option for registration, so we'll go with it!
                if crop is None:
                    width, height = im.size
                    crop = (0, 0, width, height)

                try:
                    im.seek(1)
                    img = io.imread(join(fname))
                except:
                    # sorry, a bit of a mess. If we crop we need to register before cropping. This is a very "cheap" way to do registration. Much faster then affine transforms. Can only do translations.

                    if type(crop) == tuple:
                        if register:
                            pillow = True
                            dT = self.image_table.at[find, "driftTform"]
                            if dT is None:
                                warnings.warn("No drift correction found for position")
                                im = im.crop(crop)
                            else:
                                dT = dT[6:8]
                                crp1 = (
                                    crop[0] - dT[1],
                                    crop[1] - dT[0],
                                    crop[2] - dT[1],
                                    crop[3] - dT[0],
                                )
                                im = im.crop(crp1)
                        else:
                            im = im.crop(crop)
                    if type(crop) == list:
                        assert len(crop) == len(
                            value
                        ), "C`est pas terrible! crop list length should be the same as loaded images"
                        if register:
                            pillow = True
                            dT = self.image_table.at[find, "driftTform"][6:8]
                            crp1 = (
                                crop[img_idx][0] - dT[1],
                                crop[img_idx][1] - dT[0],
                                crop[img_idx][2] - dT[1],
                                crop[img_idx][3] - dT[0],
                            )
                            im = im.crop(crp1)
                        else:
                            im = im.crop(crop[img_idx])
                    img = np.array(im)
                im.close()

                if ffield:
                    img = self._doFlatFieldCorrection(img, find)
                if not pillow:
                    if register:
                        img = self._register(img, find)
                # if it's a z-stack
                if img.ndim == 3:
                    img = img.transpose((1, 2, 0))

                imgs.append(img)

            # Best performance has most frequently indexed dimension first
            images_dict[key] = np.array(imgs) / 2**16
            if verbose:
                print("\nLoaded group {0} of images.".format(key))

        return images_dict

    def _read_nd2(
        self, ind_dict, ffield=False, register=False, verbose=True, crop=None, **kwargs
    ):
        """
        Helper function to read list of files given an ND type metadata and an index list.
        Load images into dictionary of stks.
        """
        pillow = False

        import nd2
        from PIL import Image

        with nd2.ND2File(self.unique("root_pth")[0]) as nd2imgs:
            images_dict = {}
            for key, value in ind_dict.items():
                imgs = []
                for img_idx, find in enumerate(value):
                    # Weird print style to print on same line
                    if verbose:
                        sys.stdout.write("\r" + "opening index " + str(find))
                        sys.stdout.flush()
                    # here we have to undo the channel count multiplication
                    # to get at the actual nd2 frame index
                    frame = nd2imgs.read_frame(find // nd2imgs.attributes.channelCount)
                    # then we index into the frame to get the actual channel
                    im = Image.fromarray(frame[find % nd2imgs.attributes.channelCount])
                    # PIL crop seems like a faster option for registration, so we'll go with it!
                    if crop is None:
                        width, height = im.size
                        crop = (0, 0, width, height)

                    if type(crop) == tuple:
                        if register:
                            pillow = True
                            dT = self.image_table.at[find, "driftTform"][6:8]
                            crp1 = (
                                crop[0] - dT[1],
                                crop[1] - dT[0],
                                crop[2] - dT[1],
                                crop[3] - dT[0],
                            )
                            im = im.crop(crp1)
                        else:
                            im = im.crop(crop)
                    if type(crop) == list:
                        assert len(crop) == len(
                            value
                        ), "C`est pas terrible! crop list length should be the same as loaded images"
                        if register:
                            pillow = True
                            dT = self.image_table.at[find, "driftTform"][6:8]
                            crp1 = (
                                crop[img_idx][0] - dT[1],
                                crop[img_idx][1] - dT[0],
                                crop[img_idx][2] - dT[1],
                                crop[img_idx][3] - dT[0],
                            )
                            im = im.crop(crp1)
                        else:
                            im = im.crop(crop[img_idx])

                    img = np.array(im)

                    if ffield:
                        img = self._doFlatFieldCorrection(img, find)
                    if not pillow:
                        if register:
                            img = self._register(img, find)
                    # if it's a z-stack
                    if img.ndim == 3:
                        img = img.transpose((1, 2, 0))

                    imgs.append(img)

                # Best performance has most frequently indexed dimension first
                images_dict[key] = np.array(imgs) / 2**16
                if verbose:
                    print("\nLoaded group {0} of images.".format(key))

            return images_dict

    def _read_ome(
        self, ind_dict, ffield=False, register=False, verbose=True, crop=None, **kwargs
    ):
        """
        Helper function to read list of files given an OME type metadata and a index list
        Load images into dictionary of stks.
        """
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None

        images_dict = {}
        for key, value in ind_dict.items():
            # key is groupby property value
            # value is list of filenames of images to be loaded as a stk

            imgs = []

            for img_idx, find in enumerate(value):
                fname = self.image_table.at[find, "root_pth"]
                chind = self.image_table.at[find, "Marker"]
                crop = (
                    self.image_table.at[find, "XY"][::-1]
                    / self.image_table.at[find, "PixelSize"]
                )
                crop = tuple(np.concatenate((crop, crop + self.single_image_size)))

                # Weird print style to print on same line
                if verbose:
                    sys.stdout.write("\r" + "opening file " + path.split(fname)[-1])
                    sys.stdout.flush()
                # md_logger.info("\r"+'opening '+path.split(fname)[-1])

                # For speed: use PIL when loading a single image, imread when using stack
                im = Image.open(join(fname))
                # PIL crop seems like a faster option for registration, so we'll go with it!
                if crop is None:
                    width, height = im.size
                    crop = (0, 0, width, height)

                im.seek(chind)

                im = im.crop(crop)

                img = np.array(im)
                im.close()

                # if it's a z-stack
                if img.ndim == 3:
                    img = img.transpose((1, 2, 0))

                imgs.append(img)

            # Best performance has most frequently indexed dimension first
            images_dict[key] = np.array(imgs) / 2**16
            if verbose:
                print("\nLoaded group {0} of images.".format(key))

        return images_dict

    # Function to apply flat field correction
    def _doFlatfieldCorrection(self, img, flt, **kwargs):
        """
        Perform flatfield correction.

        Parameters
        ----------
        img : numpy.ndarray
            2D image of type integer
        flt : numpy.ndarray
            2D image of type integer with the flatfield
        """
        print("Not implemented well. Woulnd't advise using")
        cameraoffset = 100.0 / 2**16
        bitdepth = 2.0**16
        flt = flt.astype(np.float32) - cameraoffset
        flt = np.divide(flt, np.nanmean(flt.flatten()))

        img = np.divide((img - cameraoffset).astype(np.float32), flt + cameraoffset)
        flat_img = img.flatten()
        rand_subset = np.random.randint(0, high=len(flat_img), size=10000)
        flat_img = flat_img[rand_subset]
        flat_img = np.percentile(flat_img, 1)
        np.place(img, flt < 0.05, flat_img)
        np.place(img, img < 0, 0)
        np.place(img, img > bitdepth, bitdepth)
        return img

    def _register_pil(self, im, find):
        """
        Perform image registration.

        Parameters
        ----------
        img : numpy.ndarray
            2D image of type integer
        find : file identifier in metadata table
        """
        from PIL import Image

        if "driftTform" in self().columns:
            dT = self.image_table.at[find, "driftTform"]
            if dT is None:
                warnings.warn("No drift correction found for position")
                return im
            dT = np.array(dT)
            if len(dT) == 9:
                M = np.reshape(dT, (3, 3)).transpose()
                imreturn = im.transform(
                    im.size, Image.AFFINE, M.flatten()[0:6], resample=Image.NEAREST
                )
                return imreturn
        else:
            warnings.warn("No drift correction found for experiment")
            return im

    def _register(self, img, find):
        """
        Perform image registration.

        Parameters
        ----------
        img : numpy.ndarray
            2D image of type integer
        find : file identifier in metadata table
        """
        import cv2

        if "driftTform" in self().columns:
            dT = self.image_table.at[find, "driftTform"]
            if dT is None:
                warnings.warn("No drift correction found for position")
                return img
            dT = np.array(dT)
            if len(dT) == 9:
                M = np.reshape(dT, (3, 3)).transpose()
                # cv2/scikit and numpy index differently
                M[0, 2], M[1, 2] = M[1, 2], M[0, 2]
                imreturn = cv2.warpAffine(
                    (img * 2**16).astype("float64"), M[:2], img.shape[::-1]
                )
                return imreturn / 2**16
                # return transform.warp(img, np.linalg.inv(M), output_shape=img.shape,preserve_range=True)
        else:
            warnings.warn("No drift correction found for experiment")
            return img

    # Calculate jitter/drift corrections
    def CalculateDriftCorrection(
        self,
        Position=None,
        frames=None,
        ZsToLoad=[0],
        Channel=None,
        threads=None,
        chunks=20,
        GPU=False,
    ):
        if GPU:
            try:
                print("Trying GPU calculation in chunks of " + str(chunks))
                self.CalculateDriftCorrectionGPUChunks(
                    Position=Position,
                    frames=frames,
                    ZsToLoad=ZsToLoad,
                    Channel=Channel,
                    chunks=chunks,
                )
            except:
                print(
                    "No GPU or no CuPy. If you have a GPU, try installing CuPy, calculating on CPU"
                )
                self.CalculateDriftCorrectionCPU(
                    Position=Position,
                    frames=frames,
                    ZsToLoad=ZsToLoad,
                    Channel=Channel,
                    threads=threads,
                )
        else:
            self.CalculateDriftCorrectionCPU(
                Position=Position,
                frames=frames,
                ZsToLoad=ZsToLoad,
                Channel=Channel,
                threads=threads,
            )

    def CalculateDriftCorrectionCPU(
        self, Position=None, frames=None, ZsToLoad=[0], Channel=None, threads=None,
    ):
        """
        Calculate image registration (jitter correction) parameters and add to metadata.

        Parameters
        ----------
        Position : str or list of strings
        ZsToLoad : list of int - which Zs to load to calculate registration. If list, registration will be the average of the different zstack layers
        Channel : str, channel name to use for registration

        """
        from oyLabImaging.Processing.improcutils import periodic_smooth_decomp

        if threads is not None:
            warnings.warn("Passing threads is deprecated. It does nothing.")

        # from scipy.signal import fftconvolve
        if Position is None:
            Position = self.posnames
        elif not isinstance(Position, (np.ndarray, list)):
            Position = [Position]

        if frames is None:
            frames = self.frames
        elif not isinstance(frames, (np.ndarray, list)):
            Position = [frames]

        if Channel is None:
            Channel = self.channels[0]
        assert Channel in self.channels, "%s isn't a channel, try %s" % (
            Channel,
            ", ".join(list(self.channels)),
        )
        print("using channel " + Channel + " for drift correction")
        for pos in Position:
            from scipy.fft import fft2, ifft2

            DataPre = self.stkread(
                Position=pos,
                Channel=Channel,
                Zindex=ZsToLoad,
                frame=frames,
                register=False,
                verbose=False,
            )

            assert (
                DataPre.ndim == 3
            ), "Must have more than 1 timeframe for drift correction"
            DataPre,_ = periodic_smooth_decomp(DataPre)
            print("\ncalculating drift correction for position " + str(pos) + " on CPU")
            DataPre = DataPre - np.mean(DataPre, axis=(1, 2), keepdims=True)

            DataPost = DataPre[1:, :, :].transpose((1, 2, 0))
            DataPre = DataPre[:-1, :, :].transpose((1, 2, 0))
            # this is in prep for # Zs>1
            DataPre = np.reshape(
                DataPre,
                (DataPre.shape[0], DataPre.shape[1], len(ZsToLoad), len(frames) - 1),
            )
            DataPost = np.reshape(
                DataPost,
                (DataPost.shape[0], DataPost.shape[1], len(ZsToLoad), len(frames) - 1),
            )

            # calculate cross correlation
            DataPost = np.rot90(DataPost, axes=(0, 1), k=2)

            # So because of dumb licensing issues, fftconvolve can't use fftw but the slower fftpack. Python is wonderful. So we'll do it all by hand like neanderthals
            img_fft_1 = fft2(DataPre, axes=(0, 1))
            img_fft_2 = fft2(DataPost, axes=(0, 1))
            imXcorr = np.abs(
                np.fft.ifftshift(
                    ifft2(img_fft_1 * img_fft_2, axes=(0, 1)),
                    axes=(0, 1),
                )
            )

            # if more than 1 slice is calculated, look for mean shift
            imXcorrMeanZ = np.mean(imXcorr, axis=2)
            c = []
            for i in range(imXcorrMeanZ.shape[-1]):
                c.append(np.squeeze(imXcorrMeanZ[:, :, i]).argmax())

            d = (
                np.transpose(
                    np.unravel_index(c, np.squeeze(imXcorrMeanZ[:, :, 0]).shape)
                )
                - np.array(np.squeeze(imXcorrMeanZ[:, :, 0]).shape) / 2
                + 1
            )  # python indexing starts at 0
            D = np.insert(np.cumsum(d, axis=0), 0, [0, 0], axis=0)

            if "driftTform" not in self.image_table.columns:
                self.image_table["driftTform"] = None

            for frmind, frame in enumerate(frames):
                inds = self.unique(Attr="index", Position=pos, frame=frame)
                for ind in inds:
                    self.image_table.at[ind, "driftTform"] = [
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        np.ceil(D[frmind, 0]),
                        np.ceil(D[frmind, 1]),
                        1,
                    ]
            print("calculated drift correction for position " + str(pos))
        self.pickle()

    def CalculateDriftCorrectionGPU(
        self, Position=None, frames=None, ZsToLoad=[0], Channel=None, threads=8
    ):
        """
        Calculate image registration (jitter correction) parameters and add to metadata.

        Parameters
        ----------
        Position : str or list of strings
        ZsToLoad : list of int - which Zs to load to calculate registration. If list, registration will be the average of the different zstack layers
        Channel : str, channel name to use for registration

        """
        from oyLabImaging.Processing.improcutils import periodic_smooth_decomp

        from cupy import _default_memory_pool, asarray
        from cupy.fft import fft2, ifft2

        # from scipy.signal import fftconvolve
        if Position is None:
            Position = self.posnames
        elif not isinstance(Position, (np.ndarray, list)):
            Position = [Position]

        if frames is None:
            frames = self.frames
        elif not isinstance(frames, (np.ndarray, list)):
            Position = [frames]

        if Channel is None:
            Channel = self.channels[0]
        assert Channel in self.channels, "%s isn't a channel, try %s" % (
            Channel,
            ", ".join(list(self.channels)),
        )
        print("using channel " + Channel + " for drift correction")

        for pos in Position:
            DataPre = asarray(
                self.stkread(
                    Position=pos,
                    Channel=Channel,
                    Zindex=ZsToLoad,
                    frame=frames,
                    register=False,
                    verbose=False,
                )
            )
            assert (
                DataPre.ndim == 3
            ), "Must have more than 1 timeframe for drift correction"
            print("\ncalculating drift correction for position " + str(pos) + " on GPU")
            DataPre = self.stkread(
                Position=pos, Channel=Channel, Zindex=ZsToLoad, frame=fr, register=False
            )
            
            DataPre,_ = periodic_smooth_decomp(DataPre)

            DataPre = DataPre - np.mean(DataPre, axis=(1, 2), keepdims=True)

            DataPost = DataPre[1:, :, :].transpose((1, 2, 0))
            DataPre = DataPre[:-1, :, :].transpose((1, 2, 0))
            # this is in prep for # Zs>1
            DataPre = np.reshape(
                DataPre,
                (DataPre.shape[0], DataPre.shape[1], len(ZsToLoad), len(fr) - 1),
            )
            DataPost = np.reshape(
                DataPost,
                (DataPost.shape[0], DataPost.shape[1], len(ZsToLoad), len(fr) - 1),
            )

            # calculate cross correlation
            DataPost = np.rot90(DataPost, axes=(0, 1), k=2)

            DataPre = asarray(DataPre)
            DataPost = asarray(DataPost)

            # So because of dumb licensing issues, fftconvolve can't use fftw but the slower fftpack. Python is wonderful. So we'll do it all by hand like neanderthals
            img_fft_1 = fft2(DataPre, axes=(0, 1))
            del DataPre
            _default_memory_pool.free_all_blocks()
            img_fft_2 = fft2(DataPost, axes=(0, 1))
            del DataPost
            _default_memory_pool.free_all_blocks()
            imXcorr = np.abs(
                np.fft.ifftshift(ifft2(img_fft_1 * img_fft_2, axes=(0, 1)), axes=(0, 1))
            )
            del img_fft_1
            _default_memory_pool.free_all_blocks()
            del img_fft_2
            _default_memory_pool.free_all_blocks()
            # if more than 1 slice is calculated, look for mean shift
            imXcorrMeanZ = np.mean(imXcorr, axis=2).get()
            del imXcorr
            c = []
            for i in range(imXcorrMeanZ.shape[-1]):
                c.append(np.squeeze(imXcorrMeanZ[:, :, i]).argmax())

            _default_memory_pool.free_all_blocks()

            d = (
                np.transpose(
                    np.unravel_index(c, np.squeeze(imXcorrMeanZ[:, :, 0]).shape)
                )
                - np.array(np.squeeze(imXcorrMeanZ[:, :, 0]).shape) / 2
                + 1
            )  # python indexing starts at 0
            D = np.insert(np.cumsum(d, axis=0), 0, [0, 0], axis=0)

            if "driftTform" not in self.image_table.columns:
                self.image_table["driftTform"] = None

            for frmind, frame in enumerate(frames):
                inds = self.unique(Attr="index", Position=pos, frame=frame)
                for ind in inds:
                    self.image_table.at[ind, "driftTform"] = [
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        np.ceil(D[frmind, 0]),
                        np.ceil(D[frmind, 1]),
                        1,
                    ]
            print("calculated drift correction for position " + str(pos))
        self.pickle()

    def CalculateDriftCorrectionGPUChunks(
        self, Position=None, frames=None, ZsToLoad=[0], Channel=None, chunks=10
    ):
        """
        Calculate image registration (jitter correction) parameters and add to metadata.

        Parameters
        ----------
        Position : str or list of strings
        ZsToLoad : list of int - which Zs to load to calculate registration. If list, registration will be the average of the different zstack layers
        Channel : str, channel name to use for registration

        """
        from cupy import asarray, get_default_memory_pool
        from cupy.fft import fft2, ifft2

        mempool = get_default_memory_pool()
        # from scipy.signal import fftconvolve
        if Position is None:
            Position = self.posnames
        elif not isinstance(Position, (np.ndarray, list)):
            Position = [Position]

        if frames is None:
            frames = self.frames
        elif not isinstance(frames, (np.ndarray, list)):
            Position = [frames]

        if Channel is None:
            Channel = self.channels[0]
        assert Channel in self.channels, "%s isn't a channel, try %s" % (
            Channel,
            ", ".join(list(self.channels)),
        )
        print("using channel " + Channel + " for drift correction")

        def chunker_with_overlap(seq, size):
            return (
                seq[np.max((0, pos - 1)) : pos + size]
                for pos in range(0, len(seq), size)
            )

        for pos in Position:
            ds = np.empty((0, 2), float)
            print("\ncalculating drift correction for position " + str(pos) + " on GPU")
            for fr in chunker_with_overlap(frames, chunks):
                DataPre = self.stkread(
                    Position=pos,
                    Channel=Channel,
                    Zindex=ZsToLoad,
                    frame=fr,
                    register=False,
                    verbose=False,
                )
                assert (
                    DataPre.ndim == 3
                ), "Must have more than 1 timeframe for drift correction"
                DataPre = DataPre - np.mean(DataPre, axis=(1, 2), keepdims=True)

                DataPost = DataPre[1:, :, :].transpose((1, 2, 0))
                DataPre = DataPre[:-1, :, :].transpose((1, 2, 0))
                # this is in prep for # Zs>1
                DataPre = np.reshape(
                    DataPre,
                    (DataPre.shape[0], DataPre.shape[1], len(ZsToLoad), len(fr) - 1),
                )
                DataPost = np.reshape(
                    DataPost,
                    (DataPost.shape[0], DataPost.shape[1], len(ZsToLoad), len(fr) - 1),
                )

                # calculate cross correlation
                DataPost = np.rot90(DataPost, axes=(0, 1), k=2)

                DataPre = asarray(DataPre)
                DataPost = asarray(DataPost)

                # So because of dumb licensing issues, fftconvolve can't use fftw but the slower fftpack. Python is wonderful. So we'll do it all by hand like neanderthals
                img_fft_1 = fft2(DataPre, axes=(0, 1))
                del DataPre
                mempool.free_all_blocks()
                img_fft_2 = fft2(DataPost, axes=(0, 1))
                del DataPost
                mempool.free_all_blocks()
                SF = img_fft_1 * img_fft_2
                del img_fft_1
                del img_fft_2
                mempool.free_all_blocks()
                imXcorr = np.abs(np.fft.ifftshift(ifft2(SF, axes=(0, 1)), axes=(0, 1)))
                del SF
                mempool.free_all_blocks()
                # if more than 1 slice is calculated, look for mean shift
                imXcorrMeanZ = np.mean(imXcorr, axis=2).get()
                del imXcorr
                mempool.free_all_blocks()
                c = []
                for i in range(imXcorrMeanZ.shape[-1]):
                    c.append(np.squeeze(imXcorrMeanZ[:, :, i]).argmax())

                mempool.free_all_blocks()

                d = (
                    np.transpose(
                        np.unravel_index(c, np.squeeze(imXcorrMeanZ[:, :, 0]).shape)
                    )
                    - np.array(np.squeeze(imXcorrMeanZ[:, :, 0]).shape) / 2
                    + 1
                )  # python indexing starts at 0
                del imXcorrMeanZ
                del c
                ds = np.concatenate((ds, d))
            D = np.insert(np.cumsum(ds, axis=0), 0, [0, 0], axis=0)

            if "driftTform" not in self.image_table.columns:
                self.image_table["driftTform"] = None

            for frmind, frame in enumerate(frames):
                inds = self.unique(Attr="index", Position=pos, frame=frame)
                for ind in inds:
                    self.image_table.at[ind, "driftTform"] = [
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        np.ceil(D[frmind, 0]),
                        np.ceil(D[frmind, 1]),
                        1,
                    ]
            print("calculated drift correction for position " + str(pos))
        self.pickle()

    def CalculateDriftCorrectionCPUChunks(
        self, Position=None, frames=None, ZsToLoad=[0], Channel=None, chunks=10
    ):
        """
        Calculate image registration (jitter correction) parameters and add to metadata.

        Parameters
        ----------
        Position : str or list of strings
        ZsToLoad : list of int - which Zs to load to calculate registration. If list, registration will be the average of the different zstack layers
        Channel : str, channel name to use for registration

        """
        from numpy import asarray
        from numpy.fft import fft2, ifft2

        # from scipy.signal import fftconvolve
        if Position is None:
            Position = self.posnames
        elif not isinstance(Position, (np.ndarray, list)):
            Position = [Position]

        if frames is None:
            frames = self.frames
        elif not isinstance(frames, (np.ndarray, list)):
            Position = [frames]

        if Channel is None:
            Channel = self.channels[0]
        assert Channel in self.channels, "%s isn't a channel, try %s" % (
            Channel,
            ", ".join(list(self.channels)),
        )
        print("using channel " + Channel + " for drift correction")

        def chunker_with_overlap(seq, size):
            return (
                seq[np.max((0, pos - 1)) : pos + size]
                for pos in range(0, len(seq), size)
            )

        for pos in Position:
            ds = np.empty((0, 2), float)
            print("\ncalculating drift correction for position " + str(pos) + " on CPU")
            for fr in chunker_with_overlap(frames, chunks):
                DataPre = self.stkread(
                    Position=pos,
                    Channel=Channel,
                    Zindex=ZsToLoad,
                    frame=fr,
                    register=False,
                    verbose=False,
                )
                assert (
                    DataPre.ndim == 3
                ), "Must have more than 1 timeframe for drift correction"
                DataPre = DataPre - np.mean(DataPre, axis=(1, 2), keepdims=True)

                DataPost = DataPre[1:, :, :].transpose((1, 2, 0))
                DataPre = DataPre[:-1, :, :].transpose((1, 2, 0))
                # this is in prep for # Zs>1
                DataPre = np.reshape(
                    DataPre,
                    (DataPre.shape[0], DataPre.shape[1], len(ZsToLoad), len(fr) - 1),
                )
                DataPost = np.reshape(
                    DataPost,
                    (DataPost.shape[0], DataPost.shape[1], len(ZsToLoad), len(fr) - 1),
                )

                # calculate cross correlation
                DataPost = np.rot90(DataPost, axes=(0, 1), k=2)

                DataPre = asarray(DataPre)
                DataPost = asarray(DataPost)

                # So because of dumb licensing issues, fftconvolve can't use fftw but the slower fftpack. Python is wonderful. So we'll do it all by hand like neanderthals
                img_fft_1 = fft2(DataPre, axes=(0, 1))
                del DataPre
                img_fft_2 = fft2(DataPost, axes=(0, 1))
                del DataPost
                SF = img_fft_1 * img_fft_2
                del img_fft_1
                del img_fft_2
                imXcorr = np.abs(np.fft.ifftshift(ifft2(SF, axes=(0, 1)), axes=(0, 1)))
                del SF
                # if more than 1 slice is calculated, look for mean shift
                imXcorrMeanZ = np.mean(imXcorr, axis=2)
                del imXcorr
                c = []
                for i in range(imXcorrMeanZ.shape[-1]):
                    c.append(np.squeeze(imXcorrMeanZ[:, :, i]).argmax())


                d = (
                    np.transpose(
                        np.unravel_index(c, np.squeeze(imXcorrMeanZ[:, :, 0]).shape)
                    )
                    - np.array(np.squeeze(imXcorrMeanZ[:, :, 0]).shape) / 2
                    + 1
                )  # python indexing starts at 0
                del imXcorrMeanZ
                del c
                ds = np.concatenate((ds, d))
            D = np.insert(np.cumsum(ds, axis=0), 0, [0, 0], axis=0)

            if "driftTform" not in self.image_table.columns:
                self.image_table["driftTform"] = None

            for frmind, frame in enumerate(frames):
                inds = self.unique(Attr="index", Position=pos, frame=frame)
                for ind in inds:
                    self.image_table.at[ind, "driftTform"] = [
                        1,
                        0,
                        0,
                        0,
                        1,
                        0,
                        np.ceil(D[frmind, 0]),
                        np.ceil(D[frmind, 1]),
                        1,
                    ]
            print("calculated drift correction for position " + str(pos))
        self.pickle()

    def try_segmentation(MD):
        from typing import List
        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np
        from magicgui import magicgui
        from magicgui.widgets import Checkbox, Container, PushButton, LineEdit
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from natsort import natsorted
        from cellpose import models
        from scipy import stats
        from oyLabImaging.Processing.improcutils import segmentation

        from oyLabImaging.Processing.imvisutils import get_or_create_viewer

        cmaps = ["cyan", "magenta", "yellow", "red", "green", "blue"]
        viewer = get_or_create_viewer()

        Position = list(natsorted(MD.posnames))[0]

        @magicgui(
            auto_call=False,
            Acquisition={"choices": list(natsorted(MD.acq))},
            Position={"choices": list(natsorted(MD.Position))},
            NucChannels={"widget_type": "Select", "choices": list(MD.channels)},
            CytoChannels={"widget_type": "Select", "choices": list(MD.channels)},
            Frame={"choices": list(MD.frame)},
            Z_Index={"choices": list(MD.Zindexes)},
            SegmentationFunction={"choices": segmentation.segmentation_types()},
        )
        def widget(
            Acquisition: List[str],
            Position: List[str],
            NucChannels: List[str],
            CytoChannels: List[str],
            Frame: List[int],
            Z_Index: List,
            SegmentationFunction: List[str],
        ):
            ch_choices = widget.NucChannels.choices
            segfun = segmentation.segtype_to_segfun(widget.SegmentationFunction.value)
            print("blabla")

        widget.tblist = []
        widget.input_dict = {}

        def _update_list(input_dict):
            # global tblist
            for arg, de in input_dict.items():
                btn = LineEdit(label=arg, value=de, name=arg)
                widget.insert(-1, btn)
                widget.tblist.append(arg)

        @widget.SegmentationFunction.changed.connect
        def _on_seg_func_change():
            [widget.remove(s) for s in widget.tblist]
            widget.tblist = []
            segfun = segmentation.segtype_to_segfun(widget.SegmentationFunction.value)
            nargs = segfun.__code__.co_argcount
            defaults = list(segfun.__defaults__)
            # ndefault = len(defaults)
            nimgs = 2  # nargs-ndefault
            args = [segfun.__code__.co_varnames[i] for i in range(nimgs, nargs)]
            widget.input_dict = {args[i]: defaults[i + 1] for i in range(len(args))}
            widget.input_dict = {**widget.input_dict}
            _update_list(widget.input_dict)

        @widget.Z_Index.changed.connect
        def _on_Z_change():
            viewer.layers.clear()
            _on_movie_clicked()

        @widget.NucChannels.changed.connect
        def _on_nch_change():
            viewer.layers.clear()
            _on_movie_clicked()

        @widget.CytoChannels.changed.connect
        def _on_cch_change():
            viewer.layers.clear()
            _on_movie_clicked()

        @widget.Frame.changed.connect
        def _on_frame_change():
            viewer.layers.clear()
            _on_movie_clicked()

        @widget.Position.changed.connect
        def _on_pos_change():
            viewer.layers.clear()
            _on_movie_clicked()

        @widget.Acquisition.changed.connect
        def _on_acq_change():
            viewer.layers.clear()
            widget.Position.choices = MD.unique(
                "Position", acq=widget.Acquisition.value
            )
            _on_movie_clicked()

        movie_btn = PushButton(text="Image")
        widget.insert(1, movie_btn)
        pixsize = MD.unique("PixelSize")[0]

        @movie_btn.clicked.connect
        def _on_movie_clicked():
            channels = widget.NucChannels.get_value() + widget.CytoChannels.get_value()
            viewer.layers.clear()
            ch_choices = widget.NucChannels.choices

            cmaps = ["red", "green", "blue", "cyan", "magenta", "yellow"]

            if len(channels) == 1:
                cmaps = ["gray"]

            for ind, ch in enumerate(channels):
                stk = MD.stkread(
                    acq = widget.Acquisition.value,
                    Position=widget.Position.value,
                    frame=widget.Frame.value,
                    Zindex=widget.Z_Index.value,
                    Channel=ch,
                    verbose=False,
                    register=False,
                )
                stk = np.arcsinh(stk/0.001)
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
                    scale=[pixsize, pixsize],
                )

        btn = PushButton(text="Try segmentation")
        widget.append(btn)

        def clean_dict(d):
            result = {}
            for key, value in d.items():
                if not value.isnumeric():
                    value = eval(value)
                else:
                    value = float(value)
                result[key] = value
            return result

        @btn.clicked.connect
        def _on_try_clicked():
            assert (
                len(widget.NucChannels.get_value()) > 0
            ), "You must pick at least a single nuclear channel"

            NucChannel = widget.NucChannels.get_value()
            CytoChannel = widget.CytoChannels.get_value()

            Pos = widget.Position.value
            frame = widget.Frame.value
            Zindex = widget.Z_Index.value
            acq = widget.Acquisition.value
            segfun = segmentation.segtype_to_segfun(widget.SegmentationFunction.value)

            Data = {}
            for ch in NucChannel + CytoChannel:
                Data[ch] = np.squeeze(
                    MD.stkread(
                        acq = acq,
                        Channel=ch,
                        frame=frame,
                        Position=Pos,
                        Zindex=Zindex,
                        verbose=False,
                    )
                )
                assert (
                    Data[ch].ndim == 2
                ), "channel/position/frame/Zindex did not return unique result"

            imagedims = np.shape(Data[NucChannel[0]])

            try:
                imgCyto = np.sum(
                    [
                        (Data[ch] - np.mean(Data[ch])) / np.std(Data[ch])
                        for ch in CytoChannel
                    ],
                    axis=0,
                )
            except:
                imgCyto = ""

            # imgNuc = np.max([Data[ch] for ch in NucChannel], axis=0)
            imgNuc = np.sum(
                [
                    (Data[ch] - np.mean(Data[ch])) / np.std(Data[ch])
                    for ch in NucChannel
                ],
                axis=0,
            )

            widget.input_dict = {t: widget.asdict()[t] for t in widget.tblist}
            widget.input_dict = clean_dict(widget.input_dict)

            L = segfun(img=imgNuc, imgCyto=imgCyto, **widget.input_dict)
            viewer.add_labels(L, scale=[pixsize, pixsize])

        container = Container(layout="horizontal")
        container.max_width = 400
        container.max_height = 700

        layout = container.native.layout()

        layout.addWidget(widget.native)  # adding native, because we're in Qt
        dock = viewer.window.add_dock_widget(container, name="Segmentation Attempts")

        matplotlib.use("Qt5Agg")

        widget.call_button.visible = False
        movie_btn.visible = False
        _on_seg_func_change()
        return widget
