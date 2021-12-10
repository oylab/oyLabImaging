# Metadata module for microscopy data
# Currently supports Wollman Lab Scopes data, Micromanager MDA, and nikon ND2 files
# AOY

from os import walk, listdir, path
import os
from os.path import join, isdir
import sys

import pandas as pd
import numpy as np
import dill as pickle
from ast import literal_eval
import warnings 
from PIL import Image
import logging

md_logger = logging.getLogger(__name__)
md_logger.setLevel(logging.DEBUG)

usecolslist = ['acq',  'Position', 'frame','Channel', 'Marker', 'Fluorophore', 'group', 
       'XY', 'Z', 'Zindex','Exposure','PixelSize', 'PlateType', 'TimestampFrame','TimestampImage', 'filename']

#usecolslist=[]

from skimage import img_as_float, img_as_uint, io

#Metadata stores all relevant info about an imaging experiment. 

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
    
    def __init__(self, pth='', load_type='local', verbose=True):
        
        # get the base path (directory where it it) to the metadata file. 
        #If full path to file is given, separate out the directory
        self._md_name=''
        
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
        if self.type==None:
            return

        
        # How should metadata be read?
        if self.type=='PICKLE':
            self._load_method=self.unpickle
        elif self.type=='TXT':
            self._load_method=self._load_metadata_txt
        elif self.type=='MM':
            self._load_method=self._load_metadata_MM
        elif self.type=='ND2':
            self._load_method=self._load_metadata_nd
        elif self.type=='TIFFS':
            self._load_method=self._load_metadata_TIF_GUI

                  
        # With all we've learned, we can now load the metadata
        self._load_metadata(verbose=verbose)

        # Handle columns that don't import from text well
        try:
            self._convert_data('XY', float)
        except Exception as e:
            self.image_table['XY'] = [literal_eval(i) for i in self.image_table['XY']]

            
        # How should files be read? 
        if self.type=="ND2":
            self._open_file=self._read_nd2
        else: #Tiffs
            if load_type=='local':
                self._open_file=self._read_local
            elif load_type=='google_cloud':
                raise NotImplementedError("google_cloud loading is not implemented.")



    def __call__(self):
        return self.image_table
          
        
    #keeping some multiples for backwards compatability
    @property
    def posnames(self):
        """
        Returns all unique position names
        """
        return self().Position.unique()
    
    @property
    def Position(self):
        """
        Returns all unique position names
        """
        return self().Position.unique()

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
    
    def unique(self, Attr=None  , sortby='TimestampFrame',**kwargs):
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
        for key, value in kwargs.items():
            if not isinstance(value, list):
                kwargs[key] = [value]
        image_subset_table = self.image_table
        # Filter images according to some criteria
        for attr in image_subset_table.columns:
            if attr in kwargs:
                image_subset_table = image_subset_table[image_subset_table[attr].isin(kwargs[attr])]
        image_subset_table.sort_values(sortby, inplace=True)
        
        if Attr is None:
            return image_subset_table.size
        elif Attr in image_subset_table.columns:  
            return image_subset_table[Attr].unique()
        elif Attr=='index':
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
        from os import path, walk
        import os
        fname, fext = path.splitext(pth)
        if path.isdir(pth):
            for subdir, curdir, filez in walk(pth):
                
                assert len([f for f in filez if f.endswith('.nd2')])<2, "directory had multiple nd2 files. Either specify a direct path or (preferably) organize your data so that every nd2 file is in a separate folder"
                        
                for f in filez:
                    fname, fext = path.splitext(f)
                    if f=='metadata.pickle':
                        return 'PICKLE'
                    if fext=='.nd2':
                        return 'ND2'
                    if fext=='.txt':
                        if fname=='metadata':
                            return 'MM'
                        if fname=='Metadata':
                            return 'TXT'
                for f in filez: #if you couldn't find any others, try tiffs
                    if fext=='.tif' or fext=='.TIF':
                        print('Manual loading from tiffs')
                        return 'TIFFS'
        else:
            fname, fext = path.splitext(str.split(pth,os.path.sep)[-1])
            if str.split(pth,os.path.sep)[-1]=='metadata.pickle':
                return 'PICKLE'
            if fext=='.nd2':
                return 'ND2'
            if fext=='.txt':
                if fname=='metadata':
                    return 'MM'
                if fname=='Metadata':
                    return 'TXT'
            if fext=='.tif' or fext=='.TIF':
                print('Manual loading from tiffs')
                return 'TIFFS'
        return None
    
    def _determine_metadata_name(self, pth):
        """
        Helper function to determine metadata name
        Returns
        -------
        str - MD name
        """
        
        from os import path, walk
        import os
        fname, fext = path.splitext(pth)
        if path.isdir(pth):
            for subdir, curdir, filez in walk(pth):
                for f in filez:
                    fname, fext = path.splitext(f)
                    if f=='metadata.pickle':
                        return f
                    if fext=='.nd2':
                        return f
                    if fext=='.txt':
                        if fname=='metadata':
                            return f
                        if fname=='Metadata':
                            return f
        else:
            fname, fext = path.splitext(str.split(pth,os.path.sep)[-1])
            if str.split(pth,os.path.sep)[-1]=='metadata.pickle':
                return str.split(pth,os.path.sep)[-1]
            if fext=='.nd2':
                return str.split(pth,os.path.sep)[-1]
            if fext=='.txt':
                if fname=='metadata':
                    return str.split(pth,os.path.sep)[-1]
                if fname=='Metadata':
                    return str.split(pth,os.path.sep)[-1]     
        return None
 

    def _load_metadata(self, verbose=True):
        """
        Helper function to load metadata.
        
        Parameters
        -------
        verbose - [True] boolean
        """
        if self.type=="TIFFS":
            self._load_method(pth=self.base_pth)
        elif self._md_name in listdir(self.base_pth):   
            self.append(self._load_method(pth=self.base_pth ,fname=self._md_name))
            if verbose:
                print('loaded ' + self.type + ' metadata from' + join(self.base_pth, self._md_name))
        else:
            #if there is no MD in the folder, look at subfolders and append all 
            for subdir, curdir, filez in walk(self.base_pth):
                for f in filez:
                    if f==self._md_name:
                        self.append(self._load_method(pth=join(subdir),fname=f))
                        if verbose:
                            print('loaded ' + self.type + ' metadata from' +join(subdir,f))

        
    
    def _load_metadata_nd(self, pth, fname='', delimiter='\t'):
        """
        Helper function to load nikon nd2 metadata.
        Returns
        -------
        image_table - pd dataframe of metadata image table
        """
        import nd2reader as nd2
        import pandas as pd
        from os.path import sep
        import time
        
        usecolslist = ['acq',  'Position', 'frame','Channel', 'XY', 'Z', 
                       'Zindex','Exposure','PixelSize', 'TimestampFrame','TimestampImage', 'filename']
        image_table = pd.DataFrame(columns=usecolslist)

        acq = fname
        with nd2.ND2Reader(join(self.base_pth,fname)) as imgs:
            Ninds = imgs.metadata['total_images_per_channel']*len(imgs.metadata['channels'])
            frames = imgs.metadata['frames']
            imgsPerFrame = Ninds/len(frames)

            XY = np.column_stack((np.array(imgs.parser._raw_metadata.x_data),np.array(imgs.parser._raw_metadata.y_data)))
            Zpos = imgs.metadata['z_coordinates']
            Zind = imgs.metadata['z_levels']
            pixsize = imgs.metadata['pixel_microns']
            acq = fname
            for i in np.arange(Ninds):
                fps = int(i/len(imgs.metadata['channels']))
                frame = int(i/imgsPerFrame)
                xy = XY[fps,]
                z = Zpos[fps]
                props = imgs.parser.calculate_image_properties(i)
                zind = props[2]
                chan = props[1]
                pos = props[0]
                exptime = imgs.parser._raw_metadata.camera_exposure_time[fps]
                framedata={'acq':acq,'Position':pos,'frame':frame,'Channel':chan,'XY':list(xy), 'Z':z, 'Zindex':zind,'Exposure':exptime,'PixelSize':pixsize,'TimestampFrame':imgs.timesteps[fps],'TimestampImage':imgs.timesteps[fps],'filename':acq}
                image_table = image_table.append(framedata, sort=False,ignore_index=True)
            image_table['root_pth'] = image_table.filename
            image_table.filename = [join(pth, f.split(os.path.sep)[-1]) for f in image_table.filename]
            return image_table
    

    
    def _load_metadata_MM(self, pth, fname='metadata.txt', delimiter='\t'):
        """
        Helper function to load a micromanager txt metadata file.
        Returns
        -------
        image_table - pd dataframe of metadata image table
        """
        import json
        try:
            with open(join(pth,fname)) as f:
                mddata = json.load(f)
        except:
            with open(join(pth,fname),'a') as f:
                f.write("}")
            with open(join(pth,fname)) as f:
                mddata = json.load(f)
            
            
        usecolslist = ['acq',  'Position', 'frame','Channel', 'Marker', 'Fluorophore', 'group', 
       'XY', 'Z', 'Zindex','Exposure','PixelSize', 'PlateType', 'TimestampFrame','TimestampImage', 'filename']
        image_table = pd.DataFrame(columns=usecolslist)
        mdsum = mddata['Summary']

        mdkeys = [key for key in mddata.keys() if key.startswith("Metadata")]

        for key in mdkeys:
            mdsing = mddata[key]
            framedata={'acq':mdsum['Prefix'],'Position':mdsing['PositionName'],'frame':mdsing['Frame'],'Channel':mdsum['ChNames'][mdsing['ChannelIndex']],'Marker':mdsum['ChNames'][mdsing['ChannelIndex']],'Fluorophore':mdsing['XLIGHT Emission Wheel-Label'],'group':mdsing['PositionName'],'XY':[mdsing['XPositionUm'],mdsing['YPositionUm']], 'Z':mdsing['ZPositionUm'], 'Zindex':mdsing['SliceIndex'],'Exposure':mdsing['Exposure-ms'] ,'PixelSize':mdsing['PixelSizeUm'], 'PlateType':'NA','TimestampFrame':mdsing['ReceivedTime'],'TimestampImage':mdsing['ReceivedTime'],'filename':mdsing['FileName']}
            image_table = image_table.append(framedata, sort=False,ignore_index=True)
    
        image_table['root_pth'] = image_table.filename
        
        
        image_table.filename = [join(pth, f.split(os.path.sep)[-1]) for f in image_table.filename]
        return image_table
    
    
    def _load_metadata_txt(self, pth, fname='Metadata.txt', delimiter='\t'):
        """
        Helper function to load a text metadata file.
        Returns
        -------
        image_table - pd dataframe of metadata image table
        """
        image_table = pd.read_csv(join(pth, fname), delimiter=delimiter)
        image_table['root_pth'] = image_table.filename
        image_table.filename = [join(pth, f.split(os.path.sep)[-1]) for f in image_table.filename]
        return image_table
    
    #functions for generic TIF loading!
    
    def _load_metadata_TIF_GUI(self,pth=''):
        GlobText = self._getPatternsFromPathGUI(pth=pth)
        box1 = self._getMappingFromGUI(GlobText)
        
    
    def _getPatternsFromPathGUI(self, pth=''):
        from ipywidgets import Button, Text, widgets, HBox, Layout
        from tkinter import Tk, filedialog
        from IPython.display import clear_output, display
        from oyLabImaging.Processing.generalutils import findregexp, findstem, extractFieldsByRegex
        out1 = widgets.Output()
        out2 = widgets.Output()
        out3 = widgets.Output()

        FolderText = Text(value='',placeholder='Enter path to image files',description='Path:',layout=Layout(width='70%', height='30px'))  
        
        GlobText = Text(value='',placeholder='*',description='Regular expression:',layout=Layout(width='70%', height='30px'), style={'description_width': 'initial'})    
        GlobText.flag=0
        def select_files(b):
            with out1:
                out1.clear_output()
                root = Tk()
                root.withdraw() # Hide the main window.
                root.call('wm', 'attributes', '.', '-topmost', True) # Raise the root to the top of all windows.
                b.dir = filedialog.askdirectory() # List of selected folder will be set button's file attribute.
                FolderText.value=b.dir+'/*'
                print(b.dir) # Print the list of files selected.

        fileselect = Button(description="Browse directory")
        fileselect.on_click(select_files)

        left_box = HBox([fileselect, FolderText])
        display(left_box)

        def on_value_change_path(change):
            with out1:
                out1.clear_output()
                print(change['new'])
            with out2:
                out2.clear_output()
                GlobText.fnames = fnamesFromPath(change['new'])
                if len(GlobText.fnames):
                    GlobText.value = findregexp(GlobText.fnames)
                    GlobText.globExp = GlobText.value
                    GlobText.patterns = extractFieldsByRegex(GlobText.globExp, GlobText.fnames)
                    GlobText.path = FolderText.value
                    GlobText.flag=1
                else:
                    GlobText.value = 'Empty File List!'
                    GlobText.flag=0
                print(*GlobText.fnames[0:min(5,len(GlobText.fnames))],sep = '\n')

        def fnamesFromPath(pth):
            import glob
            import os
            fnames = glob.glob(pth+'**/*.TIF',recursive=True)
            fnames.sort(key=os.path.getmtime)
            return fnames

        def on_value_change_glob(change):
            with out3:
                out3.clear_output()
                if len(GlobText.fnames):
                    GlobText.globExp = GlobText.value
                    GlobText.patterns = extractFieldsByRegex(GlobText.globExp, GlobText.fnames)
                else:
                    GlobText.value = 'Empty File List!'

        FolderText.observe(on_value_change_path, names='value')
        FolderText.value=pth

        GlobText.observe(on_value_change_glob, names='value')

        out1.append_stdout(pth)

        print('Files:')
        display(out2)

        display(GlobText)
        return(GlobText)

    def _getMappingFromGUI(self, GlobText):
        from ipywidgets import Button, Text, widgets, HBox, Layout
        from IPython.display import clear_output, display
        import itertools
        out1 = widgets.Output()
        out3 = widgets.Output()
        box1 = HBox()
        buttonDone = Button(description='Done',layout=Layout(width='25%', height='80px'),button_style='success',style=dict(
        font_size='48',
        font_weight='bold'))
        def change_traits(change):
            with out1:
                out1.clear_output()
                ordered_list_of_traits=[]
                for i in range(1, len(box1.children), 2):
                    ordered_list_of_traits.append(box1.children[i].value)
                box1.ordered_list_of_traits = ordered_list_of_traits
                print(ordered_list_of_traits)
                print(*GlobText.patterns[0:min(5,len(GlobText.fnames))],sep = '\n')


        def on_change(t):
            with out3:
                out3.clear_output()
                if GlobText.flag:
                    print('Found regular expression with '+str(len(GlobText.patterns[0]))+' :')

                    parts = GlobText.globExp.split('*')
                    options = ['Channel', 'Position', 'frame', 'Zindex', 'IGNORE']
                    layout = widgets.Layout(width='auto', height='40px') #set width and height

                    dddict = {}
                    for i in range(len(parts)-1):
                        # dynamically create key
                        key = parts[i]
                        # calculate value
                        value = [widgets.Label(parts[i]), widgets.Dropdown(options=options,value=options[i], layout=Layout(width='9%'))]
                        dddict[key] = value 
                    key = parts[-1]
                    value = [widgets.Label(parts[-1])]
                    dddict[key] = value 

                    ddlist = list(itertools.chain.from_iterable(dddict.values()))
                    box1.children = tuple(ddlist)
                    for dd in box1.children:
                        dd.observe(change_traits,names='value')
                    box1.children[1].value = 'frame'
                    box1.children[1].value = 'Channel'
                    display(box1)


        def on_click_done(b):
            b.disabled=True
            self._load_imagetable_Tiffs(box1.ordered_list_of_traits, GlobText.fnames, GlobText.patterns, GlobText.path)



        display(out3)
        if GlobText.flag:
            on_change(GlobText)
            

        box2 = HBox([out1, buttonDone])
        display(box2)

        GlobText.observe(on_change, names='value')
        buttonDone.on_click(on_click_done)

        return box1
    
    def _load_imagetable_Tiffs(self, ordered_list_of_traits, fnames, patterns, pth):
        """
        Helper function to load minimal metadata from tiffs.
        Returns
        -------
        image_table - pd dataframe of metadata image table
        """
        import pandas as pd
        from os.path import join, isdir


        traitdict={}
        for i,trait in enumerate(ordered_list_of_traits):
            if not trait=='IGNORE':
                key=trait
                value=[p[i] for p in patterns]
                traitdict[key]=value

        usecolslist = ['acq',  'Position', 'frame','Channel', 'Marker', 'Fluorophore', 'group', 
        'XY', 'Z', 'Zindex','Exposure','PixelSize', 'PlateType', 'TimestampFrame','TimestampImage', 'filename']
        image_table = pd.DataFrame(columns=usecolslist)


        image_table['filename']=fnames
        image_table['root_pth'] = image_table.filename

        #Default values
        image_table['acq']=pth
        image_table['XY']=[[0,0]]*len(fnames)
        image_table['Z']=0
        image_table['Zindex']=0
        image_table['Channel']=0
        image_table['Position']=0
        image_table['frame']=0
        image_table['PixelSize']=1
        
        
        #update whichever values we got from the names
        for trait in traitdict:
            image_table[trait]=traitdict[trait]
        
        try:
            image_table['frame']=[int(f) for f in image_table['frame']]
        except:
            pass
                    #todo : convert numerical strings to numbers


        image_table.filename = [join(pth, f.split(os.path.sep)[-1]) for f in image_table.filename]
        self.image_table = image_table
        
        


        
        
        
        
        
        
        
        
        
        
        
        
        

    #This is how everything gets added to a MD
    def append(self,framedata):
        """
        main method of adding data to metadata object
        Parameters
        ----------
        framedata - pd dataframe of metadata
        """
        self.image_table = self.image_table.append(framedata, sort=False,ignore_index=True)
        
        #framedata can be another MD dataframe 
        #framedata can also be a dict of column names and values: This will be handeled in scopex
        # framedata = {}
        # for attr in column_names:
        #     framedata.update(attr=v)
        


    
    
    
    # Save metadata in pickle format
    def pickle(self):
        """
        save metadata as a pickle file. Saves as 'metadata.pickle' in the metadata root path.
        """
        with open(join(self.base_pth,'metadata.pickle'), 'wb') as dbfile:
            tempfn = self.image_table['filename'].copy()
            self.image_table['filename'] = self.image_table['root_pth']
            del self.image_table['root_pth']
            pickle.dump(self, dbfile)
            self.image_table['root_pth'] = self.image_table['filename']
            self.image_table['filename'] = tempfn
            md_logger.info('saved metadata')

        
    def unpickle(self,pth,fname='*.pickle', delimiter='\t'):
        """
        load metadata from pickle file. 
        Parameters
        ----------
        pth : str - path to root folder where data is
        Returns
        -------
        image_table - pd dataframe of metadata image table
        """
        with open(join(pth,fname), 'rb') as dbfile:
            MD = pickle.load(dbfile)
            MD.image_table['root_pth'] = MD.image_table.filename
            MD.image_table.filename = [join(pth, f) for f in MD.image_table.filename]
            self._md_name = 'metadata.pickle'
            self.type = MD.type
            return MD.image_table
            
         
        
    
    def stkread(self, groupby='Position', sortby='TimestampFrame',
                finds_only=False, metadata=False, **kwargs):
        """
        Main interface of Metadata
        
        Parameters
        ----------
        groupby : str - all images with the same groupby field with be stacked
        sortby : str, list(str) - images in stks will be ordered by this(these) fields
        finds_only : Bool (default False) - lazy loading
        metadata : Bool (default False) - whether to return metadata of images
        
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
        
        image_subset_table = self.image_table
        # Filter images according to given criteria
        for attr in image_subset_table.columns:
            if attr in kwargs:
                if not isinstance(kwargs[attr], list):
                    kwargs[attr] = [kwargs[attr]]
                image_subset_table = image_subset_table[image_subset_table[attr].isin(kwargs[attr])]
           
        # Group images and sort them then extract filenames/indices of sorted images
        image_subset_table.sort_values(sortby, inplace=True)
        image_groups = image_subset_table.groupby(groupby)
        
        finds_output = {}
        mdata = {}
        for posname in image_groups.groups.keys():
            finds_output[posname] = image_subset_table.loc[image_groups.groups[posname]].index.values
            mdata[posname] = image_subset_table.loc[image_groups.groups[posname]]

        # Clunky block of code below allows getting filenames only, and handles returning 
        # dictionary if multiple groups present or ndarray only if single group
        if finds_only:
            if metadata:
                if len(mdata)==1:
                    mdata = mdata[posname]
                return finds_output, mdata
            else:
                return finds_output
        else: #Load and organize images
            if metadata:
                if len(mdata)==1:
                    mdata = mdata[posname]
                    return np.squeeze(self._open_file(finds_output,**kwargs)[posname]), mdata
                else:
                    return self._open_file(finds_output,**kwargs), mdata
            else:
                stk = self._open_file(finds_output,**kwargs) 
                if len(list(stk.keys()))==1:
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

     
    def _read_local(self, ind_dict, ffield=False,register=False, verbose=True,**kwargs):
        """
        Helper function to read list of files given an TIFF type metadata and an filename list 
        Load images into dictionary of stks.
        TODO - add crop at load
        """
   
        images_dict = {}
    
        for key, value in ind_dict.items():
            # key is groupby property value
            # value is list of filenames of images to be loaded as a stk
       
            imgs = []
            
            for img_idx, find in enumerate(value):
                fname = self.image_table.at[find,'filename']
                # Weird print style to print on same line
                if verbose:
                    sys.stdout.write("\r"+'opening file '+path.split(fname)[-1])
                    sys.stdout.flush() 
                #md_logger.info("\r"+'opening '+path.split(fname)[-1])
                
                #For speed: use PIL when loading a single image, imread when using stack
                im = Image.open(join(fname))
                try:
                    im.seek(1)
                    img = io.imread(join(fname))
                except:
                    img = np.array(im)
                im.close()

                
                if ffield:
                    img = self._doFlatFieldCorrection(img, find)
                if register:
                    img = self._register(img, find)
                #if it's a z-stack
                if img.ndim==3: 
                    img = img.transpose((1,2,0))
                
                imgs.append(img)
            
            # Best performance has most frequently indexed dimension first 
            images_dict[key] = np.array(imgs) / 2**16  
            if verbose:
                print('\nLoaded group {0} of images.'.format(key))
            
        return images_dict

    
    
    def _read_nd2(self, ind_dict, ffield=False,register=False, verbose=True,**kwargs):
        """
        Helper function to read list of files given an ND type metadata and an index list.
        Load images into dictionary of stks.
        """
        import nd2reader as nd2
        with nd2.ND2Reader(self.unique('filename')[0]) as nd2imgs:
          
            images_dict = {}
            for key, value in ind_dict.items():
                imgs = []
                for img_idx, find in enumerate(value):
                    # Weird print style to print on same line
                    if verbose:
                        sys.stdout.write("\r"+'opening index '+ str(find))
                        sys.stdout.flush()                

                    img = np.array(nd2imgs.parser.get_image(find))

                    if ffield:
                        img = self._doFlatFieldCorrection(img, find)
                    if register:
                        img = self._register(img, find)
                    #if it's a z-stack
                    if img.ndim==3: 
                        img = img.transpose((1,2,0))

                    imgs.append(img)

                # Best performance has most frequently indexed dimension first 
                images_dict[key] = np.array(imgs) / 2**16  
                if verbose:
                    print('\nLoaded group {0} of images.'.format(key))

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
        cameraoffset = 100./2**16
        bitdepth = 2.**16
        flt = flt.astype(np.float32) - cameraoffset
        flt = np.divide(flt, np.nanmean(flt.flatten()))
        
        img = np.divide((img-cameraoffset).astype(np.float32), flt+cameraoffset)
        flat_img = img.flatten()
        rand_subset = np.random.randint(0, high=len(flat_img), size=10000)
        flat_img = flat_img[rand_subset]
        flat_img = np.percentile(flat_img, 1)
        np.place(img, flt<0.05, flat_img)
        np.place(img, img<0, 0)
        np.place(img, img>bitdepth, bitdepth)
        return img
    
    
    
    def _register(self,img,find):
        """
        Perform image registration.
        
        Parameters
        ----------
        img : numpy.ndarray
            2D image of type integer
        find : file identifier in metadata table
        """
        from skimage import transform
        import cv2
       
        if 'driftTform' in self().columns:    
            dT = self.image_table.at[find, 'driftTform']
            if dT is None:
                warnings.warn("No drift correction found for position")
                return img
            dT = np.array(dT)
            if len(dT)==9:
                M = np.reshape(dT,(3,3)).transpose()
                #cv2/scikit and numpy index differently
                M[0,2], M[1,2] = M[1,2], M[0,2]
                imreturn = cv2.warpAffine((img*2**16).astype('float64'), M[:2], img.shape[::-1])
                return imreturn/2**16
                #return transform.warp(img, np.linalg.inv(M), output_shape=img.shape,preserve_range=True)
        else:
            warnings.warn("No drift correction found for experiment")
            return img
            

    # Calculate jitter/drift corrections   
    def CalculateDriftCorrection(self, Position=None,frames=None, ZsToLoad=[0], Channel='DeepBlue',threads=8,chunks=20, GPU=True):
        if GPU:
            try:
#                 self.CalculateDriftCorrectionGPU(Position=Position,frames=frames, ZsToLoad=ZsToLoad, Channel=Channel)
#             except:
                print('Trying GPU calculation in chunks of '+ str(chunks))
                self.CalculateDriftCorrectionGPUChunks(Position=Position,frames=frames, ZsToLoad=ZsToLoad, Channel=Channel,chunks=chunks)
            finally:
                print('No GPU or no CuPy. If you have a GPU, try installing CuPy')
                self.CalculateDriftCorrectionCPU(Position=Position,frames=frames, ZsToLoad=ZsToLoad, Channel=Channel,threads=threads)

        else:
            self.CalculateDriftCorrectionCPU(Position=Position,frames=frames, ZsToLoad=ZsToLoad, Channel=Channel,threads=8)


    
    
    def CalculateDriftCorrectionCPU(self, Position=None,frames=None, ZsToLoad=[0], Channel='DeepBlue',threads=8):
        """
        Calculate image registration (jitter correction) parameters and add to metadata.
        
        Parameters
        ----------
        Position : str or list of strings
        ZsToLoad : list of int - which Zs to load to calculate registration. If list, registration will be the average of the different zstack layers
        Channel : str, channel name to use for registration
            
        """
        
        #from scipy.signal import fftconvolve
        if Position is None:
            Position = self.posnames
        elif type(Position) is not list:
            Position = [Position]
        
        if frames is None:
            frames = self.frames
        elif type(frames) is not list:
            frames = [frames]
            
        assert Channel in self.channels, "%s isn't a channel, try %s" % (Channel, ', '.join(list(self.channels)))
        
        for pos in Position:
            from pyfftw.interfaces.numpy_fft import fft2, ifft2            

            DataPre = self.stkread(Position=pos, Channel=Channel, Zindex=ZsToLoad, frame=frames, register=False)
            print('\ncalculating drift correction for position ' + str(pos) + ' on CPU')
            DataPre = DataPre-np.mean(DataPre,axis=(1,2),keepdims=True)

            DataPost = DataPre[1:,:,:].transpose((1,2,0))
            DataPre = DataPre[:-1,:,:].transpose((1,2,0))
            #this is in prep for # Zs>1
            DataPre = np.reshape(DataPre,(DataPre.shape[0],DataPre.shape[1],len(ZsToLoad), len(frames)-1));
            DataPost = np.reshape(DataPost,(DataPost.shape[0],DataPost.shape[1],len(ZsToLoad), len(frames)-1));

            #calculate cross correlation
            DataPost = np.rot90(DataPost,axes=(0, 1),k=2)

            # So because of dumb licensing issues, fftconvolve can't use fftw but the slower fftpack. Python is wonderful. So we'll do it all by hand like neanderthals 
            img_fft_1 = fft2(DataPre,axes=(0,1),threads=threads)
            img_fft_2 = fft2(DataPost,axes=(0,1),threads=threads)
            imXcorr = np.abs(np.fft.ifftshift(ifft2(img_fft_1*img_fft_2,axes=(0,1),threads=threads),axes=(0,1)))

            #if more than 1 slice is calculated, look for mean shift
            imXcorrMeanZ = np.mean(imXcorr,axis=2)
            c = []
            for i in range(imXcorrMeanZ.shape[-1]):
                c.append(np.squeeze(imXcorrMeanZ[:,:,i]).argmax())

            d = np.transpose(np.unravel_index(c, np.squeeze(imXcorrMeanZ[:,:,0]).shape))-np.array(np.squeeze(imXcorrMeanZ[:,:,0]).shape)/2 + 1 #python indexing starts at 0
            D = np.insert(np.cumsum(d, axis=0), 0, [0,0], axis=0)
            
            if 'driftTform' not in self.image_table.columns:
                self.image_table['driftTform']=None

            for frmind, frame in enumerate(frames):
                inds = self.unique(Attr='index',Position=pos, frame=frame)
                for ind in inds:
                    self.image_table.at[ind, 'driftTform']=[1, 0, 0 , 0, 1, 0 , D[frmind,0], D[frmind,1], 1]
            print('calculated drift correction for position ' + str(pos))
        self.pickle()

        
    def CalculateDriftCorrectionGPU(self, Position=None,frames=None, ZsToLoad=[0], Channel='DeepBlue',threads=8):
        """
        Calculate image registration (jitter correction) parameters and add to metadata.
        
        Parameters
        ----------
        Position : str or list of strings
        ZsToLoad : list of int - which Zs to load to calculate registration. If list, registration will be the average of the different zstack layers
        Channel : str, channel name to use for registration
            
        """
        from cupy.fft import fft2, ifft2
        from cupy import _default_memory_pool, asarray
        
        #from scipy.signal import fftconvolve
        if Position is None:
            Position = self.posnames
        elif type(Position) is not list:
            Position = [Position]
        
        if frames is None:
            frames = self.frames
        elif type(frames) is not list:
            frames = [frames]
            
        assert Channel in self.channels, "%s isn't a channel, try %s" % (Channel, ', '.join(list(self.channels)))
        
        for pos in Position:

            DataPre = asarray(self.stkread(Position=pos, Channel=Channel, Zindex=ZsToLoad, frame=frames, register=False))
            print('\ncalculating drift correction for position ' + str(pos)+ ' on GPU')
            DataPre = self.stkread(Position=pos, Channel=Channel, Zindex=ZsToLoad, frame=fr, register=False)               
            DataPre = DataPre-np.mean(DataPre,axis=(1,2),keepdims=True)

            DataPost = DataPre[1:,:,:].transpose((1,2,0))
            DataPre = DataPre[:-1,:,:].transpose((1,2,0))
            #this is in prep for # Zs>1
            DataPre = np.reshape(DataPre,(DataPre.shape[0],DataPre.shape[1],len(ZsToLoad), len(fr)-1));
            DataPost = np.reshape(DataPost,(DataPost.shape[0],DataPost.shape[1],len(ZsToLoad), len(fr)-1));

            #calculate cross correlation
            DataPost = np.rot90(DataPost,axes=(0, 1),k=2)


            DataPre = asarray(DataPre)
            DataPost = asarray(DataPost)

            # So because of dumb licensing issues, fftconvolve can't use fftw but the slower fftpack. Python is wonderful. So we'll do it all by hand like neanderthals 
            img_fft_1 = fft2(DataPre,axes=(0,1))
            del DataPre
            _default_memory_pool.free_all_blocks()
            img_fft_2 = fft2(DataPost,axes=(0,1))
            del DataPost
            _default_memory_pool.free_all_blocks()
            imXcorr = np.abs(np.fft.ifftshift(ifft2(img_fft_1*img_fft_2,axes=(0,1)),axes=(0,1)))
            del img_fft_1
            _default_memory_pool.free_all_blocks()
            del img_fft_2
            _default_memory_pool.free_all_blocks()
            #if more than 1 slice is calculated, look for mean shift
            imXcorrMeanZ = np.mean(imXcorr,axis=2).get()
            del imXcorr
            c = []
            for i in range(imXcorrMeanZ.shape[-1]):
                c.append(np.squeeze(imXcorrMeanZ[:,:,i]).argmax())

            _default_memory_pool.free_all_blocks()
            
            d = np.transpose(np.unravel_index(c, np.squeeze(imXcorrMeanZ[:,:,0]).shape))-np.array(np.squeeze(imXcorrMeanZ[:,:,0]).shape)/2 + 1 #python indexing starts at 0
            D = np.insert(np.cumsum(d, axis=0), 0, [0,0], axis=0)
            
            if 'driftTform' not in self.image_table.columns:
                self.image_table['driftTform']=None

            for frmind, frame in enumerate(frames):
                inds = self.unique(Attr='index',Position=pos, frame=frame)
                for ind in inds:
                    self.image_table.at[ind, 'driftTform']=[1, 0, 0 , 0, 1, 0 , D[frmind,0], D[frmind,1], 1]
            print('calculated drift correction for position ' + str(pos))
        self.pickle()

    def CalculateDriftCorrectionGPUChunks(self, Position=None,frames=None, ZsToLoad=[0], Channel='DeepBlue',chunks=10):
        """
        Calculate image registration (jitter correction) parameters and add to metadata.
        
        Parameters
        ----------
        Position : str or list of strings
        ZsToLoad : list of int - which Zs to load to calculate registration. If list, registration will be the average of the different zstack layers
        Channel : str, channel name to use for registration
            
        """
        from cupy.fft import fft2, ifft2
        from cupy import get_default_memory_pool, asarray
        mempool = get_default_memory_pool()
        #from scipy.signal import fftconvolve
        if Position is None:
            Position = self.posnames
        elif type(Position) is not list:
            Position = [Position]
        
        if frames is None:
            frames = self.frames
        elif type(frames) is not list:
            frames = [frames]
            
        assert Channel in self.channels, "%s isn't a channel, try %s" % (Channel, ', '.join(list(self.channels)))
        
        def chunker_with_overlap(seq, size):
            return (seq[np.max((0,pos-1)):pos + size] for pos in range(0, len(seq), size))

        for pos in Position:
            ds = np.empty((0, 2), float)
            print('\ncalculating drift correction for position ' + str(pos)+ ' on GPU')
            for fr in chunker_with_overlap(frames,chunks):
                DataPre = self.stkread(Position=pos, Channel=Channel, Zindex=ZsToLoad, frame=fr, register=False)               
                DataPre = DataPre-np.mean(DataPre,axis=(1,2),keepdims=True)

                DataPost = DataPre[1:,:,:].transpose((1,2,0))
                DataPre = DataPre[:-1,:,:].transpose((1,2,0))
                #this is in prep for # Zs>1
                DataPre = np.reshape(DataPre,(DataPre.shape[0],DataPre.shape[1],len(ZsToLoad), len(fr)-1));
                DataPost = np.reshape(DataPost,(DataPost.shape[0],DataPost.shape[1],len(ZsToLoad), len(fr)-1));

                #calculate cross correlation
                DataPost = np.rot90(DataPost,axes=(0, 1),k=2)

                
                DataPre = asarray(DataPre)
                DataPost = asarray(DataPost)

                
                # So because of dumb licensing issues, fftconvolve can't use fftw but the slower fftpack. Python is wonderful. So we'll do it all by hand like neanderthals 
                img_fft_1 = fft2(DataPre,axes=(0,1))
                del DataPre
                mempool.free_all_blocks()
                img_fft_2 = fft2(DataPost,axes=(0,1))
                del DataPost
                mempool.free_all_blocks()
                SF = img_fft_1*img_fft_2
                del img_fft_1
                del img_fft_2
                mempool.free_all_blocks()
                imXcorr = np.abs(np.fft.ifftshift(ifft2(SF,axes=(0,1)),axes=(0,1)))
                del SF
                mempool.free_all_blocks()
                #if more than 1 slice is calculated, look for mean shift
                imXcorrMeanZ = np.mean(imXcorr,axis=2).get()
                del imXcorr
                mempool.free_all_blocks()
                c = []
                for i in range(imXcorrMeanZ.shape[-1]):
                    c.append(np.squeeze(imXcorrMeanZ[:,:,i]).argmax())
                
                mempool.free_all_blocks()

                d = np.transpose(np.unravel_index(c, np.squeeze(imXcorrMeanZ[:,:,0]).shape))-np.array(np.squeeze(imXcorrMeanZ[:,:,0]).shape)/2 + 1 #python indexing starts at 0
                del imXcorrMeanZ
                ds = np.concatenate((ds,d))
            D = np.insert(np.cumsum(ds, axis=0), 0, [0,0], axis=0)
        
            if 'driftTform' not in self.image_table.columns:
                self.image_table['driftTform']=None

            for frmind, frame in enumerate(frames):
                inds = self.unique(Attr='index',Position=pos, frame=frame)
                for ind in inds:
                    self.image_table.at[ind, 'driftTform']=[1, 0, 0 , 0, 1, 0 , D[frmind,0], D[frmind,1], 1]
            print('calculated drift correction for position ' + str(pos))