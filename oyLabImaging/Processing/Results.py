# Results module for microscopy data. 
# Aggregates all Pos labels for a specific experiment

# AOY
from oyLabImaging import Metadata
import numpy as np
from oyLabImaging.Processing import PosLbl
from os import walk, listdir, path
from os.path import join, isdir
import dill
import cloudpickle

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
    def __init__(self, MD=None ,pth=None, threads=10, **kwargs):
        
        
        if pth is None:
            if MD is not None:
                self.pth = MD.base_pth;
        else:
            self.pth = pth
        
        pth = self.pth
        
        if 'results.pickle' in listdir(pth):   
                r = self.load(pth ,fname='results.pickle')
                self.PosNames = r.PosNames
                self.channels = r.channels 
                self.acq = r.acq 
                self.frames = r.frames
                self.groups = r.groups
                self.PosLbls = r.PosLbls
                print('\nloaded results from pickle file')
        else:
            if MD is None:
                MD = Metadata(pth)

            if MD().empty:
                raise AssertionError('No metadata found in supplied path') 

            self.PosNames = MD.unique('Position')
            self.channels = MD.unique('Channel') 
            self.acq = MD.unique('acq') 
            self.frames = MD.unique('frame')
            self.groups = MD.unique('group')
            self.PosLbls = {}
      
    def __call__(self):
        print('Results object for path to experiment in path: \n ' + self.pth)
        print('\nAvailable channels are : ' + ', '.join(list(self.channels))+ '.')
        print('\nPositions already segmented are : ' + ', '.join(sorted([str(a) for a in self.PosLbls.keys()])))
        print('\nAvailable positions : ' + ', '.join(list([str(a) for a in self.PosNames]))+ '.')
        print('\nAvailable frames : ' + str(len(self.frames)) + '.')
        
        
    def setPosLbls(self, MD=None, groups=None, Position=None, **kwargs):
        """
        function to create PosLbl instances. 
        
        Parameters
        ----------
        Position - [All Positions] position name or list of position names

        """
        if MD is None:
            MD = Metadata(self.pth)         
        if groups is not None:
            assert(np.all(np.isin(groups,self.groups))), "some provided groups don't exist, try %s"  % ', '.join(list(self.groups))
            Position = MD.unique('Position', group=groups)
        if Position is None:
            Position = self.PosNames
            
        elif type(Position) is not list:
            Position = [Position]
            
        for p in Position:
            print('\nProcessing position ' + str(p))
            self.PosLbls.update({p : PosLbl(MD=MD, Pos=p, pth=MD.base_pth, **kwargs)})
        self.save()

    def calculate_tracks(self, Position=None, NucChannel='DeepBlue', **kwargs):
        """
        function to calculate tracks for a PosLbl instance. 
        
        Parameters
        ----------
        Position : [All Positions] position name or list of position names
        NucChannel : ['DeepBlue'] name of nuclear channel

        """
        pos=Position
        
        if pos==None:
            pos = list(self.PosLbls.keys())
        pos = pos if isinstance(pos, list) else [pos]
        assert any(elem in self.PosLbls.keys()  for elem in pos), str(pos) + ' not segmented yet'
        for p in pos:
            self.PosLbls[p].trackcells(NucChannel=NucChannel,**kwargs)
        self.save()
            
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
        assert pos in self.PosLbls.keys(), str(pos) +' not segmented yet'
        assert self.PosLbls[pos]._tracked, str(pos) +' not tracked yet'
        return self.PosLbls[pos].get_track
    
    def tracklist(self,pos=None):
        """
        Function to consolidate tracks from different positions
        Parameters
        ----------
        pos : [All positions] position name, list of position names
        
        Returns
        -------
        List of tracks in pos
        """
        if pos==None:
            pos = list(self.PosLbls.keys())
        pos = pos if isinstance(pos, list) else [pos]
        ts=[]
        for p in pos:
            t0 = self.tracks(p)
            ([ts.append(t0(i)) for i in np.arange(t0(0).numtracks)])
        return ts
    
    def show_tracks(self, pos, J=None,**kwargs):
        """
        Wrapper for PosLbl.plot_tracks
        Parameters
        ----------
        pos : position name
        J : track indices - plots all tracks if not provided
        Zindex : [0]
        
        
        Draws image stks with overlaying tracks in current napari viewer
 
        """

        assert pos in self.PosLbls.keys(), str(pos) +' not segmented yet'
        self.PosLbls[pos].plot_tracks(J=J,**kwargs)
    

    def show_points(self, pos, J=None,Channel=None,**kwargs):
        """
        Wrapper for PosLbl.plot_points
        Parameters
        ----------
        pos : position name
        Channel : [DeepBlue] str 
        Zindex : [0]
        
        Draws image stks with overlaying points in current napari viewer
        
  
        """
        if Channel not in self.channels:
            Channel = self.channels[0]
            print('showing channel '+ str(Channel))
        assert pos in self.PosLbls.keys(), str(pos) +' not segmented yet'
        self.PosLbls[pos].plot_points(Channel=Channel,**kwargs)
    
    def show_images(self, pos,Channel=None,**kwargs):
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
        print('showing channel '+ str(Channel))
        self.PosLbls[pos].plot_images(Channel=Channel,**kwargs)
    
    
    
    
    
    
    
    def save(self):
        """
        save results
        """
        with open(join(self.pth,'results.pickle'), 'wb') as dbfile:
            cloudpickle.dump(self, dbfile)
            print('saved results')
        
    def load(self,pth,fname='results.pickle'):
        """
        load results
        """
        with open(join(pth,fname), 'rb') as dbfile:
            r=dill.load(dbfile)
        return r
         