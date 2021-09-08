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
        print('\nPositions already segmented are : ' + ', '.join(sorted(self.PosLbls.keys())))
        print('\nAvailable positions : ' + ', '.join(list(self.PosNames))+ '.')
        print('\nAvailable frames : ' + str(len(self.frames)) + '.')
        
        
    def setPosLbls(self, MD=None, groups=None, Position=None, **kwargs):
        '''
        function to make Poslabels. Accepts group/position list. By default it will generate pos labels for all positions. override by specifying groups or Pos. groups override pos       
        '''
        if MD is None:
            MD = Metadata(self.pth)         
        if groups is not None:
            assert(np.all(np.isin(groups,self.groups))), "some provided groups don't exist, try %s"  % ', '.join(list(self.groups))
            Position = MD.unique('Position', group=groups)
        if Position is None:
            Position = self.PosNames
            
        elif type(Position) is str:
            Position = [Position]
            
        for p in Position:
            print('\nProcessing position ' + str(p))
            self.PosLbls.update({p : PosLbl(MD=MD, Pos=p, pth=MD.base_pth, **kwargs)})
        self.save()

    def calculate_tracks(self, pos=None, NucChannel='DeepBlue', **kwargs):
        if pos==None:
            pos = list(self.PosLbls.keys())
        pos = pos if isinstance(pos, list) else [pos]
        assert any(elem in self.PosLbls.keys()  for elem in pos), str(pos) + ' not segmented yet'
        for p in pos:
            self.PosLbls[p].trackcells(NucChannel=NucChannel,**kwargs)
        self.save()
            
    def tracks(self, pos):
        assert pos in self.PosLbls.keys(), str(pos) +' not segmented yet'
        assert self.PosLbls[pos]._tracked, str(pos) +' not tracked yet'
        return self.PosLbls[pos].get_track
    
    def show_tracks(self, pos, J=None,Channel='DeepBlue',**kwargs):
        assert pos in self.PosLbls.keys(), str(pos) +' not segmented yet'
        self.PosLbls[pos].plot_tracks(J=J,Channel=Channel,**kwargs)
    

    def show_points(self, pos, J=None,Channel='DeepBlue',**kwargs):
        assert pos in self.PosLbls.keys(), str(pos) +' not segmented yet'
        self.PosLbls[pos].plot_points(Channel=Channel,**kwargs)
    
    
    
    
    
    
    
    def save(self):
        with open(join(self.pth,'results.pickle'), 'wb') as dbfile:
            cloudpickle.dump(self, dbfile)
            print('saved results')
        
    def load(self,pth,fname='results.pickle'):
        with open(join(pth,fname), 'rb') as dbfile:
            r=dill.load(dbfile)
        return r
         