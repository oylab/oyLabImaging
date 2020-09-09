# FrameLbl module for microscopy data. 
# Loads, segments, and retrieves single cell data for a single position and a single timepoint
# AOY

import pandas as pd
import numpy as np
from skimage import measure
from oyLabCode.Processing.generalutils import regionprops_to_df
from oyLabCode.Processing.improcutils import segmentation
from oyLabCode import Metadata

class FrameLbl(object):
    def __init__(self, frame=None, MD=None ,pth=None, Pos=None, acq = None, register=False ,periring=False, periringsize=5, NucChannel='DeepBlue',cytoplasm=False,CytoChannel='Yellow', segment_type='watershed', **kwargs):

        if any([Pos is None, pth is None, frame is None]):
            raise ValueError('Please input path, position, and frame')
        
        self.pth = pth
        
        if MD is None:
            MD = Metadata(pth)
        
        if MD().empty:
            raise AssertionError('No metadata found in supplied path') 
                
        if Pos not in MD.posnames:
            raise AssertionError('Position does not exist in dataset')  
        self.posname = Pos
        
        if frame not in MD.frames:
            raise AssertionError('Frame does not exist in dataset')
        self.frame = frame
        
        
        if segment_type=='watershed':
            self._seg_fun=segmentation._segment_nuclei_watershed
        elif segment_type=='cellpose_nuclei':
            self._seg_fun=segmentation._segment_nuclei_cellpose

        
        self.channels = MD.unique('Channel',Position=Pos, frame=frame) 
        self.acq = MD.unique('acq',Position=Pos, frame=frame)
        
        self.XY = MD().at[MD.unique('index',Position=Pos, frame=frame)[0],'XY']
        self._pixelsize = MD()['PixelSize'][0]

        
        Data = {};
        for ch in self.channels:
            Data[ch] = np.squeeze(MD.stkread(Channel=ch,frame=frame, Position=Pos))
            assert Data[ch].ndim==2, "channel/position/frame did not return unique result" 
        
        self.imagedims = np.shape(Data[NucChannel]);

        L = self._seg_fun(Data[NucChannel],**kwargs)
        

            
        props = measure.regionprops(L,intensity_image=Data['DeepBlue'])     
        props_df = regionprops_to_df(props)
        props_df.drop(['mean_intensity', 'max_intensity', 'min_intensity'], axis=1,inplace=True)
        
        for ch in self.channels:
            props_channel = measure.regionprops(L,intensity_image=Data[ch])
            mean_channel = [r.mean_intensity for r in props_channel]
            max_channel = [r.max_intensity for r in props_channel]
            min_channel = [r.min_intensity for r in props_channel]
            Ninty_channel = [np.percentile(r.intensity_image,90) for r in props_channel]
            median_channel = [np.median(r.intensity_image) for r in props_channel]

            props_df['mean_'+ch] = mean_channel
            props_df['max_'+ch] = max_channel
            props_df['min_'+ch] = min_channel
            props_df['90th_'+ch] = Ninty_channel
            props_df['median_'+ch] = median_channel
        
        
        if periring:
            from skimage.morphology import disk, dilation
            se = disk(5)
            Lperi = dilation(L, se)-L
            
            for ch in self.channels:
                props_channel = measure.regionprops(Lperi,intensity_image=Data[ch])
                mean_channel = [r.mean_intensity for r in props_channel]
                max_channel = [r.max_intensity for r in props_channel]
                min_channel = [r.min_intensity for r in props_channel]
                Ninty_channel = [np.percentile(r.intensity_image,90) for r in props_channel]
                median_channel = [np.median(r.intensity_image) for r in props_channel]

                props_df['mean_'+ch + '_periring'] = mean_channel
                props_df['max_'+ch + '_periring'] = max_channel
                props_df['min_'+ch + '_periring'] = min_channel
                props_df['90th_'+ch + '_periring'] = Ninty_channel
                props_df['median_'+ch + '_periring'] = median_channel
        
        if cytoplasm:
            pass

        if register:
            ind = MD.unique('index', Position=Pos, frame=frame, Channel=NucChannel)
            Tforms = MD().at[ind[0],'driftTform']
            if Tforms is not None:
                for i in np.arange(props_df.index.size):
                    props_df.at[i,'centroid'] = tuple(np.add(props_df.at[i,'centroid'],Tforms[6:8]))
                    props_df.at[i,'weighted_centroid'] = tuple(np.add(props_df.at[i,'weighted_centroid'],Tforms[6:8]))
                print('Registered centroids')
            else:
                print('No drift correction found')
    
        self.regionprops = props_df
        
     #   Link21Mat
    
    def __call__(self):
        print('FrameLbl object for position ' + self.posname + ' at frame '+ str(self.frame) + '.')
        print('\nThe path to the experiment is: \n ' + self.pth)
        print('\nAvailable channels are : ' + ', '.join(list(self.channels))+ '.')
    
    
    @property
    def num(self):
        return self.regionprops.index.size
    
    @property
    def centroid(self):     
        return np.reshape(np.concatenate(self.regionprops['centroid']),(-1,2))
      
    @property
    def weighted_centroid(self):
        return np.reshape(np.concatenate(self.regionprops['weighted_centroid']),(-1,2))
    
    @property
    def area(self):
        return self.regionprops['area']
    
    @property
    def centroid_um(self):     
        return self.XY + self._pixelsize*self.centroid
      
    @property
    def weighted_centroid_um(self):
        return self.XY + self._pixelsize*self.weighted_centroid
    
    @property
    def area_um2(self):
        return self.area*self._pixelsize**2

    
    def mean(self,ch, periring=False):
        assert ch in self.channels, "%s isn't a channel, try %s" % (ch, ', '.join(list(self.channels)))
        if periring:
            return self.regionprops['mean_' + ch + '_periring']
        else:
            return self.regionprops['mean_' + ch]
        
    def median(self,ch, periring=False):
        assert ch in self.channels, "%s isn't a channel, try %s" % (ch, ', '.join(list(self.channels)))
        if periring:
            return self.regionprops['median_' + ch + '_periring']
        else:
            return self.regionprops['median_' + ch]
    
    def minint(self,ch, periring=False):
        assert ch in self.channels, "%s isn't a channel, try %s" % (ch, ', '.join(list(self.channels)))
        if periring:
            return self.regionprops['min_' + ch + '_periring']
        else:
            return self.regionprops['min_' + ch]
    
    def maxint(self,ch, periring=False):
        assert ch in self.channels, "%s isn't a channel, try %s" % (ch, ', '.join(list(self.channels)))
        if periring:
            return self.regionprops['max_' + ch + '_periring']
        else:
            return self.regionprops['max_' + ch]
    
    def ninetyint(self,ch, periring=False):
        assert ch in self.channels, "%s isn't a channel, try %s" % (ch, ', '.join(list(self.channels)))
        if periring:
            return self.regionprops['90th_' + ch + '_periring']
        else:
            return self.regionprops['90th_' + ch]
    
    @property
    def density(self):
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=11, algorithm='ball_tree').fit(self.weighted_centroid)
        distances,_   = nbrs.kneighbors(self.weighted_centroid)
        return 1./np.mean(distances[:,1:],axis=1)