# FrameLbl module for microscopy data. 
# Loads, segments, and retrieves single cell data for a single position and a single timepoint
# AOY
import pandas as pd
import numpy as np
from skimage import measure
from oyLabImaging.Processing.generalutils import regionprops_to_df
from oyLabImaging.Processing.improcutils import segmentation
from oyLabImaging import Metadata
from oyLabImaging.Processing.improcutils import Zernike
from scipy import stats

class FrameLbl(object):
    """
    Class for data from a single frame (single timepoint, position, experiment, multi channel). Handles image segmentation.
    Parameters
    ----------
    MD : relevant metadata OR
    pth : str path to relevant metadata
    
    Attributes
    ----------
    frame :  int - frame number
    Pos : position name
    acq : acquisition name
    Zindex : Zindex
    
    These must specify a unique frame
    
    Segmentation parameters
    -----------------------
    NucChannel : ['DeepBlue'] str name of nuclear channel
    segment_type : {[watershed], 'cellpose_nuclei', 'cellpose_cytoplasm'...}
    CytoChannel : OPTIONAL
    periring : {[False], True} should it calculate a perinuclear ring?
    periringsize : [5] size of perinuclear ring in pixels
    **kwargs : specific args for segmentation function
    
    Returns
    -------
    FrameLbl instance with segmented cells
    
    Class properties
    ----------------
     'XY',
     'acq',
     'area',
     'area_um2',
     'centroid',
     'centroid_um',
     'channels',
     'density',
     'frame',
     'imagedims',
     'img',
     'link1in2',
     'maxint',
     'mean',
     'median',
     'minint',
     'ninetyint',
     'num',
     'posname',
     'pth',
     'regionprops',
     'weighted_centroid',
     'weighted_centroid_um']
    
    Class methods
    -------------
     'img'
     'scattershow'
    
    """
    def __init__(self, frame=None, MD=None ,pth=None, Pos=None, acq = None, Zindex=0 ,register=True ,periring=True, periringsize=5, NucChannel='DeepBlue',cytoplasm=False,CytoChannel='Yellow',zernike=False, segment_type='watershed', **kwargs):

        if pth is None and MD is not None:
            pth = MD.base_pth
            
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
        

        self._seg_fun = segmentation.segtype_to_segfun(segment_type)
        
        self.channels = MD.unique('Channel',Position=Pos, frame=frame) 
        self.acq = MD.unique('acq',Position=Pos, frame=frame)
        
        self.XY = MD().at[MD.unique('index',Position=Pos, frame=frame)[0],'XY']
        self._pixelsize = MD()['PixelSize'][0]

        
        Data = {};
        for ch in self.channels:
            Data[ch] = np.squeeze(MD.stkread(Channel=ch,frame=frame, Position=Pos, Zindex=Zindex,verbose=False))
            assert Data[ch].ndim==2, "channel/position/frame/Zindex did not return unique result" 
        
        self.imagedims = np.shape(Data[NucChannel]);

        nargs = self._seg_fun.__code__.co_argcount
        args = [self._seg_fun.__code__.co_varnames[i] for i in range(2, nargs)]
        defaults = list(self._seg_fun.__defaults__)
        input_dict = {args[i]: defaults[i] for i in range(0, nargs-2)} 
        input_dict = {**input_dict, **kwargs}
        
        self._seg_params = input_dict
        
        try:
            imgCyto=Data[CytoChannel]
        except:
            imgCyto=''
        
        L = self._seg_fun(img=Data[NucChannel],imgCyto=imgCyto,**kwargs)
        

            
        props = measure.regionprops(L,intensity_image=Data[NucChannel])   
        
        
        
        props_df = regionprops_to_df(props)
        props_df.drop(['mean_intensity', 'max_intensity', 'min_intensity'], axis=1,inplace=True)
        if zernike:
            
            L1 = [list(Zernike.coeff_fast(stats.zscore(r.intensity_image)))[1] for r in props]
            K1 = [list(Zernike.coeff_fast(stats.zscore(r.intensity_image)))[2] for r in props]  
            props_df['L'] = L1
            props_df['K'] = K1
                
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
            
            if zernike:
                c1 = [list(Zernike.coeff_fast(stats.zscore(r.intensity_image)))[0] for r in props_channel]  
                props_df['zernike_'+ch] = c1            
        
        
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
                #print('\nRegistered centroids')
            else:
                print('No drift correction found')
    
        self.regionprops = props_df
        
    
    def __call__(self):
        print('FrameLbl object for position ' + self.posname + ' at frame '+ str(self.frame) + '.')
        print('\nThe path to the experiment is: \n ' + self.pth)
        print('\n '+ str(self.num) + ' cells segmented using '+ self._seg_fun.__name__)        
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
    def index(self):
        return [i for i,r in enumerate(self.area)]
    
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
    
    
    
    #presentation stuff
    def img(self,Channel='DeepBlue',verbose=False, **kwargs):
        """
        Parameters
        ----------
        Channel : [DeepBlue] str or list of strings
        
        Returns
        -------
        Image at given frame and channels
        """
        from oyLabImaging import Metadata
        pth = self.pth
        MD = Metadata(pth,verbose=verbose)
        return MD.stkread(Channel=Channel,frame=self.frame, Position=self.posname, register=True,verbose=verbose, **kwargs)
    
    def scattershow(self,Channel='DeepBlue'):
        """
        Parameters
        ----------
        Channel : [DeepBlue] str or list of strings
        
        Returns
        -------
        Image at given frame and channels and overlaying points of segmented cells
        """
        import matplotlib.pyplot as plt
        img = self.img(Channel=Channel)
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.1)
        
        #fig = plt.figure(figsize=(12,15))
        #fig.add_axes([0.1,0.6,0.8,0.5])
        plt.gca().set_title('original')
        plt.gca().axis('off')
        #np.percentile(img.flatten(),1)
        plt.imshow(img, interpolation='nearest', cmap='gray',vmin=np.percentile(img.flatten(),1),vmax=np.percentile(img.flatten(),99))
        plt.gca().patch.set_alpha(0.5)
        plt.scatter(self.centroid[:,1],self.centroid[:,0], s=5,c=self.mean(Channel), alpha=1)