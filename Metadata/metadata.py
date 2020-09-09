# Metadata module for microscopy data
# AOY

from os import walk, listdir, path
from os.path import join, isdir
import sys

import pandas as pd
import numpy as np
import dill as pickle
from ast import literal_eval
import warnings 
from PIL import Image


usecolslist = ['acq',  'Position', 'frame','Channel', 'Marker', 'Fluorophore', 'group', 
       'XY', 'Z', 'Zindex','Exposure','PixelSize', 'PlateType', 'TimestampFrame','TimestampImage', 'filename']


from skimage import img_as_float, img_as_uint, io

#Metadata stores all relevant info about an imaging experiment. 

class Metadata(object):
    
    #constractor: first, look for Metadata.pickle. If can't find, look for it in dirs. Then try .txt (back compatible). 
    def __init__(self, pth='', load_type='local'):       
        self.base_pth = pth
        
        #start with an empty MD
        self.image_table = pd.DataFrame(columns=usecolslist)
        # short circuit recursive search for metadatas if present in the top directory of 
        # the supplied pth.
        if self.base_pth:
            if 'Metadata.pickle' in listdir(pth):   
                self.append(self.unpickle(pth=join(self.base_pth) ,fname='Metadata.pickle'))
                print('loaded Metadata from pickle file')
            else:
                #if there is no MD in the folder, look at subfolders and append all 
                for subdir, curdir, filez in walk(self.base_pth):
                    for f in filez:
                        if f=='Metadata.pickle':
                            self.append(self.unpickle(pth=join(subdir),fname=f))
                            print('loaded '+join(subdir,f)+' from pickle file')
            #if there's a basepath but no pickle, it might have a txt file
            if self.image_table.empty:
                if 'Metadata.txt' in listdir(pth):
                    self.append(self.load_metadata_txt(join(pth), fname='Metadata.txt'))
                    print('loaded Metadata from txt file')

                #if there is no MD.txt in the folder, look at subfolders and append all 
                else:
                    for subdir, curdir, filez in walk(pth):
                        for f in filez:
                            if f=='Metadata.txt':
                                self.append(self.load_metadata_txt(join(pth, subdir), fname=f))
                                print('loaded '+join(subdir,f)+' from txt file')
                
                # Handle columns that don't import from text well
                try:
                    self.convert_data('XY', float)
                except Exception as e:
                    self.image_table['XY'] = [literal_eval(i) for i in self.image_table['XY']]
        # Future compability for different load types (e.g. remote vs local)
        if load_type=='local':
            self._open_file=self._read_local
        elif load_type=='google_cloud':
            raise NotImplementedError("google_cloud loading is not implemented.")


                
               


            

    def __call__(self):
        return self.image_table
          
    @property
    def posnames(self):
        return self().Position.unique()

    @property
    def frames(self):
        return list(self().frame.unique())
      
    @property
    def channels(self):
        return self().Channel.unique()
    
    @property
    def Zindexes(self):
        return self().Zindex.unique()
    
    @property
    def acqnames(self):
        return self().acq.unique()
    
    @property
    def groups(self):
        return self().group.unique()
    
    def unique(self, Attr=None ,**kwargs):
        for key, value in kwargs.items():
            if not isinstance(value, list):
                kwargs[key] = [value]
        image_subset_table = self.image_table
        # Filter images according to some criteria
        for attr in image_subset_table.columns:
            if attr in kwargs:
                image_subset_table = image_subset_table[image_subset_table[attr].isin(kwargs[attr])]
            
        if Attr is None:
            return image_subset_table.size
        elif Attr in image_subset_table.columns:  
            return image_subset_table[Attr].unique()
        elif Attr is 'index':
            return image_subset_table.index
        else:
            return None 
        
    
    
    def convert_data(self, column, dtype, isnan=np.nan):
        converted = []
        arr = self.image_table[column].values
        for i in arr:
            if isinstance(i, str):
                i = np.array(list(map(dtype, i.split())))
                converted.append(i)
            else:
                converted.append(i)
        self.image_table[column] = converted

    #backwards compatability
    def load_metadata_txt(self, pth, fname='Metadata.txt', delimiter='\t'):
        """
        Helper function to load a text metadata file.
        """
        image_table = pd.read_csv(join(pth, fname), delimiter=delimiter)
        image_table['root_pth'] = image_table.filename
        image_table.filename = [join(pth, f) for f in image_table.filename]
        return image_table
    
    
    #This is how everything gets added to a MD
    def append(self,framedata):
        self.image_table = self.image_table.append(framedata, sort=False,ignore_index=True)
        
        #framedata can be another MD dataframe 
        #framedata can also be a dict of column names and values: This will be handeled in scope
        #    str = "{"
        #    for attr in column_names:
        #        str = str + "'" + attr  + "':[" + '**value**' +"]  ,"
        #    str = str[:-1]+"}"
        #    framedata = eval(str)
        
        #OR:
        # framedata = {}
        # for attr in column_names:
        #     framedata.update(attr=v)
        
        

    
    def pickle(self):
        with open(join(self.base_pth,'Metadata.pickle'), 'wb') as dbfile:
            tempfn = self.image_table['filename'].copy()
            self.image_table['filename'] = self.image_table['root_pth']
            del self.image_table['root_pth']
            pickle.dump(self, dbfile)
            self.image_table['root_pth'] = self.image_table['filename']
            self.image_table['filename'] = tempfn
            print('saved metadata')
        
    def unpickle(self,pth,fname='Metadata.pickle'):
        with open(join(pth,fname), 'rb') as dbfile:
            MD = pickle.load(dbfile)
            MD.image_table['root_pth'] = MD.image_table.filename
            MD.image_table.filename = [join(pth, f) for f in MD.image_table.filename]
            return MD.image_table
            
        
      
    def stkread(self, groupby='Position', sortby='TimestampFrame',
                fnames_only=False, metadata=False, **kwargs):
        """
        Main interface of Metadata
        
        Parameters
        ----------
        groupby : str - all images with the same groupby field with be stacked
        sortby : str, list(str) - images in stks will be ordered by this(these) fields
        fnames_only : Bool (default False) - lazy loading
        metadata : Bool (default False) - whether to return metadata of images
        
        kwargs : Property Value pairs to subset images (see below)
        
        Returns
        -------
        stk of images if only one value of the groupby_value
        dictionary (groupby_value : stk) if more than one groupby_value
        stk/dict, metadata table if metadata=True
        fnames if fnames_only true
        fnames, metadata table if fnames_only and metadata
        
        Implemented kwargs
        ------------------
        Position : str, list(str)
        Channel : str, list(str)
        Zindex : int, list(int)
        acq : str, list(str)
        """
        
        
        #for key, value in kwargs.items():
        #    if not isinstance(value, list):
        #        kwargs[key] = [value]
        image_subset_table = self.image_table
        # Filter images according to some criteria
        for attr in image_subset_table.columns:
            if attr in kwargs:
                if not isinstance(kwargs[attr], list):
                    kwargs[attr] = [kwargs[attr]]
                image_subset_table = image_subset_table[image_subset_table[attr].isin(kwargs[attr])]
           
        # Group images and sort them then extract filenames of sorted images
        image_subset_table.sort_values(sortby, inplace=True)
        image_groups = image_subset_table.groupby(groupby)
        
        fnames_output = {}
        mdata = {}
        for posname in image_groups.groups.keys():
            fnames_output[posname] = image_subset_table.loc[image_groups.groups[posname]].filename.values
            mdata[posname] = image_subset_table.loc[image_groups.groups[posname]]

        # Clunky block of code below allows getting filenames only, and handles returning 
        # dictionary if multiple groups present or ndarray only if single group
        if fnames_only:
            if metadata:
                if len(mdata)==1:
                    mdata = mdata[posname]
                return fnames_output, mdata
            else:
                return fnames_output
        else:
            if metadata:
                if len(mdata)==1:
                    mdata = mdata[posname]
                    return np.squeeze(self._open_file(fnames_output,**kwargs)[posname]), mdata
                else:
                    return self._open_file(fnames_output,**kwargs), mdata
            else:
                stk = self._open_file(fnames_output,**kwargs) 
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
        
    def _read_local(self, filename_dict, ffield=False,register=False, verbose=False,**kwargs):
        """
        Load images into dictionary of stks.
        """

            
        images_dict = {}
        for key, value in filename_dict.items():
            # key is groupby property value
            # value is list of filenames of images to be loaded as a stk

            
            imgs = []
            
            for img_idx, fname in enumerate(value):
                # Weird print style to print on same line
                sys.stdout.write("\r"+'opening '+path.split(fname)[-1])
                sys.stdout.flush()                
                
                #For speed: use PIL when loading a single image, imread when using stack
                im = Image.open(join(fname))
                try:
                    im.seek(1)
                    img = io.imread(join(fname))
                except:
                    img = np.array(im)
                im.close()
                
                
                
                if ffield:
                    img = self.doFlatFieldCorrection(img, fname)
                if register:
                    img = self.register(img, fname)
                #if it's a z-stack
                if img.ndim==3: 
                    img = img.transpose((1,2,0))
                
                imgs.append(img)
            
            # Best performance has most frequently indexed dimension first 
            images_dict[key] = np.array(imgs) / 2**16  
            if verbose:
                print('Loaded {0} group of images.'.format(key))
            
        return images_dict



        
    def doFlatfieldCorrection(self, img, flt, **kwargs):
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
    
    
    
    def register(self,img,fname):
        #from skimage import transform
        import cv2
       
        if 'driftTform' in self().columns:        
            dT = np.array(self()['driftTform'][self.unique('index',filename=fname)])
            if dT[0]==None:
                warnings.warn("No drift correction found for position")
                return img
            elif len(dT[0])==9:
                M = np.reshape(dT[0],(3,3)).transpose()
                
                #cv2/scikit and numpy index differently
                M[0,2], M[1,2] = M[1,2], M[0,2]
                return cv2.warpAffine(img, M[:2], img.shape[::-1])
                #return transform.warp(img, np.linalg.inv(M), output_shape=img.shape,preserve_range=True)
        else:
            warnings.warn("No drift correction found for experiment")
            return img
            

    def CalculateDriftCorrection(self, pos, ZsToLoad=[1]):
        #from scipy.signal import fftconvolve
        from pyfftw.interfaces.numpy_fft import rfft2, irfft2
    
        frames = self.frames

        DataPre = self.stkread(Position=pos, Channel='DeepBlue')
        print('\ncalculating drift correction for position ' + pos)
        DataPre = DataPre-np.mean(DataPre,axis=(1,2),keepdims=True)
                                  
        DataPost = DataPre[1:,:,:].transpose((1,2,0))
        DataPre = DataPre[:-1,:,:].transpose((1,2,0))

        #this is in prep for # Zs>1
        DataPre = np.reshape(DataPre,(DataPre.shape[0],DataPre.shape[1],len(ZsToLoad), len(frames)-1));
        DataPost = np.reshape(DataPost,(DataPost.shape[0],DataPost.shape[1],len(ZsToLoad), len(frames)-1));

        #calculate cross correlation
        
        #imXcorr = fftconvolve((DataPre-np.mean(DataPre,axis=(0,1),keepdims=True)), np.rot90(DataPost-np.mean(DataPost,axis=(0,1),keepdims=True),k=2), mode='same', axes=[0,1])

        DataPost = np.rot90(DataPost,axes=(0, 1),k=2)
        
        # So because of dumb licensing issues, fftconvolve can't use fftw but the slower fftpack. Python is wonderful. So we'll do it all by hand like neanderthals 
        imXcorr = np.zeros_like(DataPre)
        for i in np.arange(DataPre.shape[-1]):
            img_fft_1 = rfft2(DataPre[:,:,:,i],axes=(0,1),threads=8)
            img_fft_2 = rfft2(DataPost[:,:,:,i],axes=(0,1),threads=8)
            imXcorr[:,:,:,i] = np.abs(np.fft.ifftshift(irfft2(img_fft_1*img_fft_2,axes=(0,1),threads=8)))
        
        
        
        #if more than 1 slice is calculated, look for mean shift
        imXcorrMeanZ = np.mean(imXcorr,axis=2)
        c = []
        for i in range(imXcorrMeanZ.shape[-1]):
            c.append(np.squeeze(imXcorrMeanZ[:,:,i]).argmax())

        d = np.transpose(np.unravel_index(c, np.squeeze(imXcorrMeanZ[:,:,0]).shape))-np.array(np.squeeze(imXcorrMeanZ[:,:,0]).shape)/2
        D = np.insert(np.cumsum(d, axis=0), 0, [0,0], axis=0)

        if 'driftTform' not in self.image_table.columns:
            self.image_table['driftTform']=None

        for frame in self.frames:
            inds = self.unique(Attr='index',Position=pos, frame=frame)
            for ind in inds:
                self.image_table.at[ind, 'driftTform']=[1, 0, 0 , 0, 1, 0 , D[frame-1,0], D[frame-1,1], 1]
        print('\ncalculated drift correction for position ' + pos)
        self.pickle()
    


from numba import jit
@jit(nopython = True)
def DownScale(imgin): #use 2x downscaling for scrol speed   
        #imgout = trans.downscale_local_mean(imgin,(Sc, Sc))
    imgout = (imgin[0::2,0::2]+imgin[1::2,0::2]+imgin[0::2,1::2]+imgin[1::2,1::2])/4
    return imgout

def stkshow(data):
    from pyqtgraph.Qt import QtCore, QtGui
    import pyqtgraph as pg
    import sys
    import skimage.transform as trans

    
    # determine if you need to start a Qt app. 
    # If running from Spyder, the answer is a no.
    # From cmd, yes. From nb, not sure actually.
    if not QtGui.QApplication.instance():
        app = QtGui.QApplication([])
    else:
        app = QtGui.QApplication.instance()
        
    ## Create window with ImageView widget
    win = QtGui.QMainWindow()
    win.resize(680,680)
    imv = pg.ImageView()
    win.setCentralWidget(imv)
    win.show()
    win.setWindowTitle('Fetching image stack...')
    
    
    
    
    resizeflg = 0;
    maxxysize = 800;
    maxdataxySc = np.floor(max(data.shape[0],data.shape[1])/maxxysize).astype('int')
    if maxdataxySc>1:
        resizeflg = 1;

    if len(data.shape)==4:#RGB assume xytc
        if data.shape[3]==3 or data.shape[3]==4:
            if resizeflg:
                dataRs = np.zeros((np.ceil(data.shape/np.array([maxdataxySc,maxdataxySc,1,1]))).astype('int'),dtype = 'uint16')    
                for i in range(0,data.shape[2]):
                    for j in range(0,data.shape[3]):
                        dataRs[:,:,i,j] = DownScale(data[:,:,i,j])
                dataRs = dataRs.transpose((2,0,1,3))
            else:
                dataRs = data;
                dataRs = dataRs.transpose((2,0,1,3))
        else:
            sys.exit('color channel needs to be RGB or RGBA')
    elif len(data.shape)==3:
        if resizeflg:
            dataRs = np.zeros((np.ceil(data.shape/np.array([maxdataxySc,maxdataxySc,1]))).astype('int'),dtype = 'uint16')    
            for i in range(0,data.shape[2]):
                dataRs[:,:,i] = DownScale(data[:,:,i])
            dataRs = dataRs.transpose([2,0,1])
        else:
            dataRs = data;
            dataRs = dataRs.transpose([2,0,1])
                
    elif len(data.shape)==2:
        if resizeflg:
            dataRs = np.zeros((np.ceil(data.shape/np.array([maxdataxySc,maxdataxySc]))).astype('int'),dtype = 'uint16')
            dataRs = DownScale(data)
        else:
            dataRs = data;
    else:
        print('Data must be 2D image or 3D image stack')
    

    
    # Interpret image data as row-major instead of col-major
    pg.setConfigOptions(imageAxisOrder='row-major')
    

    win.setWindowTitle('Stack')
    
    ## Display the data and assign each frame a 
    imv.setImage(dataRs)#, xvals=np.linspace(1., dataRs.shape[0], dataRs.shape[0]))

    ##must return the window to keep it open
    return win
    ## Start Qt event loop unless running in interactive mode.
    if __name__ == '__main__':
        import sys
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()
