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
        self.type = self._determine_metadata_type(pth)
        self.image_table = pd.DataFrame(columns=usecolslist)

        if self.type==None:
            return
        #start with an empty MD
        # short circuit recursive search for metadatas if present in the top directory of 
        # the supplied pth.
        
        
        
        if self.base_pth:
            if self.type=='PICKLE':
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

            elif self.type=='TXT':
                if 'Metadata.txt' in listdir(pth):
                    self.append(self.load_metadata_txt(join(pth), fname='Metadata.txt'))
                    print('loaded matlab Metadata from txt file')

                #if there is no Metadata.txt in the folder, look at subfolders and append all 
                else:
                    for subdir, curdir, filez in walk(pth):
                        for f in filez:
                            if f=='Metadata.txt':
                                self.append(self.load_metadata_txt(join(pth, subdir), fname=f))
                                print('loaded '+join(subdir,f)+' from txt file')
            elif self.type=='ND2':
                self.append(self.load_metadata_nd(pth))
                print('loaded metadata from nikon nd2 file')
                    
            elif self.type=='MM':
                if 'metadata.txt' in listdir(pth):
                    self.append(self.load_metadata_MM(join(pth), fname='Metadata.txt'))
                    print('loaded micromanager Metadata from txt file')

                #if there is no Metadata.txt in the folder, look at subfolders and append all 
                else:
                    for subdir, curdir, filez in walk(pth):
                        for f in filez:
                            if f=='metadata.txt':
                                self.append(self.load_metadata_MM(join(pth, subdir), fname=f))
                                print('loaded micromanager '+join(subdir,f)+' from txt file')
                
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
        elif Attr=='index':
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


        
        
        
        
        
    def _determine_metadata_type(self, pth):
        from os import path, walk
        fname, fext = path.splitext(pth)
        if path.isdir(pth):
            for subdir, curdir, filez in walk(pth):
                for f in filez:
                    fname, fext = path.splitext(f)
                    if fext=='.nd2':
                        return 'ND2'
                    if fext=='.pickle':
                        return 'PICKLE'
                    if fext=='.txt':
                        if fname=='metadata':
                            return 'MM'
                        if fname=='Metadata':
                            return 'TXT'
        else:
            fname, fext = path.splitext(pth)
            if fext=='.nd2':
                return 'ND2'
            if fext=='.pickle':
                return 'PICKLE'
            if fext=='.txt':
                if fname=='metadata':
                    return 'MM'
                if fname=='Metadata':
                    return 'TXT'     
        return None

    
    
    
    
    
        
    
    def load_metadata_nd(self, pth, fname='*.nd', delimiter='\t'):
        """
        Helper function to load a micromanager txt metadata file.
        """
        import nd2reader as nd2
        import pandas as pd
        from os.path import sep

        
        usecolslist = ['acq',  'Position', 'frame','Channel', 'XY', 'Z', 
                       'Zindex','Exposure','PixelSize', 'TimestampFrame','TimestampImage', 'filename']
        image_table = pd.DataFrame(columns=usecolslist)

        acq = pth.split(sep)[-1]
        with nd2.ND2Reader(pth) as imgs:
            Ninds = imgs.metadata['total_images_per_channel']*len(imgs.metadata['channels'])
            frames = imgs.metadata['frames']
            imgsPerFrame = Ninds/len(frames)

            XY = np.column_stack((np.array(imgs.parser._raw_metadata.x_data),np.array(imgs.parser._raw_metadata.y_data)))
            Zpos = imgs.metadata['z_coordinates']
            Zind = imgs.metadata['z_levels']
            pixsize = imgs.metadata['pixel_microns']
            acq = pth.split(sep)[-1]
            for i in np.arange(Ninds):
                frame = int(i/imgsPerFrame)
                xy = XY[frame,]
                z = Zpos[frame]
                zind = imgs.parser.calculate_image_properties(i)[2]
                chan = imgs.parser.calculate_image_properties(i)[1]
                pos = imgs.parser.calculate_image_properties(i)[0]
                exptime = imgs.parser._raw_metadata.camera_exposure_time[frame]
                framedata={'acq':acq,'Position':pos,'frame':frame,'Channel':chan,'XY':list(xy), 'Z':z, 'Zindex':zind,'Exposure':exptime ,'PixelSize':pixsize,'TimestampFrame':imgs.timesteps[frame],'TimestampImage':imgs.timesteps[frame],'filename':acq}
                image_table = image_table.append(framedata, sort=False,ignore_index=True)

            image_table['root_pth'] = image_table.filename
            image_table.filename = [join(pth, f.split('/')[-1]) for f in image_table.filename]
            return image_table
    

    
    def load_metadata_MM(self, pth, fname='metadata.txt', delimiter='\t'):
        """
        Helper function to load a micromanager txt metadata file.
        """
        import json
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
        
        
        image_table.filename = [join(pth, f.split('/')[-1]) for f in image_table.filename]
        return image_table
    
    
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
        #framedata can also be a dict of column names and values: This will be handeled in scopex
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
                    img = self._doFlatFieldCorrection(img, fname)
                if register:
                    img = self._register(img, fname)
                #if it's a z-stack
                if img.ndim==3: 
                    img = img.transpose((1,2,0))
                
                imgs.append(img)
            
            # Best performance has most frequently indexed dimension first 
            images_dict[key] = np.array(imgs) / 2**16  
            if verbose:
                print('Loaded {0} group of images.'.format(key))
            
        return images_dict



        
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
    
    
    
    def _register(self,img,fname):
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
            

    def CalculateDriftCorrection(self, Position=None, ZsToLoad=[1]):
        #from scipy.signal import fftconvolve
        if Position is None:
            Position = self.posnames
        elif type(Position) is str:
            Position = [Position]
        
        for pos in Position:
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
            print('calculated drift correction for position ' + pos)
        self.pickle()
    
