# PosLbl module for microscopy data. 
# Contains all single TP FrameLbls for a specific well/position
# Deals with tracking
# AOY

import sys
import os
import pandas as pd
import numpy as np
from skimage import measure
from oyLabCode import Metadata
from multiprocessing import Pool
from functools import partial
from oyLabCode.Processing import FrameLbl
from scipy.spatial import KDTree
import lap
from tqdm import tqdm

class PosLbl(object):
    def __init__(self, Pos=None, MD=None ,pth=None, acq = None, threads=10, **kwargs):
        

        if any([Pos is None]):
            raise ValueError('Please provide position')
        
        if pth is None:
            if MD is not None:
                self.pth = MD.base_pth;
        else:
            self.pth = pth
        
        if MD is None:
            MD = Metadata(pth)
        
        if MD().empty:
            raise AssertionError('No metadata found in supplied path') 
                
        if Pos not in MD.posnames:
            raise AssertionError('Position does not exist in dataset')  
        self.posname = Pos
  
        self.channels = MD.unique('Channel',Position=Pos) 
        self.acq = MD.unique('acq',Position=Pos) 
        self.frames = MD.unique('frame',Position=Pos)
        
        
        # Create all framelabels for the different TPs. This will segment and measure stuff. 
        
        def mute():
            sys.stdout = open(os.devnull, 'w')    

        with Pool(threads, initializer=mute) as ppool:
            #self.framelabels = np.array(ppool.map(partial(FrameLbl, MD = MD, pth = pth, Pos=Pos, **kwargs), self.frames))
            #ppool.close()
            frames = list(tqdm(ppool.imap(partial(FrameLbl, MD = MD, pth = pth, Pos=Pos, **kwargs), self.frames), total=len(self.frames)))
            ppool.close()
            ppool.join()
        self.framelabels = np.array(frames)
        print('\nFinished loading and segmenting position ' + Pos)
            
        
    def __call__(self):
        print('PosLbl object for position ' + self.posname + '.')
        print('\nThe path to the experiment is: \n ' + self.pth)
        print('\n '+ str(len(self.frames)) + ' frames processed.')        

        print('\nAvailable channels are : ' + ', '.join(list(self.channels))+ '.')
    
    

    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 
    @property
    def num(self):
        return np.array([r.num for r in self.framelabels])
    
    @property
    def centroid(self):
        return np.array([r.centroid for r in self.framelabels])

    @property
    def weighted_centroid(self):
        return np.array([r.weighted_centroid for r in self.framelabels])

    @property
    def area(self):
        return np.array([r.area for r in self.framelabels])
    
    @property
    def centroid_um(self):
        return np.array([r.centroid_um for r in self.framelabels])
    
    @property
    def weighted_centroid_um(self):
        return np.array([r.weighted_centroid_um for r in self.framelabels])
    
    @property
    def area_um2(self):
        return np.array([r.area_um2 for r in self.framelabels])

    def mean(self,ch, periring=False):
        return np.array([r.mean(ch, periring=periring) for r in self.framelabels])

    def median(self,ch, periring=False):
        return np.array([r.median(ch, periring=periring) for r in self.framelabels])
    
    def minint(self,ch, periring=False):
        return np.array([r.minint(ch, periring=periring) for r in self.framelabels])

    def maxint(self,ch, periring=False):
        return np.array([r.maxint(ch, periring=periring) for r in self.framelabels])

    def ninetyint(self,ch, periring=False):
        return np.array([r.ninetyint(ch, periring=periring) for r in self.framelabels])
    
    @property
    def density(self):
        return np.array([r.density for r in self.framelabels])
    
    
    
    
    def track(self, i=0):
        assert(i<=len(self.trackinds)), "track index must be < %i" % len(self.trackinds)
        return self.onetrack(self, i=i)
    
    #class for a single track. return everything we care about
    class onetrack(object):
        def __init__(self,outer, i=0):
            self.trackinds = outer.trackinds[i]
            self._outer = outer
        
        @property
        def T(self):
            return np.nonzero(~np.isnan(self.trackinds.astype('float')))[0]           
        @property
        def centroid(self):
            cents = self._outer.centroid
            return np.array([list(cents[j][self.trackinds[j],:]) for j in self.T])
        @property
        def weighted_centroid(self):
            cents = self._outer.weighted_centroid
            return np.array([list(cents[j][self.trackinds[j],:]) for j in self.T])
        @property
        def centroid_um(self):
            cents = self._outer.centroid_um
            return np.array([list(cents[j][self.trackinds[j],:]) for j in self.T])
        @property
        def weighted_centroid_um(self):
            cents = self._outer.weighted_centroid_um
            return np.array([list(cents[j][self.trackinds[j],:]) for j in self.T])
        @property
        def area(self):
            a = self._outer.area
            return np.array([a[j][self.trackinds[j]] for j in self.T])
        @property
        def area_um2(self):
            a = self._outer.area_um2
            return np.array([a[j][self.trackinds[j]] for j in self.T])
        def mean(self,ch, periring=False):
            m = self._outer.mean(ch, periring)
            return np.array([m[j][self.trackinds[j]] for j in self.T])
        def median(self,ch, periring=False):
            m = self._outer.median(ch, periring)
            return np.array([m[j][self.trackinds[j]] for j in self.T])
        def minint(self,ch, periring=False):
            m = self._outer.minint(ch, periring)
            return np.array([m[j][self.trackinds[j]] for j in self.T])
        def maxint(self,ch, periring=False):
            m = self._outer.maxint(ch, periring)
            return np.array([m[j][self.trackinds[j]] for j in self.T])
        def ninetyint(self,ch, periring=False):
            m = self._outer.ninetyint(ch, periring)
            return np.array([m[j][self.trackinds[j]] for j in self.T])

    
    
    
    def trackcells(self, **kwargs):
        '''
        all tracking is done using the Jonker Volgenant lap algorithm:
        R. Jonker and A. Volgenant, "A Shortest Augmenting Path Algorithm for Dense and Sparse Linear Assignment Problems", Computing 38, 325-340 (1987)
        
        TODO: try Viterbi algo for tracking
        TODO: add mitosis detector: https://academic.oup.com/bioinformatics/article/35/15/2644/5259190
        TODO: add support for splitting cells
        
        '''
        self._link(**kwargs)
        self._closegaps(**kwargs)
        
    
    def _link(self, ch=None,search_radius=75,**kwargs):
        #todo: extend for a general cost function. make class of cost functions that returns shape, cc, ii, jj
        if ch is None:
            ch = self.channels[0]
            print('No channel provided, using '+ch)
        cents = self.centroid_um
        ints = self.ninetyint(ch)
        nums = self.num

        for i in np.arange(nums.shape[0]-1):
            
            sys.stdout.write("\r"+'linking frame '+ str(i))
            sys.stdout.flush()
            
            T = KDTree(cents[i+1])
            #We calculate points in centroid(n+1) that are less than distance_upper_bound from points in centroid(n)
            dists, idx = T.query(cents[i], k=12, distance_upper_bound=search_radius)

            dists = [r[r<1E308] for r in dists]

            idx = [r[r<cents[i+1].shape[0]] for r in idx]

            #possible matches in n+1
            jj = np.concatenate(idx)

            #possible correspondence in n
            j=0
            ii=[]
            for r in idx:
                ii.append(j*np.ones_like(r))
                j+=1
            ii = np.concatenate(ii)

            #ii jj cc are now sparse matrix in COO format
            ampRatio = np.array(max(list(ints[i][ii]), list(ints[i+1][jj])))/np.array(min(list(ints[i][ii]), list(ints[i+1][jj])))

            #costs of match
            cc = np.concatenate(dists)*ampRatio


            shape = (nums[i], nums[i+1])

            cc, ii, kk = prepare_sparse_cost(shape, cc, ii, jj, cost_limit=300)
            ind1, ind0 = lap.lapmod(len(ii)-1, cc, ii, kk, return_cost=False)
            ind1[ind1 >= shape[1]] = -1
            ind0[ind0 >= shape[0]] = -1
            #inds in n+1 that match inds (1:N) in n
            ind1 = ind1[:shape[0]]
            #inds in n that match inds (1:N) in n+1
            ind0 = ind0[:shape[1]]
            self.framelabels[i].link1in2 = ind1
        self.framelabels[nums.shape[0]-1].link1in2=np.array([])
    
    

    
    def _getTrackLinks(self,i=0,l=0):
        '''
        recursive function that gets an initial frame i and starting cell label 
        l and returns all labels
        '''
        if i<len(self.frames)-1:
            if l>-1:
                return(np.append(l, self._getTrackLinks(i+1,self.framelabels[i].link1in2[l])))
            else:
                pass
        elif i==len(self.frames)-1:
            if l>-1:
                return(l)
            else:
                pass
    

    def _getAllContinuousTrackSegs(self, minseglength=5):
        '''
        function that returns the frame indexing of all contiuous tracks (chained links) longer than minseglength.

        Assumes links have been calculated
        '''
        trackbits = []
        for i in np.arange(len(self.frames)-1):
            for l in np.nonzero(np.isin(np.arange(self.framelabels[i].num), np.array([r[i] for r in trackbits]), invert=True))[0]:
                trkl = self._getTrackLinks(i=i, l=l)
                trackbits.append(np.pad(trkl.astype('object'), (i, len(self.frames)-trkl.size-i), 'constant', constant_values=(None, None)))
        trackbits = np.array(trackbits)
        
        #return tracks segments that have more than minseglength frames
        return trackbits[(np.array([np.sum(r != None) for r in trackbits])>=minseglength)]
    
    
    def _closegaps(self, maxStep=100, ch=None,maxAmpRatio=1.5, mintracklength=20, **kwargs):
        #todo: split. When a stub that starts in the middle has a plausible link, make a compound track
        if ch is None:
            ch = self.channels[0]
            

        
        trackbits = self._getAllContinuousTrackSegs(**kwargs)
        
        cents = self.centroid_um
        ints = self.ninetyint(ch)
        notdoneflag=1

        while notdoneflag:
            
            trackstarts = np.array([np.where(~np.isnan(r.astype('float')))[0][0] for r in trackbits])
            trackends = np.array([np.where(~np.isnan(r.astype('float')))[0][-1] for r in trackbits])
            
            dtmat = np.expand_dims(trackstarts,1)-np.expand_dims(trackends,0)
            possiblelinks = np.transpose(np.nonzero((dtmat>0)*(dtmat<4)))
            ii = []
            jj = []
            cc = []
            for i in np.arange(len(possiblelinks)):
                # frame1 - end of possible to link
                frame1 = trackends[possiblelinks[i][1]]
                # frame2 - beginning of possible link
                frame2 = trackstarts[possiblelinks[i][0]]
                # cell label in frame 1 to link
                ind1 = trackbits[possiblelinks[i][1]][frame1]
                # cell label in frame 2 to link
                ind2 = trackbits[possiblelinks[i][0]][frame2]
                dt = frame2-frame1
                dr = np.linalg.norm(cents[frame1][ind1]-cents[frame2][ind2])
                da = max(ints[frame1][ind1], ints[frame2][ind2])/min(ints[frame1][ind1], ints[frame2][ind2])

                if dr <= (np.sqrt(dt)*maxStep):
                    if da<=maxAmpRatio:
                        ii.append(possiblelinks[i][1])
                        jj.append(possiblelinks[i][0])
                        
                        #maybe one day we'll change this somehow. Not sure how rn
                        cost = dr*da*dt
                        
                        cc.append(cost)
            ii = np.array(ii)
            jj = np.array(jj)
            cc = np.array(cc)

                        
            if len(ii)==0:
                print('\nFinished connecting tracks')
                doneflag=1
                break

            shape = (len(trackbits),len(trackbits))
            cc, ii, kk = prepare_sparse_cost(shape,cc,ii,jj, 1000)
            match1, _ = lap.lapmod(len(ii)-1, cc, ii, kk, return_cost=False)
            match1[match1 >= shape[1]] = -1
            #inds in n+1 that match inds (1:N) in n
            match1 = np.array(match1[:shape[0]])

            trackindstofill = np.nonzero(match1+1)[0]
            trackindstoadd = match1[np.nonzero(match1+1)]

            fa = {trackindstofill[i]: trackindstoadd[i] for i in range(len(trackindstoadd))} 

            for i in fa:
                sf = trackstarts[fa[i]]
                ef = trackends[fa[i]]+1
                trackbits[i][np.arange(sf,ef)]=trackbits[fa[i]][np.arange(sf,ef)]
                trackbits[fa[i]][np.arange(sf,ef)]=None

                #add nans in gaps
                eef = trackends[i]+1
                trackbits[i][np.arange(eef,sf)]=np.nan

            #remove lines that are all Nones
            trackbits = trackbits[[any(~np.isnan(r.astype('float'))) for r in trackbits]]
        
        trackbits = trackbits[np.array([sum(~np.isnan(r.astype('float'))) for r in trackbits])>=mintracklength]
        
        sortind = np.lexsort((np.arange(len(trackbits)) ,[np.sum(np.isnan(r.astype('float'))) for r in trackbits] ,[np.sum(r==None) for r in trackbits]))
        self.trackinds = trackbits[sortind]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def prepare_sparse_cost(shape, cc, ii, jj, cost_limit):
    '''
    Transform the given sparse matrix extending it to a square sparse matrix.
    
    Parameters
    ==========
    shape: tuple
       - cost matrix shape
    (cc, ii, jj): tuple of floats, ints, ints)
        - cost matrix in COO format, see [1].
    cost_limit: float
    
    Returns
    =======
    cc, ii, kk
      - extended square cost matrix in CSR format
    
    Notes
    =====
    WARNING: Avoid using scipy.sparse.coo_matrix(cost) as it will not return the correct (cc, ii, jj).
    `coo_matrix` leaves out any zero values which are the most salient parts of the cost matrix.
    (cc, ii, jj) should include zero costs (if any) and skip all costs that are too large (infinite).
    
    1. https://en.wikipedia.org/wiki/Sparse_matrix
    '''
    assert cost_limit < np.inf
    n, m = shape
    cc_ = np.r_[cc, [cost_limit] * n,
                [cost_limit] * m, [0] * len(cc)]
    ii_ = np.r_[ii, np.arange(0, n, dtype=np.uint32),
                np.arange(n, n + m, dtype=np.uint32), n + jj]
    jj_ = np.r_[jj, np.arange(m, n + m, dtype=np.uint32),
                np.arange(0, m, dtype=np.uint32), m + ii]
    order = np.lexsort((jj_, ii_))
    cc_ = cc_[order]
    kk_ = jj_[order]
    ii_ = np.bincount(ii_, minlength=shape[0]-1)
    ii_ = np.r_[[0], np.cumsum(ii_)]
    ii_ = ii_.astype(np.uint32)
    assert ii_[-1] == 2 * len(cc) + n + m
    return cc_, ii_, kk_