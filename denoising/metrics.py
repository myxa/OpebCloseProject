import numpy as np
from typing import List
from numpy.typing import NDArray


class ICC:
    def __init__(self, data):
        self.data = data
        self.k = data.shape[1] # ses
        self.n = data.shape[0] # sub

    def sstotal(self):
        return np.var(self.data) * (self.k * self.n - 1)
    
    def msr(self):
        return np.var(np.mean(self.data, axis=1)) * self.k
    
    def msc(self):
        return np.var(np.mean(self.data, axis=0)) * self.n
    
    def mse(self):
        return (self.sstotal() - self.msr() * (self.n -1) - self.msc() * (self.k - 1)) / (self.n - 1) * (self.k - 1)
    
    def icc(self):
        return (self.msr() - self.mse()) / (self.msr() + (self.k - 1) * self.mse())


class legacyICC:
    def __init__(self, data: List[NDArray], use_mask=False, mask_file=None):
        """
        Class to calculate ICC and BSS.

        Parameters
        ----------
        data: list
            list of data for every session to compute metrics
            one session - np.array, vector of functional connectivity
            Note! for BSS calculation only two sessions should be passed
        mask_file: None or np.array, optional
            mask to be used to exclude zero elements (for simulations)

        """
        self.n_ses = len(data)
        self.n_sub = len(data[0])
        self.data = data

        if use_mask:
            
            self.thrsh = 0

            if mask_file is None:
                self.mask = [np.array(d > self.thrsh) for d in data] # 2
                self.data = [data[i] * self.mask[i] for i in range(self.n_ses)] # 2

                # len(self.data) = 2
                # self.data[0].shape = (n_sub, roi, roi)
            else:
                self.mask = [np.array([mask_file for _ in range(self.n_sub)]) for _ in range(self.n_ses)]
                self.data = [data[i] * self.mask[i] for i in range(self.n_ses)]


    @property
    def _avg_matr(self):
        return np.mean(
            np.concatenate(self.data), axis=0)
    

    def _bms(self) -> float:
        """
        Calculates the sum of squared between-subj variance,
        the average subject value subtracted from overall avg of values

        """
        # group vecs by subject
        sub_vec = [[self.data[ses][sub] for ses in range(self.n_ses)] for sub in range(self.n_sub)] # list(list*2)*nsub
        
        return np.sum([(self._avg_matr - 
                        np.mean(sub_vec[sub], axis=0)) **2 for sub in range(self.n_sub)], axis=0) * self.n_ses


    def _wms(self) -> float:
        """
        calculates the sum of squared Intra-subj variance,
        the average session value subtracted from overall avg of values

        """
        return np.sum([(self._avg_matr - 
                        np.mean(self.data[ses], axis=0)) **2 for ses in range(self.n_ses)], axis=0) * self.n_sub


    def icc(self) -> NDArray:
        """
        icc metric

        """
        bms = self._bms()
        wms = self._wms()
        #print(np.mean(bms), np.mean(wms))
        icc =  ((bms - wms) / 
                (1e-09 + bms + (self.n_ses - 1) * wms))
        return icc
    


def bss(a: List) -> float:
    """
    BSS calculation

    parameters
    ----------
    a: list
        data of two sessions
        
    returns
    -------
    array of BSS for every subject
    """
    ans = np.zeros(len(a[0]))
    for i in range(len(a[0])):
        ans[i] = np.corrcoef(a[0][i], a[1][i])[0, 1]
    return ans


def mean_fd(sub, run, data):
    return np.mean(data.get_confounds_one_subject(sub)[run-1]['framewise_displacement'][1:])

def qc_fc(fc, run, mean_fd_vec):
    qc_mat = np.zeros((fc.shape[1], fc.shape[2]))
    for i in range(fc.shape[1]):
        for t in range(fc.shape[2]):
            qc_mat[i, t] = np.corrcoef(fc[:, i, t], mean_fd_vec[run-1])[0, 1]
    return qc_mat