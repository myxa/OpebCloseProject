import os
import matplotlib.pyplot as plt
import numpy as np
from nilearn.connectome import ConnectivityMeasure
import pandas as pd


import pandas as pd

def reorder_matrix(matr):
    """
    Reorder FC matrix in Left-Right order
    
    Parameters
    ----------
    matr: np.array
        Matrix to reorder
    atlas_name: str
        One of ['HCPex', 'Schaefer200']

    Returns
    -------
    np.array
        Reordered matrix (wow!)
    """

    df = pd.read_excel('../atlas/HCPex_sorted.xlsx',
                            ).drop([401, 365, 398], axis=0).reset_index()
    map = dict(zip(df.index, df.HCPex_ID.values))

    if isinstance(matr, np.ndarray) and len(matr.shape) == 3:
        sort_0 = np.zeros_like(matr)
        for i in range(len(matr)):
            temp = matr[i][map[i]]
            sort_0[i] = temp.T[map[i]]
    else:
        sort_0 = matr[map[i]]
        sort_0 = sort_0.T[map[i]]

    return sort_0


def plot_corr_hist(ts1, ts2, title=''):
    l=[]
    for i in range(423):
        l.append(np.corrcoef(ts1[i], ts2[i])[0, 1])
    plt.hist(l, bins='auto')
    plt.title(f'{title}, median={round(np.median(l), 3)}');


def load_timeseries_yadisk(path, sub=None, run=1, task='rest', strategy=4, atlas_name='AAL'):
    """
    Load time series from folder.
    Time-series should be stored in a folder for each subject. In subject folders there should be folder for atlas.
    In atlas folder there are csv files

    Parameters
    ----------
    path: str
        Path to folder with all subject folders
    sub: list of str, optional
        List of subjects to load data in form 'sub-n'. If None, all subjects are loaded (default None)
    run: int
        Run to load
    task: str
        Task to load
    strategy: int
        Strategy to load
    atlas_name: str
        Atlas name. Should be one of ['HCPex', 'Schaefer200', 'AAL']
    
    Returns
    -------
    List of numpy.arrays 
    """
    ts = []
    if sub is None:
        sub = os.listdir(path)
    failed = []
    for i in sub:
        try:
            name = f'{i}_task-{task}_run-{run}_time-series_{atlas_name}_strategy-{strategy}.csv'
            path_to_file = os.path.join(path, f'{i}', atlas_name, name)
            ts.append(pd.read_csv(path_to_file).values)
        except FileNotFoundError:
            failed.append(i)
            continue
    print('no files available:', failed)
    return ts


def fetch_ts(path, sub=None, run=1, task='rest', strategy=4, atlas_name='AAL'):
    """
    Load time series from folder.
    Time-series should be stored in a folder for each subject. In subject folders there should be folder for atlas.
    In atlas folder there are csv files

    Parameters
    ----------
    path: str
        Path to folder with all subject folders
    sub: list of str, optional
        List of subjects to load data in form 'sub-n'. If None, all subjects are loaded (default None)
    run: int
        Run to load
    task: str
        Task to load
    strategy: int
        Strategy to load
    atlas_name: str
        Atlas name. Should be one of ['HCPex', 'Schaefer200', 'AAL']
    
    Returns
    -------
    List of numpy.arrays 
    """
        
    ts = []
    if sub is None:
        sub = os.listdir(path)
    failed = []
    for i in sub:
        if not isinstance(i, str):
            i = str(i)
        if 'sub' in i:
            i = i[4:]
        try:
            name = f'sub-{i}_task-{task}_run-{run}_time-series_{atlas_name}_strategy-{strategy}.csv'
            path_to_file = os.path.join(path, f'sub-{i}', 'time-series', atlas_name, name)
            #print(path_to_file)
            ts.append(pd.read_csv(path_to_file).values)
        except FileNotFoundError:
            failed.append(i)
            continue
    print('no files available:', failed)
    return ts





