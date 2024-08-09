from denoising.connectivity import *
import numpy as np

opened = np.load(r'C:\Users\ТМ\YandexDisk\IHB\Projects\OpenCloseFMRI\data\open_close_time-series\roi\trt_roi_HCPex_ses-3.npy')
#closed = np.load(r'C:\Users\ТМ\YandexDisk\IHB\Projects\OpenCloseFMRI\data\open_close_time-series\roi\trt_close_roi_HCPex.npy')

op = glasso(opened, L1=0.001)
np.save(r'C:\Users\ТМ\Desktop\OpenCloseProject\lasso_trt_ses-3_HCPex.npy', op)
#cl = glasso(closed, L1=0.001)
#np.save(r'C:\Users\ТМ\Desktop\OpenCloseProject\trt_close_HCPex_lasso.npy', cl)


