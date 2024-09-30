from denoising.denoise import Denoising
from denoising.atlas import Atlas
from denoising.dataset import Dataset
from denoising.helpers import *
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)

# paste derivatives path
derivatives_path = '/data/Projects/OpenClose_RMET/bids/derivatives'
#'/arch/OpenCloseBeijin/INDI_Lite_NIFTI/derivatives/'
# #'/data/Projects/TestRetest_NYU/bids/derivatives'
#'/arch/OpenCloseBeijin/INDI_Lite_NIFTI/derivatives/' # 
# '/arch/OpenCloseProject/derivatives/' 

# enter number of runs
runs = 2
sessions = 1
task = 'rest'
tr = 2.5

# enter atlas name
atlases = ['AAL', 'Brainnetome', 'Schaefer200', 'HCPex']
#atlas_name = 'HCPex'

folder = '/data/Projects/OpenCloseChina/ts'

#for atlas_name in atlases:

data = Dataset(derivatives_path=derivatives_path, 
               sessions=sessions,
               TR=tr,
               runs=runs,
               task=task)


for atlas_name in atlases:
    for i in range(1, 7):
        atlas = Atlas(atlas_name=atlas_name, 
                      mean_mask='/home/tm/projects/OpenCloseProject/notebooks/mean_mask_rmet_03.nii.gz')
        denoise = Denoising(data, atlas, 
                            # enter denoising options here
                            strategy=i, 
                            use_GSR=False, 
                            use_cosine=True, 
                            smoothing=None) 

        ts = denoise.denoise(save_outputs=True, folder=folder)
    
# cd /home/tm/projects/OpenCloseProject
# nohup /home/tm/projects/OpenCloseProject/nilearn_env/bin/python script.py &