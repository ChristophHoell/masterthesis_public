
from .t2m_dataset import HumanML3D,KIT, CelebVData

from os.path import join as pjoin
__all__ = [
    'HumanML3D', 'KIT', "CelebVData",  'get_dataset',]

def get_dataset(opt, split='train', mode='train', accelerator=None):
    if opt.dataset_name == 't2m' :
        dataset = HumanML3D(opt, split, mode, accelerator)
    elif opt.dataset_name == 'kit' :
        dataset = KIT(opt,split, mode, accelerator)
    elif opt.dataset_name == "celebv":
        split_file = split + ".txt"
        dataset = CelebVData(opt.data_dir + "opt.txt", split_file, mode)
    else:
        raise KeyError('Dataset Does Not Exist')
    
    if accelerator:
        accelerator.print('Completing loading %s dataset' % opt.dataset_name)
    else:
        print('Completing loading %s dataset' % opt.dataset_name)
    
    return dataset

