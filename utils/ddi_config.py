from easydict import EasyDict as edict


DDI_CONFIG = edict()
DDI_CONFIG.data_dir = '/sharefs/sharefs-test_data/deepddi/data/DrugBank5.0_Approved_drugs'
DDI_CONFIG.label_file = '/sharefs/sharefs-test_data/deepddi/data/DrugBank_known_ddi.txt'
DDI_CONFIG.train_ratio = 0.6