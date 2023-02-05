import pandas as pd
import numpy as np
import argparse
from pathlib import Path

'''
DOWNLOAD_DIR="/sharefs//chem_data/pubchem/"
EXTRACTION_FILE='names_properties_1m.txt' #'iupacs_properties_10.txt'
FINAL_OUTPUT_DIR='data_1m'

python txt2csv.py \
--input_dir=/sharefs//chem_data/pubchem/names_properties_1m.txt \
--output_dir=/sharefs//chem_data/pubchem/data_1m 

rlaunch --private-machine=group --charged-group=health --cpu=16 --gpu=0 --memory=100000 \
-- python txt2csv.py \
--input_dir=/sharefs//chem_data/pubchem/data_1m_5cols/names_properties.txt \
--output_dir=/sharefs//chem_data/pubchem/data_1m_5cols 
'''

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    # path = '/sharefs//chem_data/pubchem/data/'
    print('Reading...')
    df = pd.read_csv(input_dir,delimiter='\t',header=None)
    # df.columns = ['Preferred','Canonical','Formula'] # Canonical<
    df.columns = ['Preferred','CAS','Systematic','Traditional','Canonical','Formula','Mass','LogP'] # add column names
    print(df.head())
    print('Original df.shape', df.shape) # 898483, 4
    # print(df.info)

    print('Dropping NAs in IUPACs and SMILES')
    df.dropna(subset=['Preferred','Canonical'],inplace=True) # not drpopping nas in task labels like formula, mass. log p
    print('Ater dropna: df.shape', df.shape) # 896718, 4

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print('writing...')
    iupac = df[['Preferred','CAS','Systematic','Traditional']]
    smiles = df['Canonical']

    
    iupac.to_csv(Path(args.output_dir) / 'iupacs.csv',index=False) # m:ode='a', index=False, header=False # not using appending mode
    print('Finished writing iupac!')
    print(iupac.shape)
    print(iupac.head())

    smiles.to_csv(Path(args.output_dir) / 'smiles.csv',index=False) # mode='a', index=False, header=False
    print('Finished writing smiles!')
    print(smiles.shape)
    print(smiles.head())
    
    '''
    ### discretize the continuous mass and log p labels
    groups = 20
    quantile_bins = [(1/groups)*x for x in range(groups+1)]
    labels = [i for i in range(groups)]

    print(df['LogP'].values)
    print(df['LogP'].values.shape)
    
    mass_quantiles = np.nanquantile(df['Mass'].values,q=quantile_bins)
    logp_quantiles = np.nanquantile(df['LogP'].values,q=quantile_bins)
    print('mass_quantiles',mass_quantiles) 
    print('logp_quantiles',logp_quantiles)
    # groups = 10,mass_quantiles [1.0e+00 1.9e+02 2.3e+02 2.6e+02 2.8e+02 3.0e+02 3.2e+02 3.4e+02 3.7e+02 4.6e+02 1.8e+04]
    # groups = 10,logp_quantiles [-70.2   0.6   1.5   2.1   2.5   3.    3.4   3.8   4.4   5.2  78. ]

    # df['Mass_label'] = pd.cut(x=df['Mass'],bins=mass_quantiles)#,labels=labels)
    # df['LogP_label'] = pd.cut(x=df['LogP'],bins=logp_quantiles)#,labels=labels)
    df['Mass_label'] = pd.cut(x=df['Mass'],bins=groups,precision=1)# retbins=True)#,labels=labels) #mass竟然有负数...
    df['LogP_label'] = pd.cut(x=df['LogP'],bins=groups,precision=1)# retbins=True)#,labels=labels)
    '''

    df.to_csv(Path(args.output_dir) / 'names_properties_8cols.csv',index=False) # names_properties_5cols.csv
    print('Finished writing names_properties!')
    print(df.shape)
    print(df.head())

