import pandas as pd
import re

if __name__=='__main__':
    print('Reading...')
    data = pd.read_csv('/sharefs//chem_data/pubchem/data_1m/iupacs.csv')
    print('Replacing...')
    data['no_bracs'] = data['Preferred'].apply(lambda x:x.replace('(',' ').replace(')',' ').replace('[',' ').replace(']',' ').strip())
    
    pattern = r'[0-9]'
    data['no_bracs_nums'] = data['no_bracs'].apply(lambda x: re.sub(pattern,'',x)).apply(lambda x:x.replace(',','')).apply(lambda x:x.replace('-',' '))

    data.drop(['Preferred'],axis=1,inplace=True)
    data.drop(['no_bracs'],axis=1,inplace=True)
    
    print('Writing...')
    data.to_csv('/sharefs//chem_data/pubchem/data_1m/iupacs_no_bracs_nums.csv',index=False)