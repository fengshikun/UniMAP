import pandas as pd

if __name__=='__main__':
    print('Reading...')
    data = pd.read_csv('/sharefs//chem_data/pubchem/data_1m/iupacs.csv')
    print('Replacing...')
    data['no_bracs'] = data['Preferred'].apply(lambda x:x.replace('(',' ').replace(')',' ').replace('[',' ').replace(']',' ').strip())
    data.drop(['Preferred'],axis=1,inplace=True)
    print('Writing...')
    data.to_csv('/sharefs//chem_data/pubchem/data_1m/iupacs_no_bracs.csv',index=False)