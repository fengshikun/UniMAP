import copy
import re
import argparse
from pathlib import Path
import pyparsing
import json
from tqdm import tqdm
import sys
import random

'''
rlaunch --private-machine=group --charged-group=health --cpu=8 --gpu=0 --memory=50000 \
-- python parsing.py --dataset=/sharefs/ylx/chem_data/pubchem/data_1m/100.csv \
--output_dir=/sharefs/ylx/chem_data/pubchem/data_1m/processed/

nohup rlaunch --private-machine=group --charged-group=health --cpu=8 --gpu=0 --memory=50000 \
-- python parsing.py --dataset=/sharefs/ylx/chem_data/pubchem/data_1m_5cols/iupacs.csv \
--output_dir=/sharefs/ylx/chem_data/pubchem/data_1m_5cols/processed/ \
> parsing.log &
'''

def print_list(lst, level=0):
    # print('    ' * (level - 1) + '+---' * (level > 0) + lst[0])
    for l in lst:
        if type(l) is list:
            print_list(l, level + 1)
        else:
            print('    ' * level + '+---' + l)



def get_dict_sub(lst, pos_list, parsed_iupac_dic, level=0):
    # lst = ['5', ['1H', 'isopropylbutyl'], '4', 'propylundecane']
    # print(get_dict(lst))
    ## {'isopropylbutyl': {0: [5], 1: [1]}, 'propylundecane': {0: [4]}}
    for l in lst:
        if type(l) is list:
            get_dict_sub(l, pos_list, parsed_iupac_dic, level + 1)
        else:
            if type(l) is str:
                temp = re.findall(r'\d+', l) # '1H'-> 1
                num_list = list(map(int, temp))
                # if l.isdigit():
                if len(num_list) > 0:
                    # assert len(res)==1 
                    if len(pos_list) > 0 and max(pos_list.keys())>=level:
                        for key in list(pos_list):
                            if key>=level: 
                                pos_list.pop(key)

                    if level not in list(pos_list):
                        pos_list[level] = []

                    # pos_list[level].append(int(l))
                    pos_list[level]+=num_list

                else:
                    # dic[l] = pos_list.copy() # not enough
                    parsed_iupac_dic[l] = copy.deepcopy(pos_list)
                    if len(pos_list)>0:
                        pos_list.pop(max(pos_list.keys()),None)
                        # print(dic) # stepwise check

    return parsed_iupac_dic


def get_dict(lst):
    pos_list = {}
    parsed_iupac_dic={}
    res = get_dict_sub(lst, pos_list, parsed_iupac_dic)
    return res

def get_val_dic_list(idx_value_list):#,iupac_list):
    indexes = []
    final_dic_list = []
    #assert len(idx_value_list) == len(iupac_list)
    for idx, dic in idx_value_list: 
        indexes.append(idx)
        #dic['index'] = idx
        #dic['iupac'] = iupac_list[idx]
        final_dic_list.append(dic)
    return final_dic_list, indexes

def get_train_dic_list(parsed_iupac_list,val_indexes): # iupac_list
    indexes = []
    final_dic_list = []
    #assert len(idx_value_list) == len(iupac_list)
    for idx, dic in enumerate(parsed_iupac_list):
        if idx not in val_indexes:
            indexes.append(idx)
            #dic['index'] = idx
            #dic['iupac'] = iupac_list[idx]
            final_dic_list.append(dic)
    return final_dic_list, indexes

if __name__=='__main__':
    
    '''
    #lst = ['a', ['b', 'c', ['d', 'i'], 'e'], 'f', ['g', 'h', ['j', 'k', 'l', 'm']]]
    #lst = ['5', ['1,2', 'isopropylbutyl'], '4', 'propylundecane']
    lst = ['5', ['1', 'isopropylbutyl'], ['2','3', 'dimethane','4','xxx'], '4', 'propylundecane']
    pos_list = {} # the variables must be claimed out of the recursion func
    parsed_iupac_dic={}
    print(get_dict(lst))
    '''
    
    sys.setrecursionlimit(10000) 
    random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    #parser.add_argument("--output_file", required=True, type=str)
    parser.add_argument("--val_percent", required=False, default='0.1' ,type=float)

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    processed_name_list = []
    iupac_list = []

    thecontent = pyparsing.Word(pyparsing.alphanums) #| '-' # | '+' #pyparsing.alphanums
    parens = pyparsing.nestedExpr( '(', ')', content=thecontent)

    with open(args.dataset,'r') as f:
        # with open(args.output_dir/args.output_file,'w'):
        #names = f.readlines()[1:]
        names = f.read().splitlines()[1:]
        for i,line in enumerate(tqdm(names)):
            #print(line)
            line_ = line.replace('"','')\
                .replace('[','(').replace(']',')')\
                .replace(',','sep').replace('.','sep')\
                .replace('-',' ').replace('+',' ').replace(';',' ').replace('&',' ').replace('?',' ')
            #clean = re.sub(r"[,.;+-@#?!&$]+", " ", line_)
            nested_line_ = '('+line_+')'
            name_list = parens.parseString(nested_line_).asList()[0]
            #print(name_list)
            # parsed_iupac_dic = {}
            # pos_list = {}
            instance_dic = {}
            name_dic = get_dict(name_list)
            instance_dic['index'] = i
            instance_dic['iupac'] = line.replace('"','')
            instance_dic['parsed_iupac'] = name_dic
            
            #print(name_dict)
            processed_name_list.append(instance_dic)
            #iupac_list.append(line.replace('"','')) # keep original iupac name
        f.close()

    with open(Path(args.output_dir) / 'full.jsonl','w') as w:
        # json.dump(processed_name_list,w)
        for row in processed_name_list:
            print(json.dumps(row), file=w)
        w.close()
    
    if args.val_percent > 0:
        val_size = int(len(processed_name_list)*args.val_percent)
        print('Sampling validation set,val_percent:',args.val_percent)
        val_idx_value = random.sample(list(enumerate(processed_name_list)), val_size) # [(id,value),...]

        print('Getting val&train dic list...')
        val_final_dic_list,val_indexes = get_val_dic_list(val_idx_value)#,iupac_list)
        train_final_dic_list,train_indexes = get_train_dic_list(processed_name_list,val_indexes) #iupac_list

        assert len(set(val_indexes) & set(train_indexes)) == 0

        '''
        val_indexes = []
        val_values = []
        for idx, val in val_idx_value:
            val_indexes.append(idx)
            val['index'] = idx
            val['iupac'] = iupac_list[idx]
            val_values.append(val)
        print('Taking the left as training set...')

        train_indexes = []
        train_values = []
        for i, e in enumerate(processed_name_list):
            if i not in val_indexes:
                train_indexes.append(i)
                train_values.append(e)
        train_values =  [processed_name_list[i] for i, e in enumerate(processed_name_list) if i not in val_indexes]
        '''
    
        print('Val/Train size: ',len(val_final_dic_list),len(train_final_dic_list)) # Val/Train size:  89671 807047

        print('writing files...')
        with open(Path(args.output_dir) / 'val.jsonl','w') as w: # val
            for row in val_final_dic_list:
                print(json.dumps(row), file=w)
            w.close()

        with open(Path(args.output_dir) / 'train.jsonl','w') as w: # train
            for row in train_final_dic_list:
                print(json.dumps(row), file=w)
            w.close()