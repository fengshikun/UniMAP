# UniMAP-pretrain


The code is based on huggingface transformers(4.9.2), [chemBerta](https://github.com/seyonechithrananda/bert-loves-chemistry)  ,and [C5T5](https://github.com/dhroth/c5t5).



### Data downloading and extraction


#### Download pubchem
The scrpits in ./download_pubchem are for downlaoding data from [pubchem](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/XML/) and further extration of certain fields (like iupac name, smiles, formula).

#### Generate ECFP fingerprints and function group labels

```python -u extract_feat_labels.py --smiles_file pubchme_smiles_file.lst```


### Pre-training:

```python -u -m torch.distributed.launch --nproc_per_node 8 --master_port 1233 train_multimodal_uni.py --run_name unimap_pretrain --dataset_path pubchme_smiles_file.lst --logging_steps 5 --tokenizer_path iupac_regex/ --smiles_tokenizer_path smiles_tokenizer/ --num_train_epochs 40 --output_dir unimap_pretraining --per_device_train_batch_size 64 --atom_vocab_file vocab/Merge_vocab.pkl --atom_vocab_size 10535  --function_group_file fg_labels.npy --finger_print_file ecfp_regression.npy --mlm_probability 0.2 --smiles_only --get_frag --pooler_type avg --fp16 --gnn_number_layer 3 --graph_max_seq_size 128 --mlm_group_probability 0.6 --gnn_dropout 0 --check_frag```

### Fine-tuning(MoleculeNet and DTA):


test_gnn_smiles_uni_avg_srun.sh
```
cd finetune;
declare -a test_sets=("bbbp" "clintox" "kiba" "davis")

for test_set in "${test_sets[@]}"; do
CUDA_VISIBLE_DEVICES=0 python -u finetune_mm_split.py --datasets $test_set --pretrained_model_name_or_path $1 --tokenizer_path iupac_regex/ --smiles_tokenizer_path smiles_tokenizer/  --lang_only --graph_uni --output_dir $1/graph_uni --pooler_type avg --split scaffold --n_seeds 3 --split scaffold --n_seeds 1 --n_trials 20 --graph_max_seq_size 128 --gnn_number_layer 3 --per_device_train_batch_size 64 --num_train_epochs_max 100  --number_seed 3  
done
```

bash test_gnn_smiles_uni_avg_srun.sh ./train_uni_smile



### Fine-tuning(DDI)

test_ddi_srun.sh:
```
cd finetune;
python finetune_mm_split_ddi.py --datasets deepddi --split random --pretrained_model_name_or_path $1 --tokenizer_path ../iupac_regex/ --smiles_tokenizer_path ../smiles_tokenizer --output_dir $1/graph_uni --lang_only --graph_uni --per_device_train_batch_size 32 --pooler_type avg --n_seeds 3 --n_trials 20 --gnn_number_layer 3

```

bash test_ddi_srun.sh ./train_uni_smile