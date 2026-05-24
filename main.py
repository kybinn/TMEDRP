import sys  
import pandas as pd
import torch
import json
import os
import argparse
import itertools
import dataload
import config
import pretraining, finetuning
from copy import deepcopy
from utils.tools import get_mask,set_seed_all
import time
import numpy as np
import warnings
from utils.tools import smiles_to_chemberta_embeddings

def wrap_params(training_params, type='unlabeled'):
    aux_dict = {k: v for k, v in training_params.items() if k not in ['unlabeled', 'labeled']}
    aux_dict.update(**training_params[type])
    return aux_dict

def make_dir(new_folder_name):
    if not os.path.exists(new_folder_name):
        os.makedirs(new_folder_name)

def dict_to_str(d):
    return "_".join(["_".join([k, str(v)]) for k, v in d.items()])

def main(args, drug, params_dict):
    start_time = time.time()
    device = config.CUDA_ID
    set_seed_all(2024)

    # Load mix gene expressions for both ~9000 tcga and ~1000 cell line
    gex_features_df = pd.read_csv(config.gex_feature_file, index_col=0)  # 11325 rows x 1307 columns， first column is Tissue, rest are gene expressions
    TME_gene_df = pd.read_csv(config.tme_feature_file, index_col=0)
    gmt_path = config.pathway_file
    mask_tme, pathway_tme = get_mask(gmt_path, mask_col=20, genes=TME_gene_df.columns)

    TME_cancer_Genes_df = pd.read_csv(config.tme_can_feature_file, index_col=0) 
    TME_cancer_gmt_path = config.pathway_can_file
    mask_can, pathway_can = get_mask(TME_cancer_gmt_path, mask_col=64, genes=TME_cancer_Genes_df.columns) 
    
    # Load traning params
    with open(os.path.join('train_params.json'), 'r') as f:
        training_params = json.load(f)
    training_params['unlabeled'].update(params_dict)
    training_params['labeled'].update(params_dict)
    param_str = dict_to_str(params_dict)
    model_save_folder = os.path.join('model_save')

    training_params.update({'device': device, 
                            'total_genes': gex_features_df.shape[1]-1, 
                            'model_save_folder': os.path.join(model_save_folder, param_str),
                            'retrain_flag': args.retrain_flag})
    
    make_dir(training_params['model_save_folder'])
    
    # Data construction for the pre-training
    s_dataloaders, t_dataloaders = dataload.get_unlabeled_dataloaders(gex_features_df=gex_features_df, 
                                                                    TMEgenes =TME_gene_df.columns,  
                                                                    TME_cancer_genes = TME_cancer_Genes_df.columns,                                         
                                                                    seed=2024,
                                                                    test_ratio=0.1,
                                                                    batch_size=training_params['unlabeled']['batch_size'])
    s_train = s_dataloaders[0]
    cancerGenes = next(iter(s_train))[1].size(1)
    TMEGenes = next(iter(s_train))[2].size(1)
    TME_cancer_genes = next(iter(s_train))[3].size(1)
    training_params.update({'cancer_genes': cancerGenes, 
                            'TME_genes': TMEGenes,
                            'tme_cancer_genes': TME_cancer_genes,
                            'n_cancer_pathways': len(pathway_can),
                            'n_pathways': len(pathway_tme)})

    # Start pretraining
    cancer_encoder, tme_encoder,TME_cancer_encoder = pretraining.training(s_dataloaders=s_dataloaders,
                                   t_dataloaders=t_dataloaders,
                                   mask_tme=mask_tme,
                                    mask_can=mask_can,
                                   **wrap_params(training_params, type='unlabeled'))

    
    # Data construction for the fine-tuning
    labeled_dataloader = dataload.get_multi_labeled_dataloader(gex_features_df=gex_features_df,
                                                            TMEgenes =TME_gene_df.columns,
                                                            TME_cancer_genes = TME_cancer_Genes_df.columns,    
                                                            seed=2024,
                                                            batch_size=training_params['labeled']['batch_size'],
                                                            drug=drug,
                                                            ccle_measurement=args.measurement,
                                                            threshold_gdsc=args.thres_g,
                                                            threshold_label=args.thres_s,
                                                            n_splits=args.n) 
    
    fold = 0
    all_results = []
    for train_labeled_ccle, test_labeled_ccle, labeled_tcga in labeled_dataloader:
        print('\n################ Dataset Fold: {} ################'.format(fold))
        ft_cancer_encoder = deepcopy(cancer_encoder)
        ft_tme_encoder = deepcopy(tme_encoder)
        ft_TME_cancer_encoder = deepcopy(TME_cancer_encoder)
        print('================Drugs:================\n', drug)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") 
            # DRP training for the CCLE dataset 
            
            smiles = config.drug_smiles
            smiles_strs =  []
            for i in range(len(smiles)):
                smiles_str = smiles[i]
                smiles_strs.append(smiles_str)

            ### chemberta embedding for drugs
            drug_embedding = smiles_to_chemberta_embeddings(smiles_strs) 

            network = finetuning.training(drug_embedding=drug_embedding,cancer_encoder=ft_cancer_encoder,
                                            tme_encoder=ft_tme_encoder,
                                            TME_cancer_encoder=ft_TME_cancer_encoder,
                                            s_dataloader=train_labeled_ccle,
                                            val_dataloader=test_labeled_ccle,
                                            tcga_dataloader=labeled_tcga,
                                            **wrap_params(training_params, type='labeled'))
            
            # Transfer drug response prediction for TCGA dataset   
            print("\n================Transfer testing for TCGA data================")   
            test_loss, results, y_true, y_pred = finetuning.testing(drug_embedding=drug_embedding,model=network,
                                            t_dataloader=labeled_tcga,
                                            device=training_params['device'])
            print('testing loss : {:.4f}'.format(test_loss))
            print('test-avg (TCGA) metrics:\n\t {}'.format(np.around(results, 4)))


        results = results.set_index('Drug').T
        task_save_folder = os.path.join('results', 'fold-'+str(fold))
        make_dir(task_save_folder)
        file_name = os.path.join(task_save_folder, param_str)
        with open(f'{file_name}.csv', 'w') as f:
            results.to_csv(f)      
        fold = fold+1
        all_results.append(results)
        torch.save(network.state_dict(), os.path.join(task_save_folder, 'DRPmodel.pt'))


    # Calculate the average result of 5-CV    
    avg_result = np.mean(np.array(all_results), 0)  
    avg_result = pd.DataFrame(avg_result)
    avg_result.columns = drug; avg_result.index = ['Precision','F1','PRAUC','Accuracy','ROC_AUC']
    file_name = os.path.join('results', param_str)
    with open(f'{file_name}.csv', 'w') as f:
        avg_result.to_csv(f)      
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pretraining and Fine_tuning')
    parser.add_argument('--metric', dest='metric', nargs='?', default='auroc', choices=['auroc', 'auprc'])
    parser.add_argument('--measurement', dest='measurement', nargs='?', default='Z_SCORE', choices=['Z_SCORE', 'LN_IC50', 'AUC'])
    parser.add_argument('--preEpoch', type=int, default=90, help='the number of epoch in pretraining') 
    parser.add_argument('--thres_gdsc', dest='thres_g', nargs='?', type=float, default=0.0)
    parser.add_argument('--thres_label', dest='thres_s', nargs='?', type=float, default=0.1)
    parser.add_argument('--n', dest='n', nargs='?', type=int, default=5) #folds for cross validation
    parser.add_argument('--alph', dest='alph', nargs='?', type=float, default=0.005, help='Coefficient of transfer loss')
    parser.add_argument('--beta', dest='beta', nargs='?', type=float, default=0.005, help='Coefficient of contrastive loss')

    train_group = parser.add_mutually_exclusive_group(required=False)
    train_group.add_argument('--train', dest='retrain_flag', action='store_true')
    train_group.add_argument('--no-train', dest='retrain_flag', action='store_false')
    parser.set_defaults(retrain_flag=False)
    
    norm_group = parser.add_mutually_exclusive_group(required=False)
    norm_group.add_argument('--norm', dest='norm_flag', action='store_true')
    norm_group.add_argument('--no-norm', dest='norm_flag', action='store_false')
    parser.set_defaults(norm_flag=True)

    args = parser.parse_args()
    # params_grid = {
    #     "pretrain_num_epochs": [50, 60, 70, 80, 90, 100],
    #     "uda_num_epochs": [200, 300, 400, 500]
    # }
    preEpoch = 90
    params_grid = {
        "pretrain_num_epochs": [preEpoch],
        "train_num_epochs": [200]
    }

    keys, values = zip(*params_grid.items())
    params_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for param in params_list:      
        main(args=args, params_dict=param,
             drug=["5-Fluorouracil", "Cisplatin", "Cyclophosphamide", "Docetaxel", 
                   "Doxorubicin", "Etoposide", "Gemcitabine", "Paclitaxel", "Temozolomide"])