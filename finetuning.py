# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import torch.nn as nn
import config
from models import DRPmodel
from torch_geometric.data import Batch
from utils.loss_functions import bina_metric
import pandas as pd
from itertools import cycle
from utils.loss_functions import coral,ortho_loss
from torch.nn import functional as F
def multi_eval_epoch(drug_embedding,model, loader, device):
    model.eval()
    total_loss = 0
    alpha = 0
    y_true, y_pred = [], []
    auc_list, aupr_list, acc_list, f1_list = [], [], [], []

    result_per_epoch = []
    labels_per_epoch = []
    for all,geneCancer,geneTME,gene_TME_can, labels,mask, _ in loader:       
        geneCancer = geneCancer.to(device)
        geneTME = geneTME.to(device)
        gene_TME_can = gene_TME_can.to(device)
        labels = labels.to(device)
        mask = mask.to(device)
        
        drug_inputs = []
        cancergene_inputs = []
        tmegene_inputs = []
        gene_TME_can_inputs = []
        labels_real = []          
        for n in range(mask.shape[1]):
            for m in range(mask.shape[0]):
                if mask[m][n] > 0:
                    cancergene_inputs.append(geneCancer[m])
                    tmegene_inputs.append(geneTME[m])
                    gene_TME_can_inputs.append(gene_TME_can[m])
                    drug_inputs.append(drug_embedding[n])
                    labels_real.append(labels[m][n])
        
        cancergene_inputs = torch.stack(cancergene_inputs, dim=0)
        tmegene_inputs = torch.stack(tmegene_inputs, dim=0)
        gene_TME_can_inputs = torch.stack(gene_TME_can_inputs, dim=0)
        labels_real = torch.stack(labels_real, dim=0)
        batched_graph = drug_inputs

        with torch.no_grad():
            predictions = model(cancergene_inputs,tmegene_inputs,gene_TME_can_inputs,batched_graph)
            probabilities = torch.sigmoid(predictions)
            loss  = nn.BCEWithLogitsLoss()(predictions,labels_real)
            total_loss += loss
            y_true.extend(labels_real.cpu().detach().numpy().tolist())
            y_pred.extend(probabilities.cpu().detach().numpy().tolist())


        result_in_batch = torch.zeros_like(labels)
        i = 0        
        for n in range(mask.shape[1]):
            for m in range(mask.shape[0]):
                if mask[m][n] > 0:
                    result_in_batch[m][n] = torch.sigmoid(predictions[i])
                    i+=1
        result_per_epoch.append(result_in_batch.cpu().detach().numpy())
        labels_per_epoch.append(labels.cpu().detach().numpy()) 


    result_per_epoch = np.vstack(result_per_epoch) 
    labels_per_epoch = np.vstack(labels_per_epoch)         
    y_true = np.array(y_true)   
    y_pred = np.array(y_pred)

         
    df_metrics = pd.DataFrame(columns=['Drug','Precision','F1','PRAUC','Accuracy','ROC_AUC'])
    for drug_idx in range(result_per_epoch.shape[1]):
        drug_results = result_per_epoch[:, drug_idx]
        drug_labels = labels_per_epoch[:, drug_idx]
        valid_indices = np.where(drug_labels != -1)[0]
        if(len(valid_indices) == 0):
            precision, recall, f1, prauc, roc_auc,accuracy = -1, -1, -1, -1, -1,-1
        else:
            drug_results = drug_results[valid_indices]
            drug_labels = drug_labels[valid_indices]
            precision, recall, f1, prauc, roc_auc,accuracy = bina_metric(drug_labels, (drug_results>=0.5).astype(int), drug_results)
        
        new_row = {'Drug': drug_idx,
                    'Precision': precision,
                    'F1': f1,
                    'PRAUC': prauc,
                    'Accuracy': accuracy,
                    'ROC_AUC': roc_auc}
        df_metrics = pd.concat([df_metrics, pd.DataFrame([new_row])], ignore_index=True)
        
    
    return total_loss, df_metrics , y_true, y_pred


def testing(drug_embedding,model, t_dataloader, device):
    model.train(False)  
    test_loss, results, y_true, y_pred = multi_eval_epoch(drug_embedding=drug_embedding,model=model,
                                          loader=t_dataloader,
                                          device=device) 
    precision, recall, f1, prauc, roc_auc,accuracy = bina_metric(y_true, (y_pred>=0.5).astype(int), y_pred)
    
    print(
        f"** TESTING on TCGA data**   "
        f"loss: {test_loss:.5f}, "
        f"Precision: {precision:.5f}, "
        f"Recall: {recall:.5f}, "
        f"F1: {f1:.5f}, "
        f"PRAUC: {prauc:.5f}, "
        f"ROC_AUC: {roc_auc:.5f}. "
    )
    return test_loss, results, y_true, y_pred




def training(drug_embedding,cancer_encoder,tme_encoder,TME_cancer_encoder,s_dataloader,val_dataloader,tcga_dataloader, **kwargs):
    model = DRPmodel(cancer_encoder, tme_encoder,TME_cancer_encoder,kwargs['latent_dim']).to(kwargs['device'])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=kwargs['lr'])
                                  
    best_loss_sum = np.inf
    val_rocauc = 0
    
            
    print("\n============Training for CCLE data================")
    best_epoch = 0
    for epoch in range(int(kwargs['train_num_epochs'])):
        result_per_epoch = []
        labels_per_epoch = []
        y_true, y_pred = [], []
        training_loss_sum = 0          
        coral_loss_sum = 0
        ortholoss_sum = 0      
        domain_loss_sum = 0
        loss_predic_sum = 0
        co_express_alignment_loss_sum = 0

        len_loader = len(s_dataloader) 
        model.train(True)
        optimizer.zero_grad()
        len_loader = min(len(s_dataloader), len(tcga_dataloader))  
        for i, batch in enumerate(zip(tcga_dataloader, cycle(s_dataloader))):
            s_batch = batch[1]
            tcga_batch = batch[0]
            geneCancer = s_batch[1].to(kwargs['device']) 
            geneTME = s_batch[2].to(kwargs['device'])
            geneTME_Cancer = s_batch[3].to(kwargs['device'])
            labels = s_batch[4].to(kwargs['device']) 
            mask = s_batch[5].to(kwargs['device']) 

            tcga_geneCancer = tcga_batch[1].to(kwargs['device']) 
            tcga_geneTME = tcga_batch[2].to(kwargs['device'])
            tcga_geneTME_Cancer = tcga_batch[3].to(kwargs['device'])
           

            s_tme, s_attn_weights = tme_encoder(geneTME)        
            s_tme_can, s_attn_weights_can = TME_cancer_encoder(geneTME_Cancer)        
            tcga_tme, tcga_attn_weights = tme_encoder(tcga_geneTME)         
            tcga_tme_can, tcga_attn_weights_can = TME_cancer_encoder(tcga_geneTME_Cancer)      
            lam1 = kwargs['lam1']
            
            ortholoss = ortho_loss(s_tme, tcga_tme)
            coral_loss = lam1 * coral(s_tme_can, tcga_tme_can)
            domain_loss = 0.01*ortholoss + 10*coral_loss  
                        
            coral_loss_sum += coral_loss.cpu().detach().item()
            ortholoss_sum += ortholoss.cpu().detach().item()
            domain_loss_sum += domain_loss.cpu().detach().item()



            drug_inputs = []
            cancergene_inputs = []
            tmegene_inputs = []
            tmegene_cancer_inputs = []
            labels_real = []        
            for n in range(mask.shape[1]):
                for m in range(mask.shape[0]):
                    if mask[m][n] > 0:
                        cancergene_inputs.append(geneCancer[m])
                        tmegene_inputs.append(geneTME[m])
                        tmegene_cancer_inputs.append(geneTME_Cancer[m])
                        drug_inputs.append(drug_embedding[n])
                        labels_real.append(labels[m][n])
            
            cancergene_inputs = torch.stack(cancergene_inputs, dim=0).to(kwargs['device']) 
            tmegene_inputs = torch.stack(tmegene_inputs, dim=0).to(kwargs['device']) 
            tmegene_cancer_inputs = torch.stack(tmegene_cancer_inputs, dim=0).to(kwargs['device'])
            labels_real = torch.stack(labels_real, dim=0).to(kwargs['device']) 
            batched_graph = drug_inputs

            predictions = model(cancergene_inputs,tmegene_inputs,tmegene_cancer_inputs,batched_graph)
            loss_predic  = nn.BCEWithLogitsLoss()(predictions,labels_real) 
            loss_predic_sum += loss_predic.cpu().detach().item()
            
            
            # coExpressionAttention
            s_cancer_embedding = model.cancer_encoder(geneCancer) 
            s_TME_cancer_embedding , att1 = model.TME_cancer_encoder(geneTME_Cancer)
            s_concat_cancer = torch.cat([s_cancer_embedding,s_TME_cancer_embedding], dim=-1)
            s_coExpression = model.coExpressionAttention(s_concat_cancer)
            tcga_cancer_embedding = model.cancer_encoder(tcga_geneCancer)
            tcga_TME_cancer_embedding,att2 = model.TME_cancer_encoder(tcga_geneTME_Cancer)
            tcga_concat_cancer = torch.cat([tcga_cancer_embedding,tcga_TME_cancer_embedding], dim=-1)
            tcga_coExpression = model.coExpressionAttention(tcga_concat_cancer)

            co_express_alignment_loss = 0
            s_cancerType = s_batch[6]
            t_cancerType = tcga_batch[6]
            common_values = set(s_cancerType.tolist()) & set(t_cancerType.tolist())
            value_to_indices_map = {}
            for value in common_values:
                indices_in_a = (s_cancerType == value).nonzero(as_tuple=True)[0].tolist()
                indices_in_b = (t_cancerType == value).nonzero(as_tuple=True)[0].tolist()
                value_to_indices_map[value] = {
                    'a_indices': indices_in_a,
                    'b_indices': indices_in_b
                }
            for value, indices_dict in value_to_indices_map.items():
                indices_a = indices_dict.get('a_indices', [])
                indices_b = indices_dict.get('b_indices', [])                
                mean_a = None
                mean_b = None
                if indices_a: 
                    values_from_a = s_coExpression[indices_a]
                    mean_a = torch.mean(values_from_a,dim=0) 
                if indices_b: 
                    values_from_b = tcga_coExpression[indices_b]
                    mean_b = torch.mean(values_from_b,dim=0)                     
                mse = F.mse_loss(mean_a, mean_b)
                co_express_alignment_loss += mse 
            
            co_express_alignment_loss_sum += co_express_alignment_loss.cpu().detach().item()


            # MC Dropout 
            batch_loss_tme_sentry = 0
            for d_idx in range(len(drug_embedding)):
                current_drug_emb = torch.tensor(drug_embedding[d_idx]).to(kwargs['device']).unsqueeze(0)
                tcga_drug_inputs = current_drug_emb.repeat(tcga_geneCancer.shape[0], 1)

                model.train() 
                _, p2_v1 = model(tcga_geneCancer, tcga_geneTME, tcga_geneTME_Cancer, tcga_drug_inputs, return_split=True)
                _, p2_v2 = model(tcga_geneCancer, tcga_geneTME, tcga_geneTME_Cancer, tcga_drug_inputs, return_split=True)
                prob2_v1 = torch.sigmoid(p2_v1)
                prob2_v2 = torch.sigmoid(p2_v2)
                consistent_mask = (prob2_v1 > 0.5) == (prob2_v2 > 0.5)
                avg_prob2 = (prob2_v1 + prob2_v2) / 2
                tme_entropy = -(avg_prob2 * torch.log(avg_prob2 + 1e-12) + (1 - avg_prob2) * torch.log(1 - avg_prob2 + 1e-12))
                loss_curr_drug = 0
                if consistent_mask.any():
                    loss_curr_drug += torch.mean(tme_entropy[consistent_mask])     
                if (~consistent_mask).any():
                    loss_curr_drug -= torch.mean(tme_entropy[~consistent_mask])                  
                batch_loss_tme_sentry += loss_curr_drug
            loss_tme_sentry = batch_loss_tme_sentry / len(drug_embedding)


            lambda_ent = 0.1
            loss = loss_predic + domain_loss + co_express_alignment_loss + lambda_ent * loss_tme_sentry
            loss.backward()
            optimizer.step()
            training_loss_sum += loss.cpu().detach().item()
            
            y_true.extend(labels_real.cpu().detach().numpy().tolist())
            probabilities =  torch.sigmoid(predictions)
            y_pred.extend(probabilities.cpu().detach().numpy().tolist())

            
            
            result_in_batch = torch.zeros_like(labels)
            i = 0           
            for n in range(mask.shape[1]):
                for m in range(mask.shape[0]):
                    if mask[m][n] > 0:
                        result_in_batch[m][n] = torch.sigmoid(predictions[i])
                        i+=1
            result_per_epoch.append(result_in_batch.cpu().detach().numpy())
            labels_per_epoch.append(labels.cpu().detach().numpy())

        result_per_epoch = np.vstack(result_per_epoch)
        labels_per_epoch = np.vstack(labels_per_epoch)
        for drug_idx in range(result_per_epoch.shape[1]):
            drug_results = result_per_epoch[:, drug_idx]
            drug_labels = labels_per_epoch[:, drug_idx]
            precision, recall, f1, prauc, roc_auc,accuracy = bina_metric(drug_labels, (drug_results>=0.5).astype(int), drug_results)

        y_true = np.array(y_true)   
        y_pred = np.array(y_pred)
        
        precision, recall, f1, prauc, roc_auc,accuracy = bina_metric(y_true, (y_pred>=0.5).astype(int), y_pred) 

        print(
            f"Epoch {epoch+1}: "
            f"training loss: {training_loss_sum:.5f}, "
            f"Precision: {precision:.5f}, "
            f"Recall: {recall:.5f}, "
            f"F1: {f1:.5f}, "
            f"PRAUC: {prauc:.5f}, "
            f"ROC_AUC: {roc_auc:.5f}. "
        )



        model.eval()
        val_loss, _, val_y_true, val_y_pred = multi_eval_epoch(drug_embedding=drug_embedding,model=model,loader=val_dataloader,device=kwargs['device'])
        val_precision, val_recall, val_f1, val_prauc, val_roc_auc,val_accuracy = bina_metric(val_y_true, (val_y_pred>=0.5).astype(int), val_y_pred)
        print(
            f"validate loss: {val_loss:.5f}, "
            f"Precision: {val_precision:.5f}, "
            f"Recall: {val_recall:.5f}, "
            f"F1: {val_f1:.5f}, "
            f"PRAUC: {val_prauc:.5f}, "
            f"ROC_AUC: {val_roc_auc:.5f}. "
        )
        if (val_roc_auc > val_rocauc):
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(kwargs['model_save_folder'], 'DRPmodel.pt'))    
            val_rocauc = val_roc_auc

        
        if epoch - best_epoch > 50:
            break
    
    model.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'DRPmodel.pt')))
    
    return model

