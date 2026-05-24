import os
import torch
import numpy as np
from models import extractionForCancer,extractionForTME,Discriminator,Decoder,ReverseLayerF
import config
from itertools import chain
from itertools import cycle
from torch.nn import functional as F
from utils.loss_functions import Adversarial_loss,coral,ortho_loss


def eval_epoch(lam1,cancer_encoder, tme_encoder,TME_cancer_encoder,decoder_cancer,decoder_TME, discriminator1,loader1, loader2, device):
    total_loss = 0

    for i, batch in enumerate(zip(loader2, cycle(loader1))):
        s_batch=batch[1]
        t_batch=batch[0]
        s_gene_all = s_batch[0].to(device)    
        s_gene_cancer = s_batch[1].to(device) 
        s_gene_tme = s_batch[2].to(device)    
        s_gene_tme_cancer = s_batch[3].to(device)   
        t_gene_all = t_batch[0].to(device)
        t_gene_cancer = t_batch[1].to(device)
        t_gene_tme = t_batch[2].to(device)
        t_genetme_cancer = t_batch[3].to(device)

        with torch.no_grad():
            s_cancer = cancer_encoder(s_gene_cancer)                 
            s_domain_output1 = discriminator1(s_cancer)            
            s_tme, s_attn_weights = tme_encoder(s_gene_tme)        
            s_tme_can, s_attn_weights_can = TME_cancer_encoder(s_gene_tme_cancer)         
            s_fea_rec1 = decoder_cancer(s_cancer)                      
            s_fea_rec2 = decoder_TME(torch.cat([s_tme_can,s_tme], dim=1))                        

            t_cancer = cancer_encoder(t_gene_cancer)
            t_domain_output1 = discriminator1(t_cancer)
            t_tme, t_attn_weights = tme_encoder(t_gene_tme)
            t_tme_can, t_attn_weights_can = TME_cancer_encoder(t_genetme_cancer)
            t_fea_rec1 = decoder_cancer(t_cancer)
            t_fea_rec2 = decoder_TME(torch.cat([t_tme_can,t_tme], dim=1))
 
            # reconstruct loss
            rec_s = F.mse_loss(s_gene_cancer, s_fea_rec1)+ F.mse_loss(s_gene_tme, s_fea_rec2)
            rec_t = F.mse_loss(t_gene_cancer, t_fea_rec1)+ F.mse_loss(t_gene_tme, t_fea_rec2)
            recon_loss = rec_s + rec_t            
            # domain loss
            transfer_loss1 = Adversarial_loss(s_domain_output1, t_domain_output1) 
            orthloss = ortho_loss(s_tme, t_tme)
            coral_loss = lam1 * coral(s_tme_can, t_tme_can)
            domain_loss = transfer_loss1 + 0.1*orthloss +  coral_loss            
            loss = recon_loss + domain_loss
            total_loss += loss.cpu().detach().item() 
            
    return total_loss


def train_step(alpha,lam1,cancer_encoder, tme_encoder,TME_cancer_encoder,decoder_cancer,decoder_TME, discriminator1,s_batch, t_batch, device, optimizer):
    s_gene_all = s_batch[0].to(device)    
    s_gene_cancer = s_batch[1].to(device) 
    s_gene_tme = s_batch[2].to(device)    
    s_gene_tme_cancer = s_batch[3].to(device)    
    t_gene_all = t_batch[0].to(device)
    t_gene_cancer = t_batch[1].to(device)
    t_gene_tme = t_batch[2].to(device)
    t_genetme_cancer = t_batch[3].to(device)

    s_cancer = cancer_encoder(s_gene_cancer)                 
    s_reverse_feature = ReverseLayerF.apply(s_cancer, alpha) 
    s_domain_output1 = discriminator1(s_reverse_feature)    
    s_tme, s_attn_weights = tme_encoder(s_gene_tme)         
    s_tme_can, s_attn_weights_can = TME_cancer_encoder(s_gene_tme_cancer)        
    s_fea_rec1 = decoder_cancer(s_cancer)                       
    s_fea_rec2 = decoder_TME(torch.cat([s_tme_can,s_tme], dim=1))                    

    t_cancer = cancer_encoder(t_gene_cancer)
    t_reverse_feature = ReverseLayerF.apply(t_cancer, alpha) 
    t_domain_output1 = discriminator1(t_reverse_feature)
    t_tme, t_attn_weights = tme_encoder(t_gene_tme)
    t_tme_can, t_attn_weights_can = TME_cancer_encoder(t_genetme_cancer)
    t_fea_rec1 = decoder_cancer(t_cancer)
    t_fea_rec2 = decoder_TME(torch.cat([t_tme_can,t_tme], dim=1))


    # reconstruct loss
    rec_s = F.mse_loss(s_gene_cancer, s_fea_rec1)+ F.mse_loss(s_gene_tme, s_fea_rec2)
    rec_t = F.mse_loss(t_gene_cancer, t_fea_rec1)+ F.mse_loss(t_gene_tme, t_fea_rec2)
    recon_loss = rec_s + rec_t
    # domain loss
    transfer_loss1 = Adversarial_loss(s_domain_output1, t_domain_output1) 
    ortholoss = ortho_loss(s_tme, t_tme)
    coral_loss = lam1 * coral(s_tme_can, t_tme_can)
        
    domain_loss = transfer_loss1 + 0.1*ortholoss + coral_loss
    loss = recon_loss + domain_loss
    loss.backward()
    optimizer.step()
    return loss.cpu().detach().item() 


def training(s_dataloaders, t_dataloaders,mask_tme,mask_can, **kwargs):
    s_train = s_dataloaders[0]
    s_test = s_dataloaders[1]
    t_train = t_dataloaders[0]
    t_test = t_dataloaders[1]
    
    cancer_encoder = extractionForCancer(dim_input=kwargs['cancer_genes'], 
                                         dim_hid1=kwargs['cancer_encoder_hidden_dims'][0],
                                         dim_hid2=kwargs['cancer_encoder_hidden_dims'][1], 
                                         dim_final=kwargs['cancer_encoder_hidden_dims'][2]).to(kwargs['device'])   

    tme_encoder = extractionForTME(mask_tme, embed_dim=mask_tme.size(1),depth=2,num_heads=2).to(kwargs['device'])  
    TME_cancer_encoder = extractionForTME(mask_can, embed_dim=mask_can.size(1),depth=2,num_heads=2).to(kwargs['device'])  
    discriminator1 = Discriminator(input_dim=kwargs['cancer_encoder_hidden_dims'][2]).to(kwargs['device'])

    decoder_cancer = Decoder(input_dim = kwargs['cancer_encoder_hidden_dims'][2],
                      output_dim = kwargs['cancer_genes'],
                      hidden_dims = kwargs['decoder_hidden_dims'],
                      drop = kwargs['drop']).to(kwargs['device'])
    decoder_TME = Decoder(input_dim = kwargs['n_pathways']+kwargs['n_cancer_pathways'],
                      output_dim = kwargs['TME_genes'],
                      hidden_dims = kwargs['decoder_hidden_dims'],
                      drop = kwargs['drop']).to(kwargs['device'])
    
    
    params = [cancer_encoder.parameters(),
              tme_encoder.parameters(),
              TME_cancer_encoder.parameters(),
              decoder_cancer.parameters(),
              decoder_TME.parameters(),
              discriminator1.parameters()]
    optimizer = torch.optim.AdamW(chain(*params), lr=kwargs['lr'])
    best_threshold = np.inf


    print("================Pre-training for feature extraction================")
    if kwargs['retrain_flag']:  
        for epoch in range(int(kwargs['pretrain_num_epochs'])):
            train_loss_all = 0
            val_loss_all = 0
            len_loader = min(len(s_train), len(t_train))  
            cancer_encoder.train()
            tme_encoder.train()
            TME_cancer_encoder.train()
            decoder_cancer.train()
            decoder_TME.train()
            discriminator1.train()
            
            optimizer.zero_grad()
            for i, batch in enumerate(zip(t_train, cycle(s_train))):
                p = float(i + epoch * len_loader) / int(kwargs['pretrain_num_epochs']) / len_loader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                train_loss = train_step(alpha,kwargs['lam1'],cancer_encoder, tme_encoder,TME_cancer_encoder,
                                        decoder_cancer,decoder_TME, discriminator1, 
                                        s_batch=batch[1], t_batch=batch[0],
                                        device=kwargs['device'], optimizer=optimizer)
                train_loss_all += train_loss
            print('Pre-training epoch = {}, training loss = {:.4f}'.format(epoch+1, train_loss_all))

         
            cancer_encoder.eval()
            tme_encoder.eval()
            TME_cancer_encoder.eval()
            decoder_cancer.eval()
            decoder_TME.eval()
            discriminator1.eval()

            val_loss = eval_epoch(kwargs['lam1'],cancer_encoder, tme_encoder,TME_cancer_encoder,decoder_cancer,decoder_TME, discriminator1,loader1=s_test, loader2=t_test,device=kwargs['device'])
            val_loss_all = val_loss
            print('validate loss = {:.4f}'.format(val_loss_all))
           
           
            if (val_loss_all < best_threshold):
                torch.save(cancer_encoder.state_dict(), os.path.join(kwargs['model_save_folder'], 'cancer_encoder.pt'))
                torch.save(tme_encoder.state_dict(), os.path.join(kwargs['model_save_folder'], 'tme_encoder.pt'))   
                torch.save(TME_cancer_encoder.state_dict(), os.path.join(kwargs['model_save_folder'], 'TME_cancer_encoder.pt')) 
                best_threshold = val_loss_all
            
        cancer_encoder.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'cancer_encoder.pt')))
        tme_encoder.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'tme_encoder.pt')))
        TME_cancer_encoder.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'TME_cancer_encoder.pt')))
    else:
        try:
            cancer_encoder.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'cancer_encoder.pt')))
            tme_encoder.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'tme_encoder.pt')))
            TME_cancer_encoder.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'TME_cancer_encoder.pt')))
        except FileNotFoundError:
            raise Exception("No pre-trained encoder")


    return cancer_encoder, tme_encoder,TME_cancer_encoder

