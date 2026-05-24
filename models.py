import torch
import torch.nn as nn
from torch.nn import functional as F
from copy import deepcopy
import numpy as np
from torch.autograd import Function
from utils.transformer_layer import Block
from utils.customized_linear import CustomizedLinear
from einops import rearrange
from functools import partial
import copy
from collections import OrderedDict


def random_zero(tensor, probability):
    mask = torch.bernoulli(torch.full_like(tensor, probability))
    return tensor * mask


def get_weight(att_mat,device):
    att_mat = torch.stack(att_mat).squeeze(1)
    #print(att_mat.size())
    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=2)
    #print(att_mat.size())
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(3))
    aug_att_mat = att_mat.to(device) + residual_att.to(device)
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    #print(aug_att_mat.size())
    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
    joint_attentions[0] = aug_att_mat[0]
    
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    #print(joint_attentions.size())
    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    #print(v.size())
    v = v[:,0,1:]
    #print(v.size())
    return v

def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)  


class FeatureEmbed(nn.Module):
    def __init__(self, mask, embed_dim=192, fe_bias=True, norm_layer=None):
        super().__init__()
        self.num_patches = mask.shape[1]
        self.embed_dim = embed_dim
        mask = np.repeat(mask,embed_dim,axis=1) 
        self.mask = mask
        self.fe = CustomizedLinear(self.mask)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def forward(self, x):
        num_samples = x.shape[0]
        x = rearrange(self.fe(x), 'h (w c) -> h c w ', c=self.num_patches)
        x = self.norm(x)
        return x


class Transformer(nn.Module):
    def __init__(self, mask, fe_bias=True,
                 embed_dim=128, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=FeatureEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            num_classes (int): number of classes for classification head
            num_genes (int): number of feature of input(expData) 
            embed_dim (int): embedding dimension
            depth (int): depth of transformer 
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT modelss
            drop_ratio (float): dropout rate 
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): feature embed layer
            norm_layer: (nn.Module): normalization layer
        """
        super(Transformer, self).__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.feature_embed = embed_layer(mask = mask, embed_dim=embed_dim, fe_bias=fe_bias)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.ModuleList()
        for i in range(depth):
            layer = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                          norm_layer=norm_layer, act_layer=act_layer)
            self.blocks.append(copy.deepcopy(layer))
        self.norm = norm_layer(embed_dim)
        if representation_size:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity() 
        # Weight init
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        x = self.feature_embed(x) 
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1) 
        attn_weights = []
        tem = x
        for layer_block in self.blocks:
            tem, weights = layer_block(tem)
            attn_weights.append(weights)
        x = self.norm(tem)
        device = x.device
        return self.pre_logits(x[:, 0]), attn_weights 
    

    def forward(self, x):
        latent, attn_weights = self.forward_features(x)        
        return latent, attn_weights


class extractionForCancer(nn.Module):
    def __init__(self, dim_input, dim_hid1, dim_hid2, dim_final):
        super(extractionForCancer, self).__init__()
        mlp_layers = [
            nn.Linear(dim_input, dim_hid1),
            nn.ReLU(),
            nn.Linear(dim_hid1, dim_hid2),
            nn.ReLU(),
            nn.Linear(dim_hid2, dim_final)
        ]
        self.mlp = nn.Sequential(*mlp_layers)
        
    def forward(self, RNAseq):
        RNA_embedding = self.mlp(RNAseq)
        
        return RNA_embedding


class CoExpressionAttention(nn.Module):
    def __init__(self, input_dim):
        super(CoExpressionAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        
    def forward(self, x):
        q = self.query(x).unsqueeze(2) 
        k = self.key(x).unsqueeze(1)   
        
        attention_matrix = torch.matmul(q, k) # Outer Product
        attention_matrix = F.softmax(attention_matrix, dim=-1) 
        return attention_matrix


class extractionForTME(nn.Module):
    def __init__(self,mask, embed_dim=128,depth=2,num_heads=2,has_logits: bool = True):
        super(extractionForTME, self).__init__()
        self.geneTransformer = Transformer( 
                        mask = mask,
                        embed_dim=embed_dim,
                        depth=depth,
                        num_heads=num_heads,
                        drop_ratio=0.5, attn_drop_ratio=0.5, drop_path_ratio=0.5,
                        representation_size=embed_dim if has_logits else None)
              
    def forward(self, TME):
        TME_embedding, attn_weights = self.geneTransformer(TME)
        return TME_embedding, attn_weights


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None, drop=0.1, act_fn=nn.SELU, **kwargs):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.drop = drop
        hidden_dims = deepcopy(hidden_dims)
        hidden_dims.insert(0, input_dim)
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=True),
                    act_fn(),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.Dropout(self.drop))
            )
        self.module = nn.Sequential(*modules)
        self.output_layer = nn.Sequential(nn.Linear(hidden_dims[-1], output_dim, bias=True))
        self.reset_para()

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return

    def forward(self, inputs):
        embed = self.module(inputs)
        output = self.output_layer(embed)
        return output


# Models for training
class DRPmodel(nn.Module):
    def __init__(self, cancer_encoder, tme_encoder,TME_cancer_encoder,emb_dim):
        super(DRPmodel, self).__init__()
        self.cancer_encoder = cancer_encoder
        self.tme_encoder = tme_encoder
        self.TME_cancer_encoder = TME_cancer_encoder
        mlp_layers = [nn.Linear(20, emb_dim),
                        nn.ReLU()]
        self.mlp = nn.Sequential(*mlp_layers)
        self.coExpressionAttention = CoExpressionAttention(emb_dim*2)


        self.crosssAtte1 = nn.MultiheadAttention(emb_dim, 1, dropout=0.2, batch_first=True)
        self.crosssAtte2 = nn.MultiheadAttention(emb_dim, 1, dropout=0.2, batch_first=True)
        self.crosssAtte3 = nn.MultiheadAttention(emb_dim*3 , 2, dropout=0.2, batch_first=True)
        self.crosssAtte4 = nn.MultiheadAttention(emb_dim*2 , 2, dropout=0.2, batch_first=True)
        

        self.output_layer_CLDRP = nn.Linear(emb_dim*3, 1) 
        self.output_layer_TMEDRP = nn.Linear(emb_dim*2, 1)

        self.drug_mapping = nn.Linear(384, emb_dim)  
        
    def forward(self, geneCancer,geneTME,tmegene_cancer,drug_embedding, return_split=False):
        cancer_embedding = self.cancer_encoder(geneCancer) 
        TME_embedding, attn_weights = self.tme_encoder(geneTME)
        TME_gene_cancer_embedding, attn_weights2 = self.TME_cancer_encoder(tmegene_cancer)
        TME_embedding = self.mlp(TME_embedding)
        drug_embedding = self.drug_mapping(
            torch.tensor(drug_embedding).to(cancer_embedding.device)
        )

        concat_cancer = torch.cat([cancer_embedding,TME_gene_cancer_embedding], dim=-1)   
        coExpression = self.coExpressionAttention(concat_cancer)
        coExpression_cancer = torch.matmul(coExpression,concat_cancer.unsqueeze(2)).squeeze(2) 
        combined_CL_drug = torch.cat([coExpression_cancer, drug_embedding], dim=-1) 
        # combined_CL_drug = torch.cat([concat_cancer, drug_embedding], dim=-1) 
        pred_CL_drug = self.output_layer_CLDRP(combined_CL_drug).squeeze(-1) 


        emb_dim = cancer_embedding.shape[1]
        TME_cell = self.crosssAtte1(TME_embedding,cancer_embedding,cancer_embedding)[0] 
        TME_drug = self.crosssAtte2(TME_embedding,drug_embedding,drug_embedding)[0]
        TME_cross = torch.cat([TME_cell, TME_drug], dim=-1) 
        TME_end = self.crosssAtte4(TME_cross,TME_cross,TME_cross)[0] 
        pred_TME_drug = self.output_layer_TMEDRP(TME_end).squeeze(-1) 

        
        if return_split:
            return pred_CL_drug, pred_TME_drug
        
        final_pred = 0.5 * pred_CL_drug + 0.5 * pred_TME_drug 

        return final_pred


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential()
        self.layer.add_module('fc1', nn.Linear(input_dim, input_dim//4))
        self.layer.add_module('bn1', nn.BatchNorm1d(input_dim//4))
        self.layer.add_module('relu1', nn.SELU(True))
        self.layer.add_module('fc2', nn.Linear(input_dim//4, input_dim//8))
        self.layer.add_module('bn2', nn.BatchNorm1d(input_dim//8))
        self.layer.add_module('relu2', nn.SELU(True))
        self.layer.add_module('fc3', nn.Linear(input_dim//8, 1))

    def forward(self, input_data):
        output = self.layer(input_data)
        return output
    
