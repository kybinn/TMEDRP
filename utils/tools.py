from collections import OrderedDict
from ogb.utils import smiles2graph
import numpy as np
import torch
from torch_geometric.data import Data


from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
from rdkit.Chem import rdFingerprintGenerator
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 防止国内网络问题
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import config

def smiles_to_chemberta_embeddings(
    smiles_list, device="cuda", batch_size=1, pooling="cls"
):
    """
    将 SMILES 序列转换为 ChemBERTa embedding。
    输出形状: (N, hidden_size)
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_ID
    # 加载预训练模型
    model_name = "utils/ChemBERTa-77M-MTR"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(
            range(0, len(smiles_list), batch_size), desc="ChemBERTa embedding"
        ):
            batch = smiles_list[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=256,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state  # (B, L, H)

            if pooling == "cls":
                emb = hidden_states[:, 0, :]  # [CLS] token
            else:
                emb = hidden_states.mean(dim=1)

            all_embeddings.append(emb.cpu().numpy())

    all_embeddings = np.vstack(all_embeddings)
    print(
        f"[ChemBERTa] Finished encoding {len(smiles_list)} molecules | dim={all_embeddings.shape[1]}"
    )
    return all_embeddings



def get_drug_ecfp(smiles, radius=2, n_bits=2048):
    """
    特征一：提取药物的结构特征 (用于预测药物对基因/蛋白的作用)
    返回 ECFP4 指纹。
    """
    _morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, 
        fpSize=2048, 
        includeChirality=True
    )
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # 生成 Morgan 指纹 (ECFP)
        fp = _morgan_gen.GetFingerprint(mol)
        return np.array(fp).astype('float32')
    except:
        return None

def get_drug_tme_features(smiles):
    """
    特征二：提取药物与微环境相关的理化性质 (用于预测药物在临床环境中的表现)
    返回一个包含 LogP, TPSA, 分子量, 电荷等特征的字典。
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        
        # 计算微环境相关的核心指标
        features = {
            # 1. 脂水分配系数 (LogP): 决定药物在间质(Stroma)中的分布和细胞膜穿透力
            "MolLogP": Descriptors.MolLogP(mol),
            
            # 2. 极性表面积 (TPSA): 决定药物对酸性环境(pH)的敏感度和渗透性
            "TPSA": Descriptors.TPSA(mol),
            
            # 3. 分子量 (MolWt): 决定药物在致密肿瘤组织中的扩散系数
            "MolWt": Descriptors.MolWt(mol),
            
            # 4. 可旋转键数量: 反应分子的柔性，影响在复杂基质中的运动
            "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
            
            # 5. 氢键受体/供体数量: 影响药物与微环境蛋白质/水分子的结合
            "HAcceptors": Descriptors.NumHAcceptors(mol),
            "HDonors": Descriptors.NumHDonors(mol)
        }
        return features
    except:
        return None

### 以上是20260202新加入

##### 把单个SMILES字符串转换成图数据 #####
def graph_from_smile(smiles):
    graph = smiles2graph(smiles)  # 假设返回 {'edge_index': Tensor, 'edge_feat': Tensor, 'node_feat': Tensor, 'num_nodes': int}
    result = {
        'edge_index': graph['edge_index'],
        'edge_attr': graph['edge_feat'],
        'x': graph['node_feat']
    }
    result = {k: torch.from_numpy(v) for k, v in result.items()}
    # 将一个 NumPy 的 ndarray（多维数组）​​ ​直接转换为 PyTorch 的 Tensor（张量）
    return Data(**result)

def read_gmt(fname, sep='\t', min_g=0, max_g=5000):
    # 读取一个 GMT 文件，并将其中的内容解析成一个 Python 字典（OrderedDict）
    # 只保留基因数量在 [min_g, max_g] 范围内的通路
    """
    Read GMT file into dictionary of gene_module:genes.\n
    min_g and max_g are optional gene set size filters.

    Args:
        fname (str): Path to gmt file
        sep (str): Separator used to read gmt file.
        min_g (int): Minimum of gene members in gene module.
        max_g (int): Maximum of gene members in gene module.
    Returns:
        OrderedDict: Dictionary of gene_module:genes.
    """
    dict_pathway = OrderedDict()
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            val = line.split(sep)
            if min_g <= len(val[2:]) <= max_g:
                dict_pathway[val[0]] = val[2:]
    return dict_pathway

def create_pathway_mask(feature_list, dict_pathway, add_missing=1, fully_connected=True, to_tensor=False):
    # 根据一组基因（feature_list）和一个通路-基因字典（dict_pathway），构建一个“通路-基因关联掩码矩阵（mask）”，形状为：[基因数量, 通路数量]
    """
    Creates a mask of shape [genes,pathways] where (i,j) = 1 if gene i is in pathway j, 0 else.

    Expects a list of genes and pathway dict.
    Note: dict_pathway should be an Ordered dict so that the ordering can be later interpreted.

    Args:
        feature_list (list): List of genes in single-cell dataset.
        dict_pathway (OrderedDict): Dictionary of gene_module:genes.
        add_missing (int): Number of additional, fully connected nodes.
        fully_connected (bool): Whether to fully connect additional nodes or not.
        to_tensor (False): Whether to convert mask to tensor or not.
    Returns:
        torch.tensor/np.array: Gene module mask.
    """
    assert type(dict_pathway) == OrderedDict
    p_mask = np.zeros((len(feature_list), len(dict_pathway)))
    pathway = list()
    for j, k in enumerate(dict_pathway.keys()):
        pathway.append(k)
        for i in range(p_mask.shape[0]):
            if feature_list[i] in dict_pathway[k]:
                p_mask[i,j] = 1.
    if add_missing:
        n = 1 if type(add_missing)==bool else add_missing
        # Get non connected genes
        if not fully_connected:
            idx_0 = np.where(np.sum(p_mask, axis=1)==0)
            vec = np.zeros((p_mask.shape[0],n))
            vec[idx_0,:] = 1.
        else:
            vec = np.ones((p_mask.shape[0], n))
        p_mask = np.hstack((p_mask, vec))
        for i in range(n):
            x = 'node %d' % i
            pathway.append(x)
    if to_tensor:
        p_mask = torch.Tensor(p_mask)
    return p_mask,np.array(pathway)

def get_mask(gmt_path,mask_col,genes):
    ##########加载 mask 行是基因，列是通路##########
    mask_ratio = 0.015
    if gmt_path is None:
        mask = np.random.binomial(1,mask_ratio,size=(genes, mask_col)) #(1047, 106)
        # 二项分布随机采样函数，生成一个二维数组，有 args.TME_dim_init行，mask_col列，1 的概率为 mask_ratio
        pathway = list()
        for i in range(mask_col):
            x = 'node %d' % i
            pathway.append(x)
        print('Full connection!')
    else:
        # gmt_path = '../'+gmt_path
        ## 下面全要改 TODO
        print('Loading mask from gmt file...')
        path_dict = read_gmt(gmt_path) # key是通路名，value是基因列表 3262个通路
        mask,pathway = create_pathway_mask(feature_list=genes,
                                          dict_pathway=path_dict)
        # 保留那些与较多基因有关联的通路，并只选择最重要的若干个通路（比如 top N 个），最终得到精简后的 mask 和 pathway 列表
        pathway = pathway[np.sum(mask,axis=0)>4]
        mask = mask[:,np.sum(mask,axis=0)>4]
        #print(mask.shape) # (5205, 3245)
        # 这些通路最能体现病人样本与细胞系之间的系统性差异
        
        pathway = pathway[sorted(np.argsort(np.sum(mask,axis=0))[-min(mask_col,mask.shape[1]):])]
        mask = mask[:,sorted(np.argsort(np.sum(mask,axis=0))[-min(mask_col,mask.shape[1]):])]
        # 取出mask中不为0的行的索引
        index_nonzero = np.where(np.sum(mask, axis=1)>0)[0]
        mask = mask[index_nonzero,:]
        #print(mask.shape) 
        # mask.shape (5055, 106)
        print('Mask loaded!')
    return torch.Tensor(mask), pathway

def set_seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

