import json
import os
from pronto import Ontology
import pickle
from tqdm import tqdm
import logging
from sentence_transformers import SentenceTransformer
import pandas as pd
from datetime import datetime

from .utils import get_default_save_path
from .logging_utils import get_logger

logger = get_logger(__name__)

class ST_Builder():
    def __init__(
        self,
        hp_obo_path: str,
        model: str,
        save_path: str | None,
        hp_addon_path: str | None = None,
    ):
        self.SAVE_PATH = save_path  or get_default_save_path()
        
        # 验证HPO文件是否存在
        if not os.path.exists(hp_obo_path):
            raise FileNotFoundError(f"Can not find hpo.obo: {hp_obo_path}")
        
        # Load HPO Ontology
        self.HP_OBO_PATH = hp_obo_path
        self.HPO = Ontology(handle=hp_obo_path)
        
        # addon
        self.HP_ADDON_PATH = hp_addon_path
        
        # Preprocess root nodes
        self.ROOT = [term.id for term in self.HPO["HP:0000001"].subclasses(distance=1)]
        self.term_root = {}
        for term in self.HPO.terms():
            term_id = term.id
            if term_id == 'HP:0000001':
                self.term_root[term_id] = ''
                continue
            for parent in term.superclasses():
                if parent.id in self.ROOT:
                    self.term_root[term_id] = parent.id
                    break
        
        # Load Sentence Transformer model
        self.model_name = model
        self.Model = SentenceTransformer(model)

    
    def _extract_hpo_terms(self):
        """
        从HPO本体中提取相关术语信息
        
        Returns:
            list: 包含HPO术语信息的字典列表
        """
        hpo_terms = []
        
        for term in tqdm(self.HPO.terms(), desc='Extracting HPO terms'):
            term_id = term.id
            if not term_id.startswith("HP:"):
                continue

            # 只处理表型异常相关的术语
            root_id = self.term_root.get(term_id, '')
            if root_id != 'HP:0000118':  # Phenotypic abnormality
                continue

            # 提取术语基本信息
            term_info = {
                'id': term_id,
                'name': term.name,
                'definition': str(term.definition),
                'synonyms': [s.description for s in term.synonyms],
                'umls_id': [xref.id for xref in term.xrefs if xref.id.startswith('UMLS')],
                'root_id': root_id,
                'alt_ids': list(term.alternate_ids)
            }
            hpo_terms.append(term_info)

        logger.info(f"Extracted {len(hpo_terms)} HPO terms")
        return hpo_terms

    def _load_addon_terms(self):
        """
        加载附加HPO术语信息
        
        Returns:
            pd.DataFrame or None: 附加术语数据框，如果没有则返回None
        """
        if not self.HP_ADDON_PATH:
            return None
            
        try:
            addon_df = pd.read_csv(self.HP_ADDON_PATH)
            logger.info(f"Loaded {len(addon_df)} addon HPO records")
            return addon_df
        except Exception as e:
            logger.warning(f"Failed to load addon HPO file: {e}")
            return None

    def _create_text_records(self, hpo_terms, addon_df=None):
        """
        创建用于向量化的文本记录
        
        Args:
            hpo_terms (list): HPO术语列表
            addon_df (pd.DataFrame, optional): 附加术语数据框
            
        Returns:
            tuple: (record_ids, texts) 记录ID列表和对应的文本列表
        """
        records = []
        record_idx = 0
        
        # 处理标准HPO术语
        for term in tqdm(hpo_terms, desc="Processing HPO terms"):
            term_id = term['id']
            
            # 处理术语名称
            if term['name']:
                records.append((f"{record_idx}_{term_id}_name", term['name']))
                record_idx += 1
            
            # 处理同义词
            for synonym in term['synonyms']:
                if synonym.strip():  # 确保同义词不为空
                    records.append((f"{record_idx}_{term_id}_synonym", synonym))
                    record_idx += 1
            
            # 处理定义
            if term['definition'] and term['definition'] != 'None':
                records.append((f"{record_idx}_{term_id}_definition", term['definition']))
                record_idx += 1
        
        # 处理附加术语
        if addon_df is not None:
            for _, row in addon_df.iterrows():
                if 'HP_ID' in row and 'info' in row and row['info']:
                    records.append((f"{record_idx}_{row['HP_ID']}_addon", row['info']))
                    record_idx += 1
        
        record_ids = [r[0] for r in records]
        texts = [r[1] for r in records]
        
        logger.info(f"Created {len(records)} text records for embedding")
        return record_ids, texts

    def _generate_embeddings(self, texts, batch_size:int = 256):
        """
        生成文本嵌入向量
        
        Args:
            texts (list): 待编码的文本列表
            batch_size (int): 批处理大小
            
        Returns:
            torch.Tensor: 嵌入向量张量
        """
        try:
            embeddings = self.Model.encode(
                texts, 
                batch_size=batch_size, 
                show_progress_bar=False, 
                convert_to_tensor=True
            )
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    def _save_embeddings_matrix(self, record_ids, embeddings, record_text):
        output_file = os.path.join(self.SAVE_PATH, 'hpo_embeddings_matrix.pkl')
        
        try:
            with open(output_file, "wb") as f:
                pickle.dump({
                    "hpo_ids": record_ids, 
                    "hpo_text": record_text,
                    "embeddings": embeddings
                }, f)
            logger.info(f"Saved embeddings matrix to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save embeddings matrix: {e}")
            raise

    def generate_hpo_matrix(self):
        try:
            # 1. 提取HPO术语
            hpo_terms = self._extract_hpo_terms()
            
            # 2. 加载附加术语
            addon_df = self._load_addon_terms()
            
            # 3. 创建文本记录
            record_ids, texts = self._create_text_records(hpo_terms, addon_df)
            
            if not texts:
                raise ValueError("No valid text records found for embedding generation")
            
            # 4. 生成嵌入向量
            embeddings = self._generate_embeddings(texts)
            
            # 5. 保存向量矩阵
            self._save_embeddings_matrix(record_ids, embeddings, texts)

            logger.info("HPO matrix generation completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to generate HPO matrix: {e}")
            raise

    def build(self):
        self.generate_hpo_matrix()
        # Information
        info_dict = {
            "build_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.model_name,
            "hpo_obo_path": self.HP_OBO_PATH
        }
        with open(os.path.join(self.SAVE_PATH, "database_info.json"), "w") as f:
            json.dump(info_dict, f, ensure_ascii=False, indent=4)
            
