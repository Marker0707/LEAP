import json
import os
from pronto import Ontology
from tqdm import tqdm
import logging
from sentence_transformers import SentenceTransformer
import pandas as pd
from datetime import datetime
from sklearn import preprocessing
from .utils import get_default_save_path

class ST_Builder():
    def __init__(
        self,
        hp_obo_path: str,
        model: str,
        save_path: str,
        hp_addon_path: str = "",
    ):
        self.logger = logging.getLogger(__name__)
        self.SAVE_PATH = save_path  or get_default_save_path()
        
        # 验证HPO文件是否存在
        if not os.path.exists(hp_obo_path):
            raise FileNotFoundError(f"Can not find hpo.obo: {hp_obo_path}")
        try:
            os.makedirs(self.SAVE_PATH, exist_ok=True)
            test_file = os.path.join(self.SAVE_PATH, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            raise PermissionError(f"Permission Error: {self.SAVE_PATH}: {e}")
        
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


    def generate_database(self):
        """
        生成HPO数据库，提取相关信息后保存至1_hpo_database.json
        """
        hpo_list = []
        for term in tqdm(self.HPO.terms(), desc='Generating HPO database'):
            term_id = term.id
            if not term_id.startswith("HP:"):
                continue

            # Extract term info
            name = term.name
            definition = str(term.definition)
            synonyms = [s.description for s in term.synonyms]
            alt_ids = list(term.alternate_ids)
            umls_ids = [xref.id for xref in term.xrefs if xref.id.startswith('UMLS')]

            # Root node
            root_id = self.term_root.get(term_id, '')

            if root_id == 'HP:0000118':  # Phenotypic abnormality
                hpo_list.append({
                    'id': term_id,
                    'name': name,
                    'definition': definition,
                    'synonyms': synonyms,
                    'umls_id': umls_ids,
                    'root_id': root_id,
                    'alt_ids': alt_ids
                })

        with open(os.path.join(self.SAVE_PATH, '1_hpo_database.json'), 'w') as f:
            json.dump(hpo_list, f)
    
    
    def generate_hpo_matrix(self):
        """
        根据HPO数据库生成经归一化处理后的向量矩阵并保存
        """
        # HPO database
        with open(os.path.join(self.SAVE_PATH, '1_hpo_database.json'), 'r') as f:
            hpo_db = json.load(f)
        
        # addon hpo
        if self.HP_ADDON_PATH:
            hp_addon = pd.read_csv(self.HP_ADDON_PATH)
        
        records = []
        idx = 0
        for term in tqdm(hpo_db, desc="Generating embeddings"):
            # Process name
            if term['name']:
                records.append((f"{idx}_{term['id']}_name", term['name']))
                idx +=1
            
            # Process synonyms
            for syn in term['synonyms']:
                records.append((f"{idx}_{term['id']}_synonym", syn))
                idx +=1
            
            # Process definition
            for defi in term['definition']:
                records.append((f"{idx}_{term['id']}_definition", defi))
                idx +=1
        
        # addon hpo
        if self.HP_ADDON_PATH:
            for _, row in hp_addon.iterrows():
                records.append((f"{idx}_{row['HP_ID']}_name", row['info']))
                idx += 1
            print("Addon HPO records:", len(hp_addon))
            
        # Batch encode
        texts = [r[1] for r in records]
        embeddings = self.Model.encode(texts, batch_size=256, show_progress_bar=False)
        
        # Save as DataFrame
        df = pd.DataFrame(embeddings, index=[r[0] for r in records])
        # Normalize
        hpo_embeddings_norm = pd.DataFrame(preprocessing.normalize(df, norm='l2'),index=df.index)
        hpo_embeddings_norm.to_pickle(os.path.join(self.SAVE_PATH, '2_hpo_embeddings_matrix.pkl'))


    def bulid(self):
        self.generate_database()
        self.generate_hpo_matrix()
        # Information
        info_dict = {
            "build_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.model_name,
            "hpo_obo_path": self.HP_OBO_PATH
        }
        with open(os.path.join(self.SAVE_PATH, "database_info.json"), "w") as f:
            json.dump(info_dict, f, ensure_ascii=False, indent=4)
            
        self.logger.info(f"Build complete! Matrix path: {self.SAVE_PATH}")
