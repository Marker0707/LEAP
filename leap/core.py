# ==== Imports ====
import os
import logging
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn import preprocessing
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional, List, Any, Tuple
import numpy as np

from .node_post import retain_furthest_nodes, find_same_branch_groups, merge_child_nodes
from .builder import ST_Builder
from .llm_client import LLMClient
from .utils import get_default_save_path, check_hpo_matrix
from .gene_rank import GeneRankResult
from .config import load_config

# Logging
logger = logging.getLogger(__name__)


@dataclass
class LEAPResult:
    input_content: str
    llm_client: str
    embedding_model: str
    llm_result: list[str]
    retain_furthest: bool
    use_weighting: bool
    raw_leap_df: pd.DataFrame
    final_leap_result: list[str]
    created_time: datetime = field(default_factory=datetime.now, init=False)
    gene_rank_result: list[GeneRankResult] = field(default_factory=list)

    def __str__(self):
        return (
            f"LEAPResult(input_content, llm_client, embedding_model, "
            f"llm_result, raw_leap_df, final_leap_result, gene_rank_result, "
            f"creat_time: {self.created_time})"
        )

    def rank_gene(
        self,
            weight_list: Optional[List[float]] = None,
            tool: Literal["PhenoApt"] = "PhenoApt"
    ) -> None:
        # 先检查final_leap_result有没有hpo id
        if not (isinstance(self.final_leap_result, list) & len(self.final_leap_result) > 0):
            logger.error("Aborted: final_leap_result is empty or not a list")
            return None

        # 根据选择的gene rank软件懒加载
        if tool == "PhenoApt":
            from .gene_rank import PhenoAptRank
            self.gene_rank_result.append(
                # 返回的结果append到self.gene_rank_result里
                PhenoAptRank().rank_gene(hpo_list=self.final_leap_result, weight_list=weight_list)
            )
        return None


class LEAP:
    def __init__(
        self,
        model: str,
        llm_client: LLMClient,
        cut_off: float = 0.72,
        config_path: Optional[str] = None,
        hp_addon_path: Optional[str] = "",
    ) -> None:

        # >>>>>>>>>>>>>>>>> config >>>>>>>>>>>>>>>>>
        config = load_config(config_path)
        # save path
        self.SAVE_PATH = config["SAVE_PATH"] or get_default_save_path()
        if not os.path.exists(self.SAVE_PATH):
            os.makedirs(self.SAVE_PATH)
        # hpo obo path
        self.HP_OBO_PATH = config["HP_OBO_PATH"]

        # cutoff
        self.CUTOFF = cut_off
        # <<<<<<<<<<<<<<<<< config <<<<<<<<<<<<<<<<<

        # load model
        self.model_name = model
        self.Model = SentenceTransformer(model)

        # load LLM client
        self.llm_client = llm_client

        # Initiation database and emb matrix
        if check_hpo_matrix(path=self.SAVE_PATH, model=model, hpo_obo_path=self.HP_OBO_PATH):  # mode : str
            logger.info(f'Check HPO matrix: Pass, which is saved at {self.SAVE_PATH}')
        else:
            logger.warning(f'Check HPO matrix: Fail. Building HPO matrix, which will be saved at {self.SAVE_PATH}')
            ST_Builder(save_path=self.SAVE_PATH, hp_obo_path=self.HP_OBO_PATH, model=model, hp_addon_path=hp_addon_path).bulid()
            logger.info(f"HPO matrix: Build successfully, which is saved at {self.SAVE_PATH}")

        self.HPO_MATRIX = pd.read_pickle(os.path.join(self.SAVE_PATH, '2_hpo_embeddings_matrix.pkl'))


    def _calculate_pheno_emb(
        self,
        description_list: List[str]
    ) -> pd.DataFrame:
        description_embeddings_id = [f'{idx}_{i}' for idx, i in enumerate(description_list)]
        description_embeddings = self.Model.encode(description_list, batch_size=256, show_progress_bar=True).tolist()
        description_df = pd.DataFrame(description_embeddings, index = description_embeddings_id)
        return description_df


    # def _calculate_similarity(
    #     self,
    #     phenotype_emb_df: pd.DataFrame,
    #     hpo_emb_df: pd.DataFrame
    # ) -> pd.DataFrame:
    #     description_embeddings_norm = pd.DataFrame(preprocessing.normalize(phenotype_emb_df, norm='l2'),index=phenotype_emb_df.index)
    #     hpo_similarity_matrix = description_embeddings_norm.dot(hpo_emb_df.T)  # hpo_emb再builder里已经归一化
        
    #     max_hpo = []
    #     for idx in range(0, hpo_similarity_matrix.shape[0]):
    #         single_pheno = hpo_similarity_matrix.iloc[idx, :].sort_values(ascending=False)
    #         single_pheno = single_pheno[single_pheno >= self.CUTOFF].index
    #         single_pheno = [i.split('_')[1] for i in single_pheno]
    #         while len(single_pheno) < 101:
    #             single_pheno = single_pheno + ['Not Matched']
    #         max_hpo.append({'id' : hpo_similarity_matrix.index[idx],
    #                         'hpo_top_1' : single_pheno[0],
    #                         'hpocode_top_100' : single_pheno})
        
    #     max_hpo_df = pd.DataFrame(max_hpo)
    #     max_hpo_df.set_index('id', inplace=True)
    #     return max_hpo_df, hpo_similarity_matrix
    
    
    def _calculate_similarity(
        self,
        phenotype_emb_df: pd.DataFrame,
        hpo_emb_df: pd.DataFrame
    ) -> pd.DataFrame:
        CUTOFF = float(self.CUTOFF)
        K = 101
        NOT_MATCHED = "Not Matched"

        # 1) 归一化 phenotype
        ph = phenotype_emb_df.to_numpy(dtype=np.float32, copy=False)
        ph_norm = np.maximum(np.linalg.norm(ph, axis=1, keepdims=True), 1e-12)
        ph = ph / ph_norm

        # 2) HPO 向量
        hpo = hpo_emb_df.to_numpy(dtype=np.float32, copy=False)

        # 3) 相似度矩阵
        sim = ph @ hpo.T

        # 4) 取每行 Top-K
        n = sim.shape[1]
        if K < n:
            topk_part_idx = np.argpartition(-sim, K-1, axis=1)[:, :K]
            topk_part_val = np.take_along_axis(sim, topk_part_idx, axis=1)
            order_in_k = np.argsort(-topk_part_val, axis=1)
            topk_idx = np.take_along_axis(topk_part_idx, order_in_k, axis=1)
            topk_val = np.take_along_axis(topk_part_val, order_in_k, axis=1)
        else:
            topk_idx = np.argsort(-sim, axis=1)
            topk_val = np.take_along_axis(sim, topk_idx, axis=1)

        # 5) 预解析 HPO 代码
        hpo_index = np.array(hpo_emb_df.index)
        hpo_codes = np.array([str(s).split('_')[1] if '_' in str(s) else str(s)
                            for s in hpo_index], dtype=object)

        top1_codes, top_lists = [], []
        for i in range(sim.shape[0]):
            valid_mask = topk_val[i] >= CUTOFF
            idxs = topk_idx[i][valid_mask]
            codes = hpo_codes[idxs].tolist()

            if len(codes) < K:
                codes.extend([NOT_MATCHED] * (K - len(codes)))

            top1_codes.append(codes[0])
            top_lists.append(codes[:K])

        max_hpo_df = pd.DataFrame(
            {"hpo_top_1": top1_codes, "hpocode_top_100": top_lists},
            index=phenotype_emb_df.index
        )

        hpo_similarity_matrix = pd.DataFrame(sim, index=phenotype_emb_df.index, columns=hpo_emb_df.index)
        return max_hpo_df, hpo_similarity_matrix
    

    def _phrase2hpo(
        self,
        gpt_out: List[str],
        content: str,
        furthest: bool,
        use_weighting: bool
        ):
        
        # 计算相似度并排序
        phenotype_emb_df = self._calculate_pheno_emb(description_list = gpt_out)
        result_df, raw_sim_mtx = self._calculate_similarity(phenotype_emb_df, hpo_emb_df=self.HPO_MATRIX)

        # 如果使用top10累加权重
        if use_weighting:
            for idx, row in result_df.iterrows():
                if (row["hpo_top_1"] == "Not Matched") or (row["hpocode_top_100"][0:10].count("Not Matched")>=7):  # 如果top1是Not Matched或者Not Matched出现大于等于7次，那么直接跳过
                    continue
                same_branch = find_same_branch_groups([node for node in row["hpocode_top_100"][0:10] if node != "Not Matched"]) # 取top100的0到10来计算权重，需要排除Not Match不然会报错Node在图G中
                # 每个分支所含节点的个数--每个元组的长度
                nodes_per_branch = [len(tup) for tup in same_branch]
                # 含节点数最多的分支
                most_nodes_branch = same_branch[nodes_per_branch.index(max(nodes_per_branch))] # 如果同时有多个最大值，index只取第一个出现的最大值
                # 必须要调用retain_furthest_nodes，求最远的节点作为top1。正常来说只会返回一个值
                assert len(retain_furthest_nodes(list(most_nodes_branch))) == 1
                result_df.at[idx, "hpo_top_1"] = retain_furthest_nodes(list(most_nodes_branch))[0]
        
        # 在进行下一步之前，去除Not Matched和重复的节点
        result_df = result_df[result_df["hpo_top_1"] != "Not Matched"]
        result_df = result_df.drop_duplicates(subset="hpo_top_1")
        
        # 只保留最末端/最细节的节点
        if furthest:
            return LEAPResult(
                input_content=content,
                llm_client=self.llm_client.llm_name,
                embedding_model=self.model_name,
                llm_result=gpt_out,
                raw_leap_df=raw_sim_mtx,
                final_leap_result=merge_child_nodes(list(set(retain_furthest_nodes(result_df["hpo_top_1"].to_list())))),
                retain_furthest=furthest,
                use_weighting=use_weighting,
            )
        else:
            return LEAPResult(
                input_content=content,
                llm_client=self.llm_client.llm_name,
                embedding_model=self.model_name,
                llm_result=gpt_out,
                raw_leap_df=raw_sim_mtx,
                final_leap_result=merge_child_nodes(result_df["hpo_top_1"].to_list()),
                retain_furthest=furthest,
                use_weighting=use_weighting,
            )

    def _phrase2hpo_bulk(self, input_list: List[str]):
        """
        This is a func for testing.
        """
        phenotype_emb_df = self._calculate_pheno_emb(description_list = input_list)
        result_df, _ = self._calculate_similarity(phenotype_emb_df, hpo_emb_df=self.HPO_MATRIX)
        return result_df
    
    def convert_ehr(
        self,
        content: str,
        furthest: bool = True,
        use_weighting: bool = False
    ) -> Optional[LEAPResult]:
        try:
            gpt_out = self.llm_client.extract_phenotypes(content)
            return self._phrase2hpo(
                gpt_out = gpt_out,
                content = content,
                furthest = furthest,
                use_weighting = use_weighting
            )
        except Exception as E:
            logger.error(f"Convert_EHR error: {E}")
            return None

    def ehr2gene(
        self,
        content: str,
        furthest: bool = True,
        use_weighting: bool = False,
        tool: Literal["PhenoApt"] = "PhenoApt",
        weight_list: Optional[List[float]] = None,
    ) -> Optional[LEAPResult]:
        obj = self.convert_ehr(
            content = content,
            furthest = furthest,
            use_weighting = use_weighting
        )
        obj.rank_gene(tool=tool, weight_list=weight_list)
        return obj
