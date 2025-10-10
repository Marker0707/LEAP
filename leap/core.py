# ==== Imports ====
import os
from sentence_transformers import CrossEncoder, SentenceTransformer, util
import pandas as pd
from typing import Literal, Optional, List
import torch

from .builder import ST_Builder
from .llm_client import LLMClient
from .utils import get_default_save_path, check_hpo_matrix, validate_save_path
from .result import LEAPResult
from .logging_utils import get_logger
from . import node_post
from .node_post import retain_furthest_nodes, find_same_branch_groups, merge_child_nodes

# Logging
logger = get_logger(__name__)


class LEAP:
    def __init__(
        self,
        model: str,
        llm_client: LLMClient,
        hp_obo_path: str,
        save_path: str = "",
        hp_addon_path: str = "",
    ) -> None:

        # save path
        if validate_save_path(save_path):
            self.SAVE_PATH = save_path
            logger.info(f"Using user-defined save path: {self.SAVE_PATH}")
        else:
            self.SAVE_PATH = get_default_save_path()
            if validate_save_path(self.SAVE_PATH):
                logger.warning(f"Using default save path: {self.SAVE_PATH}")
            else:
                logger.error(f"Failed to create default save path: {self.SAVE_PATH}")
                raise

        # hpo obo path
        self.HP_OBO_PATH = hp_obo_path

        # 配置 node_post 模块
        node_post.set_config(self.SAVE_PATH, self.HP_OBO_PATH)

        # load LLM client
        self.llm_client = llm_client

        # Initiation database and emb matrix
        if check_hpo_matrix(path=self.SAVE_PATH, model=model, hpo_obo_path=self.HP_OBO_PATH):  # mode : str
            logger.info(f'Check HPO matrix: Pass, which is saved at {self.SAVE_PATH}')
        else:
            logger.warning(f'Check HPO matrix: Fail. Building HPO matrix, which will be saved to {self.SAVE_PATH}')
            ST_Builder(save_path=self.SAVE_PATH, hp_obo_path=self.HP_OBO_PATH, model=model, hp_addon_path=hp_addon_path).build()
            logger.info(f"HPO matrix: Build successfully, which is saved to {self.SAVE_PATH}")

        logger.debug(f"Loading HPO matrix from: {os.path.join(self.SAVE_PATH, 'hpo_embeddings_matrix.pkl')}")
        self.HPO_MATRIX = pd.read_pickle(os.path.join(self.SAVE_PATH, 'hpo_embeddings_matrix.pkl'))

        # load model
        self.model_name = model
        self.Model = SentenceTransformer(model)

    
    def _retrieve(
        self, 
        query_list: List[str],
        top_k: int
        ):
        query_embedding = self.Model.encode(query_list, convert_to_tensor=True, show_progress_bar=False)
        hits = util.semantic_search(query_embedding, self.HPO_MATRIX["embeddings"], top_k=top_k)
        return hits


    def _rerank(
        self,
        query: str,
        hit: List[dict],
        answer_text: List[str],
        rerank_model: CrossEncoder,
    ):
        """
        single query
        """
        ranked_result = rerank_model.rank(
            query, 
            [answer_text[i["corpus_id"]] for i in hit], 
            return_documents=True, 
            top_k=5 , 
            show_progress_bar=False
        )
        return ranked_result


    def _apply_weighting(self, result_df: pd.DataFrame, cutoff: float, mode: Literal["retrieve", "rerank"]) -> pd.DataFrame:
        """
        Apply weighting to the result dataframe using same branch groups and furthest nodes.
        
        Args:
            result_df: DataFrame containing the retrieval results
            cutoff: Score cutoff threshold
            mode: The mode of operation, either "retrieve" or "rerank"

        Returns:
            Modified DataFrame with updated hpo_top_1 values based on weighting
        """
        if not result_df.empty:
            for idx, row in result_df.iterrows():
                if (row[f"{mode}_hpo_top_100"][3]["score"] < cutoff):  # Not Matched出现大于等于7次，那么直接跳过
                    continue
                same_branch = find_same_branch_groups([node["label"] for node in row[f"{mode}_hpo_top_100"][0:10] if node["score"] >= cutoff]) # 取top100的0到10来计算权重，需要排除Not Match不然会报错Node在图G中
                # 每个分支所含节点的个数--每个元组的长度
                nodes_per_branch = [len(tup) for tup in same_branch]
                # 含节点数最多的分支
                most_nodes_branch = same_branch[nodes_per_branch.index(max(nodes_per_branch))] # 如果同时有多个最大值，index只取第一个出现的最大值
                # 必须要调用retain_furthest_nodes，求最远的节点作为top1。正常来说只会返回一个值
                assert len(retain_furthest_nodes(list(most_nodes_branch))) == 1
                result_df.at[idx, f"{mode}_hpo_top_1"] = retain_furthest_nodes(list(most_nodes_branch))[0]
        else:
            logger.warning("Result dataframe is empty after applying cutoff. No entries to process for weighting.")
        
        return result_df


    def _phrase2hpo_no_rerank(
        self,
        gpt_out: List[str],
        content: str,
        furthest: bool,
        use_weighting: bool,
        retrieve_cutoff: float,
        top_k: int,
        ):
        # predefine a result object
        result_obj = LEAPResult(
                input_content=content,
                llm_client=getattr(self.llm_client, 'llm_name', 'Unknown'),
                embedding_model=self.model_name,
                llm_result=gpt_out,
                retain_furthest=furthest,
                use_weighting=use_weighting,
                rerank=False,
                retrieve_result=pd.DataFrame(),
                weighted_result=None,
                rerank_result=None,
                final_leap_result=[],
            )
        
        # Retrive Step
        retrieve_hits = self._retrieve(query_list=gpt_out, top_k=top_k)
        answer_labels = self.HPO_MATRIX["hpo_ids"]

        result_list = []
        for query, hit in zip(gpt_out, retrieve_hits):
            hit_labels = [answer_labels[i["corpus_id"]] for i in hit]
            hit_scores = [i["score"] for i in hit]
            result_list.append({
                "query": query,
                "retrieve_hpo_top_1": hit_labels[0].split("_")[1],
                "retrieve_score_top_1": hit_scores[0],
                "retrieve_hpo_top_100": [{"label": label.split("_")[1], "score": score} for label, score in zip(hit_labels[:100], hit_scores[:100])]
            })

        # Cutoff filter
        result_df = pd.DataFrame(result_list, index=None)
        result_obj.retrieve_result = result_df.copy()  # 保存未过滤的检索结果
        result_df = result_df[result_df["retrieve_score_top_1"] >= retrieve_cutoff].reset_index(drop=True)
        
        # 如果使用top10累加权重
        if use_weighting:
            result_df = self._apply_weighting(result_df, retrieve_cutoff, mode="retrieve").drop(columns="retrieve_score_top_1")  # 由于top1被weight编辑，所以score对不上，就没意义
            result_obj.weighted_result = result_df.copy()  # 保存加权后的结果
        
        # 在进行下一步之前，去除Not Matched和重复的节点
        result_df = result_df.drop_duplicates(subset="retrieve_hpo_top_1")
        
        # 只保留最末端/最细节的节点
        if furthest:
            result_obj.final_leap_result = merge_child_nodes(list(set(retain_furthest_nodes(result_df["retrieve_hpo_top_1"].to_list()))))
        else:
            result_obj.final_leap_result = merge_child_nodes(result_df["retrieve_hpo_top_1"].to_list())

        return result_obj


    def _phrase2hpo_rerank(
        self,
        rerank_model_name: str, 
        gpt_out: List[str],
        content: str,
        furthest: bool,
        use_weighting: bool,
        retrieve_cutoff: float,
        rerank_cutoff: float,
        top_k: int
        ):
        
        logger.warning("The retrieve cutoff is invalid when setting rerank=True")
        
        # Initialize CrossEncoder for re-ranking
        cross_encoder_model = CrossEncoder(rerank_model_name, activation_fn=torch.nn.Sigmoid())
        
        # Predefine result object
        result_obj = LEAPResult(
            input_content=content,
            llm_client=getattr(self.llm_client, 'llm_name', 'Unknown'),
            embedding_model=self.model_name,
            llm_result=gpt_out,
            retain_furthest=furthest,
            use_weighting=False,
            rerank=True,
            retrieve_result=pd.DataFrame(),
            weighted_result=None,
            rerank_result=None,
            final_leap_result=[],
            )
        
        # Retrive & Rerank Step
        retrieve_hits = self._retrieve(query_list=gpt_out, top_k=top_k)
        answer_labels = self.HPO_MATRIX["hpo_ids"]
        answer_text = self.HPO_MATRIX["hpo_text"]

        result_list = []
        rerank_result_list = []
        for query, hit in zip(gpt_out, retrieve_hits):
            # Retrieve
            hit_labels = [answer_labels[i["corpus_id"]] for i in hit]
            hit_scores = [i["score"] for i in hit]
            hit_text = [answer_text[i["corpus_id"]] for i in hit]
            result_list.append({
                "query": query,
                "retrieve_hpo_top_1": hit_labels[0].split("_")[1],
                "retrieve_score_top_1": hit_scores[0],
                "retrieve_text_top_1": hit_text[0],
                "retrieve_hpo_top_100": [{"label": label.split("_")[1], "score": score} for label, score in zip(hit_labels[:100], hit_scores[:100])]
            })
            # Rerank
            rerank_result = self._rerank(query=query, hit=hit, answer_text=answer_text, rerank_model=cross_encoder_model)
            rerank_labels = [hit_labels[i["corpus_id"]] for i in rerank_result]
            rerank_texts = [hit_text[i["corpus_id"]] for i in rerank_result]
            rerank_scores = [i["score"] for i in rerank_result]
            rerank_result_list.append({
                "query": query,
                "rerank_hpo_top_1": rerank_labels[0].split("_")[1],
                "rerank_score_top_1": rerank_scores[0],
                "rerank_text_top_1": rerank_texts[0],
                "rerank_hpo_top_5": [{"label": label.split("_")[1], "score": score} for label, score in zip(rerank_labels[:100], rerank_scores[:100])]
            })

        # save retrieve result
        result_df = pd.DataFrame(result_list, index=None)
        result_obj.retrieve_result = result_df.copy()  # 保存未过滤的retrieve检索结果

        # rerank cutoff filter
        rerank_result_df = pd.DataFrame(rerank_result_list, index=None)
        rerank_result_df = rerank_result_df[rerank_result_df["rerank_score_top_1"] >= rerank_cutoff]
        result_obj.rerank_result = rerank_result_df.copy()  # 保存过滤后的rerank检索结果
        
        # 如果使用top10累加权重
        if use_weighting:
            rerank_result_df = self._apply_weighting(rerank_result_df, rerank_cutoff, mode="rerank").drop(columns="rerank_score_top_1")
            result_obj.weighted_result = rerank_result_df.copy()  # 保存加权后的结果

        # 在进行下一步之前，去除重复的节点
        rerank_result_df = rerank_result_df.drop_duplicates(subset="rerank_hpo_top_1")

        # 只保留最末端/最细节的节点
        if furthest:
            result_obj.final_leap_result = merge_child_nodes(list(set(retain_furthest_nodes(rerank_result_df["rerank_hpo_top_1"].to_list()))))
        else:
            result_obj.final_leap_result = merge_child_nodes(rerank_result_df["rerank_hpo_top_1"].to_list())

        return result_obj

    
    def convert_ehr(
        self,
        content: str,
        rerank_model_name: Optional[str],
        rerank: bool = False,
        furthest: bool = True,
        use_weighting: bool = False,
        top_k: int = 500,
        retrieve_cutoff: float = 0.7,
        rerank_cutoff: float = 0.5
    ) -> Optional[LEAPResult]:
        
        # LLM process
        gpt_out = self.llm_client.extract_phenotypes(content)  # 异常捕获均由模块自己处理，在这里不需要重复捕获异常
        
        try:
            if rerank:
                assert rerank_model_name is not None, "When setting rerank=True, rerank_model_name must be provided."
                return self._phrase2hpo_rerank(
                    gpt_out = gpt_out,
                    content = content,
                    furthest = furthest,
                    use_weighting = use_weighting,
                    top_k = top_k,
                    retrieve_cutoff = retrieve_cutoff,
                    rerank_cutoff = rerank_cutoff,
                    rerank_model_name = rerank_model_name
                )
            else:
                return self._phrase2hpo_no_rerank(
                    gpt_out = gpt_out,
                    content = content,
                    furthest = furthest,
                    use_weighting = use_weighting,
                    top_k = top_k,
                    retrieve_cutoff = retrieve_cutoff
                )
        except:
            logger.error("Error occurred in retrieval and/or reranking")
            raise EOFError("Error occurred in retrieval and/or reranking")
