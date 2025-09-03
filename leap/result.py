from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
from typing import Literal, Optional, List

from .gene_rank import GeneRankResult
from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class LEAPResult:
    input_content: str
    llm_client: str
    embedding_model: str
    llm_result: list[str]
    rerank: bool
    retain_furthest: bool
    use_weighting: bool
    
    retrieve_result: pd.DataFrame
    weighted_result: Optional[pd.DataFrame]
    rerank_result: Optional[pd.DataFrame]
    final_leap_result: list[str]
    
    created_time: datetime = field(default_factory=datetime.now, init=False)
    gene_rank_result: list[GeneRankResult] = field(default_factory=list)

    def __str__(self):
        return (
            f"LEAPResult(input_content, llm_client, embedding_model, "
            f"llm_result, raw_leap_df, final_leap_result, gene_rank_result, "
            f"created_time: {self.created_time})"
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
