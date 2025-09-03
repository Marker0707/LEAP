import requests
from io import StringIO
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from .logging_utils import get_logger


# Logging
logger = get_logger(__name__)

# TODO 后期可以加个handler根据输入的软件种类自动选择对象，删掉core中的for loop


@dataclass
class GeneRankResult:
    rank_tool: str
    hpo_list: List[str]
    weight_list: List[float]
    response_status: int
    response_df: pd.DataFrame

    def __str__(self):
        return (
            f"GeneRankResult(hpo_list={self.hpo_list}, "
            f"rank_tool={self.rank_tool},"
            f"weight_list={self.weight_list}, "
            f"response_status={self.response_status}, "
            f"response_df_shape={self.response_df.shape})"
        )

    def to_genelist(self):
        return self.response_df["gene_symbol"].tolist()


class GeneRank(ABC):
    @abstractmethod
    def rank_gene(self, hpo_list: List[str], weight_list: Optional[List[float]] = None) -> GeneRankResult:
        pass


class PhenoAptRank(GeneRank):
    def rank_gene(self, hpo_list: List[str], weight_list: Optional[List[float]] = None) -> GeneRankResult:
        # 参数校验
        if not isinstance(hpo_list, list) or not all(isinstance(h, str) for h in hpo_list):
            raise ValueError("hpo_list must be a list of strings")
        if weight_list is not None:
            if not isinstance(weight_list, list) or len(weight_list) != len(hpo_list):
                raise ValueError("weight_list must be a list of the same length as hpo_list")
            if not all(isinstance(w, (int, float)) for w in weight_list):
                raise ValueError("weight_list must contain numbers")

        hpo = ','.join(hpo_list)
        if weight_list is None:
            weight = ','.join(['1'] * len(hpo_list))
        else:
            weight = ','.join([str(i) for i in weight_list])
        url = f'https://phenoapt.imperialgene.com/phenoapt/api/v3/rank-gene-csv?p={hpo}&weights={weight}'
        try:
            response = requests.get(url, timeout=10)
            logger.info(f"Request phenoapt.imperialgene.com : Status {response.status_code}")
            response.raise_for_status()
            result = StringIO(response.text)
            return GeneRankResult(
                rank_tool='PhenoApt',
                hpo_list= hpo_list,
                weight_list= [float(wt) for wt in weight.split(",")],
                response_status= response.status_code,
                response_df= pd.read_csv(result)
            )
            # return pd.read_csv(result)
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
