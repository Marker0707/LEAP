import os
from platformdirs import user_cache_dir
import json

def get_default_save_path() -> str:
    """
    利用platformdirs自动选择默认缓存/保存目录, 如果目录不存在则新建一个
    """
    save_path = user_cache_dir("leap-hpo")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)  # 目录不存在时创建目录
    return save_path


def check_hpo_matrix(path, model, hpo_obo_path):
    matrix_path = os.path.join(path, '2_hpo_embeddings_matrix.pkl')
    info_path = os.path.join(path, 'database_info.json')
    if os.path.exists(matrix_path) and os.path.exists(info_path):
        with open(info_path, "r") as f:
            info_dict = json.load(f)
        if (info_dict["model"] == model) and (info_dict["hpo_obo_path"] == hpo_obo_path):
            return True
        else:
            return False
    else:
        return False