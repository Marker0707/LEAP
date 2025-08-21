from importlib.metadata import version as _version, PackageNotFoundError
import importlib, types
from .core import LEAP

# ---- metadata ---------------------------------------------------------------
__all__ = ["LEAP"]              # 让 `from leap-hpo import *` 只导出 LEAP
__version__: str = _version("leap-hpo")


# ---- lazy import ---------------------------------------------------------------
_lazy_modules = {"builder", "gene_rank"}
def __getattr__(name: str) -> types.ModuleType:
    if name in _lazy_modules:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module          # 缓存到全局，后续访问不再走这里
        return module
    raise AttributeError(name)

# ---- logging ---------------------------------------------------------------
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())