from ..bootstrap import ensure_repo_root
from functools import lru_cache
from lib import ObjectNamespace

ensure_repo_root()


@lru_cache
def get_status():
    return ObjectNamespace(status="OK", rvc=ObjectNamespace())


STATUS = get_status()
