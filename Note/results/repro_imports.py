"""复现用 import 兼容（不向上游仓库根目录增加命名 shim 文件）。"""
from __future__ import annotations

import sys


def patch_quick_nav_imports(source: str) -> str:
    """将 quick/real_robot 入口里对私有 localization 模块的 import 指到 `localization`。"""
    return source.replace(
        "from wangbo_localization import",
        "from localization import",
    )
