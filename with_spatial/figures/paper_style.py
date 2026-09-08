from __future__ import annotations

from matplotlib import font_manager


def _first_available(names: list[str], fallback: str) -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in names:
        if name in available:
            return name
    return fallback


CHINESE_FONT = _first_available(
    ["SimSun", "Songti SC", "Noto Serif CJK SC", "AR PL SungtiL GB"],
    "DejaVu Sans",
)
WESTERN_FONT = _first_available(
    ["Times New Roman", "Tinos", "Nimbus Roman"],
    "DejaVu Serif",
)

CN = font_manager.FontProperties(family=CHINESE_FONT)
EN = font_manager.FontProperties(family=WESTERN_FONT)


def apply_tick_fonts(ax, size: float = 8.0) -> None:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(EN)
        label.set_fontsize(size)
