"""Extract local reference notes used by the MetaOpt diagnostics."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REF_DIR = ROOT / "docs" / "references"


class ZhihuArticleParser(HTMLParser):
    """Small stdlib parser for saved Zhihu article HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.capture_depth = 0
        self.skip_depth = 0
        self.tag_stack: list[str] = []
        self.current_tag: str | None = None
        self.current: list[str] = []
        self.blocks: list[tuple[str, str]] = []
        self.title = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k: v or "" for k, v in attrs}
        cls = attr.get("class", "")
        if tag in {"script", "style", "nav", "footer", "aside"}:
            self.skip_depth += 1
            return
        if self.capture_depth == 0 and "Post-RichText" in cls:
            self.capture_depth = 1
        elif self.capture_depth > 0:
            self.capture_depth += 1
        if self.skip_depth or self.capture_depth == 0:
            return
        if tag in {"h1", "h2", "h3", "p", "li", "pre", "code"}:
            self._flush()
            self.current_tag = tag
            self.current = []

    def handle_endtag(self, tag: str) -> None:
        if self.skip_depth:
            if tag in {"script", "style", "nav", "footer", "aside"}:
                self.skip_depth -= 1
            return
        if self.capture_depth > 0:
            if tag in {"h1", "h2", "h3", "p", "li", "pre", "code"}:
                self._flush()
            self.capture_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.skip_depth or self.capture_depth == 0:
            return
        text = data.strip()
        if text:
            self.current.append(text)

    def _flush(self) -> None:
        if not self.current_tag or not self.current:
            self.current_tag = None
            self.current = []
            return
        text = re.sub(r"\s+", " ", " ".join(self.current)).strip()
        if text and not _looks_like_recommendation(text):
            self.blocks.append((self.current_tag, text))
        self.current_tag = None
        self.current = []


def _looks_like_recommendation(text: str) -> bool:
    banned = [
        "推荐阅读",
        "相关推荐",
        "评论",
        "赞同",
        "添加评论",
        "分享",
        "收藏",
        "发布于",
    ]
    return any(token in text for token in banned)


def extract_zhihu() -> Path:
    html_path = REF_DIR / "Learning to learn by gradient descent by gradient descent， Pytorch 实践 - 知乎.html"
    out_path = REF_DIR / "zhihu_50638287_extracted.md"
    raw = html_path.read_text(encoding="utf-8", errors="replace")
    title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, re.I | re.S)
    title = html.unescape(title_match.group(1)).strip() if title_match else html_path.stem
    parser = ZhihuArticleParser()
    parser.feed(raw)
    parser._flush()

    lines = [
        f"# {title}",
        "",
        "> 来源：本地保存的知乎 HTML。该材料只作为通俗解释参考，不作为学术证据。",
        "",
    ]
    if not parser.blocks:
        lines.extend([
            "正文提取失败：未在 HTML 中找到可解析的 Post-RichText 正文块。",
            "",
        ])
    else:
        for tag, text in parser.blocks:
            if tag == "h1":
                lines.append(f"# {text}")
            elif tag == "h2":
                lines.append(f"## {text}")
            elif tag == "h3":
                lines.append(f"### {text}")
            elif tag in {"pre", "code"} and len(text) > 40:
                lines.extend(["```python", text, "```"])
            elif tag == "li":
                lines.append(f"- {text}")
            else:
                lines.append(text)
            lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    path = extract_zhihu()
    print(path)
