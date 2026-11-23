import datetime
import os
import sys
import time
from pathlib import Path
from typing import List, Literal

import feedparser
import requests
import yaml
from google import genai
from google.genai import types
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")


class GlossaryTerm(BaseModel):
    term: str = Field(description="記事中に出てくる専門用語や略語")
    meaning: str = Field(description="前提知識がなくてもわかる1〜2行の解説")


class DeepDiveArticle(BaseModel):
    category_key: Literal["ms", "stock", "ai"]
    category_label: str = Field(description="例: Microsoft 基盤・セキュリティ / Microsoft AI・クラウド / 株・経済 / AI・テック")
    tags: List[str] = Field(description="短いキーワードタグ2〜3個")
    title: str = Field(description="要点が伝わる日本語見出し")
    background: str = Field(description="これまでのあらすじ・前提知識（3〜4文で丁寧に）")
    structure: str = Field(description="何が起きたか・裏側の構造解説（4〜5文でわかりやすく）")
    causal_flow: List[str] = Field(description="因果・波及の3ステップ（必ず3要素）")
    glossary: List[GlossaryTerm] = Field(description="用語ミニ辞典（2〜3語）")
    contrarian: str = Field(description="逆張り視点・リスク・実務上の注意点（2〜3文）")
    source_url: str = Field(description="参照した元記事のURL")


class QuickRadarItem(BaseModel):
    category_key: Literal["ms", "game", "trend"]
    category_label: str = Field(description="例: ゲーム / 世間の話題 / MS小ネタ")
    title: str = Field(description="日本語見出し")
    summary: str = Field(description="前提知識ゼロの人向けの3行解説")
    why_now: str = Field(description="なぜ今これが話題・重要なのかを1文で")
    source_url: str = Field(description="元記事のURL")


class CrossColumn(BaseModel):
    title: str = Field(description="複数トピックを横断するコラムのタイトル")
    body: str = Field(description="今日の複数記事を繋げて得られる気づき（4〜5文）")


class EditionOutput(BaseModel):
    deep_dives: List[DeepDiveArticle]
    quick_radars: List[QuickRadarItem]
    cross_column: CrossColumn


def fetch_articles(feed_groups: dict, max_per_feed: int = 8) -> str:
    lines = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; DailySignalBot/1.0)"}
    for group_name, urls in feed_groups.items():
        lines.append(f"\n### ソースカテゴリ: {group_name}")
        for url in urls:
            try:
                print(f"RSS取得中: {url}")
                res = requests.get(url, headers=headers, timeout=15)
                res.raise_for_status()
                parsed = feedparser.parse(res.content)
                for entry in parsed.entries[:max_per_feed]:
                    title = entry.get("title", "").strip()
                    link = entry.get("link", "").strip()
                    summary = entry.get("summary", "").strip()[:200].replace("\n", " ")
                    if title and link:
                        lines.append(f"- タイトル: {title}\n  URL: {link}\n  概要: {summary}")
            except Exception as e:
                print(f"RSS取得スキップ ({url}): {e}")
    return "\n".join(lines)


def generate_edition(client: genai.Client, mode: str, interests: str, raw_news: str) -> EditionOutput:
    if mode == "work":
        instruction = """
あなたはMicrosoft 365 / Azure / エンドポイント管理に精通した優秀なITアナリストです。
以下のニュース一覧とユーザーの関心メモをもとに、「MS仕事用」の日刊紙面JSONを作成してください。
- deep_dives は必ず2本作成すること：
  1本目: Microsoft基盤・セキュリティ・管理系（Entra ID / Intune / Exchange / SharePoint / Defender 等）
  2本目: Microsoft AI・クラウド・モダンワーク系（Copilot / Azure / 新機能・提案ネタ等）
  ※どちらも category_key は "ms" とすること。
- quick_radars は、その他で押さえておくと便利なIT・MS関連の短いトピックを2本（category_key は "ms"）。
- 英語の公式記事であっても、すべて自然な日本語で噛み砕いて解説すること。
- 記事のURLは必ずニュース一覧にある実在の
