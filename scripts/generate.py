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
- 記事のURLは必ずニュース一覧にある実在のURLをそのまま使うこと。
"""
    else:
        instruction = """
あなたはユーザーの思考を拡張する優秀なリサーチ編集長です。
以下のニュース一覧とユーザーの関心メモをもとに、「私生活用」の日刊紙面JSONを作成してください。
- deep_dives は必ず2本作成すること：
  1本目: 株・経済（category_key: "stock"）— ニュースの背景から二次的な波及先・逆張り視点まで丁寧に解説
  2本目: AI・テック（category_key: "ai"）— 技術の本質と今後1〜3年の構造変化を丁寧に解説
- quick_radars は必ず4本作成すること：
  ゲーム関連2本（category_key: "game"）＋ 世間の話題・無関係枠2本（category_key: "trend"）。
- すべて自然な日本語で、前提知識がなくてもスッと頭に入る粒度で書くこと。
- 記事のURLは必ずニュース一覧にある実在のURLをそのまま使うこと。
"""

    prompt = f"""
{instruction}

【ユーザーの関心メモ】
{interests}

【取得した最新ニュース一覧】
{raw_news}
"""

    candidate_models = list(dict.fromkeys([
        MODEL_NAME,
        "gemini-3.5-flash-lite",
    ]))

    last_error = None
    for idx, model_name in enumerate(candidate_models):
        max_attempts = 1 if idx == 0 and len(candidate_models) > 1 else 3
        for attempt in range(1, max_attempts + 1):
            try:
                print(f"モデル試行中: {model_name} ({attempt}/{max_attempts})")
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=EditionOutput,
                        temperature=0.5,
                    ),
                )
                print(f"生成成功: {model_name}")
                return EditionOutput.model_validate_json(response.text)
            except Exception as e:
                last_error = e
                msg = str(e)
                if "404" in msg or "NOT_FOUND" in msg:
                    print(f"{model_name} は利用不可のためスキップします")
                    break
                if ("503" in msg or "UNAVAILABLE" in msg or "429" in msg) and attempt < max_attempts:
                    wait = 15 * attempt
                    print(f"{model_name} 混雑のため {wait} 秒後にリトライします...")
                    time.sleep(wait)
                else:
                    print(f"{model_name} での生成に失敗。次の候補へ移ります")
                    break

    raise last_error


def render_html(mode: str, data: EditionOutput, date_str: str) -> str:
    env = Environment(loader=FileSystemLoader(str(BASE_DIR / "templates")))
    template = env.get_template("page.html")
    return template.render(
        mode=mode,
        page_title="MS仕事用" if mode == "work" else "私生活用",
        edition_label="MS WORK EDITION" if mode == "work" else "LIFE & SIGNAL EDITION",
        date_str=date_str,
        data=data,
    )


def notify_discord(webhook_url: str, title: str, page_url: str, data: EditionOutput):
    if not webhook_url:
        print(f"Webhook未設定のため通知スキップ: {title}")
        return
    headlines = "\n".join([f"・**{a.title}**" for a in data.deep_dives])
    radars = "\n".join([f"・{r.title}" for r in data.quick_radars[:2]])
    content = (
        f"## {title}\n"
        f"**【Deep Dive】**\n{headlines}\n\n"
        f"**【Quick Radar ほか】**\n{radars}\n\n"
        f"👉 **紙面を読む:** {page_url}"
    )
    res = requests.post(webhook_url, json={"content": content}, timeout=15)
    res.raise_for_status()
    print(f"Discord通知完了: {title}")


def main():
    api_key = os.environ["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)

    jst = datetime.timezone(datetime.timedelta(hours=9))
    now = datetime.datetime.now(jst)
    date_str = now.strftime("%Y/%m/%d (%a)")
    target = os.environ.get("TARGET_EDITION", "both")  # both / work / life

    interests = (BASE_DIR / "config" / "interests.txt").read_text(encoding="utf-8")
    feeds = yaml.safe_load((BASE_DIR / "config" / "feeds.yaml").read_text(encoding="utf-8"))

    DOCS_DIR.mkdir(exist_ok=True)
    repo_full = os.environ.get("GITHUB_REPOSITORY", "")
    if "/" in repo_full:
        owner, repo = repo_full.split("/", 1)
        base_url = f"https://{owner}.github.io/{repo}"
    else:
        base_url = "https://example.com"

    modes = ["work", "life"] if target == "both" else [target]

    for mode in modes:
        print(f"=== {mode} 版を生成中 ===")
        raw_news = fetch_articles(feeds[mode])
        edition = generate_edition(client, mode, interests, raw_news)
        html = render_html(mode, edition, date_str)
        (DOCS_DIR / f"{mode}.html").write_text(html, encoding="utf-8")

        if mode == "work":
            notify_discord(
                os.environ.get("DISCORD_WEBHOOK_WORK", ""),
                f"💼 MS Work Brief ({date_str})",
                f"{base_url}/work.html",
                edition,
            )
        else:
            notify_discord(
                os.environ.get("DISCORD_WEBHOOK_LIFE", ""),
                f"🌐 Daily Signal ({date_str})",
                f"{base_url}/life.html",
                edition,
            )

    # ルートURLを開いたときに life.html へ自動移動するページを配置
    index_html = '<meta http-equiv="refresh" content="0; url=./life.html">'
    (DOCS_DIR / "index.html").write_text(index_html, encoding="utf-8")
    print("完了しました！")


if __name__ == "__main__":
    main()
