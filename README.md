# Scaffold Layout MVP

住宅平面図画像から寸法を推定し、標準ルールで足場割付を自動提案して帳票出力するMVPです。

## MVP Scope

- 入力: 寸法入り平面図画像 (PNG/JPG)
- 処理: 前処理 + OCR + 外周抽出 + 標準スパン割付
- 運用: 半自動 (自動提案を人が最終調整)
- 出力: スパン一覧・部材集計の CSV / Excel

## Standard Rules (Initial)

- 基本スパン: `1800mm` 優先
- 最小端部スパン: `900mm`
- 最大スパン: `1800mm`
- 隅角は独立区間として端部調整
- 開口補正は MVP では「除外長」を手入力で適用

## Quick Start

```bash
cd /Users/shomasaito/scaffold-layout-mvp
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
streamlit run src/scaffold_mvp/app.py
```

## Main Modules

- `src/scaffold_mvp/image_pipeline.py`: 画像前処理、寸法 OCR、外周候補抽出
- `src/scaffold_mvp/layout_engine.py`: 標準ルールのスパン割付ロジック
- `src/scaffold_mvp/reporting.py`: CSV / Excel 帳票生成
- `src/scaffold_mvp/app.py`: 半自動 UI

## Streamlit Community Cloud Deploy

1. このディレクトリを GitHub に push
2. [Streamlit Community Cloud](https://share.streamlit.io/) で `New app`
3. 設定:
   - Repository: 対象リポジトリ
   - Branch: `main` (または運用ブランチ)
   - Main file path: `src/scaffold_mvp/app.py`
4. `Deploy` を実行

このリポジトリには Cloud 用に以下を同梱しています。

- `requirements.txt`: Python依存
- `packages.txt`: OS依存 (`tesseract-ocr`)
- `runtime.txt`: Pythonバージョン固定
