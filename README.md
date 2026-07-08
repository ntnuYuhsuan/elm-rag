# Edge-LLM RAG

本地端 RAG 系統，以生醫文獻回答生理訊號（PPG、ECG、HRV、LTHR、IMU）相關問題。適用 Apple Silicon，完全離線運行。

## 前置需求

- Python 3.11、Conda
- [Ollama](https://ollama.com/)（本地 LLM 推論）
- PDF 論文放於 `data/` 目錄



## QuickStart

```bash
# 1. 環境
conda create -n elm-rag python=3.11 -y
conda activate elm-rag
pip install -r requirements.txt

# 2. 下載 LLM（預設 qwen3:8b）
ollama pull qwen3:8b
```  

```bash
# 建立跨語種向量庫 (預設儲存於 chroma_db_m3/)
python src/m3-direct/build_index.py

# 執行 RAG 問答
python src/m3-direct/app.py

# 執行基準批次評估
python src/m3-direct/evaluate.py
```

## Settings

實驗參數（模型、溫度、相似度門檻、Prompt）見 `config.yaml`。

## 參考文獻

將對應 PDF 放入 `data/` 目錄後執行 `build_index.py` 建立索引。

### PPG 運動偽影

| 論文 | 作者 | 年份 | 連結 |
|------|------|------|------|
| TROIKA: A General Framework for Heart Rate Monitoring | Zhang et al. | 2015 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/25252274/) |
| SPARE: A Spectral Peak Recovery Algorithm for PPG Signals | Masinelli et al. | 2021 | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8070644/) |
| Motion artifact removal from PPG signals (tICA + adaptive filter) | Peng et al. | 2014 | [Semantic Scholar](https://www.semanticscholar.org/paper/Motion-artifact-removal-from-photoplethysmographic-Peng-Zhang/12ec78e724ce5f6abd70656e21ac292778c4fbf0) |

### IMU 跌倒偵測

| 論文 | 作者 | 年份 | 連結 |
|------|------|------|------|
| Human Falling Detection Algorithm Based on Multisensor Data Fusion with SVM | Pan et al. | 2020 | [DOI](https://doi.org/10.1155/2020/8826088) |
| A comparison of accuracy of fall detection algorithms (threshold-based vs. machine learning) | Aziz et al. | 2017 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/27106749/) |

### LTHR / 運動強度閾值

| 論文 | 作者 | 年份 | 連結 |
|------|------|------|------|
| Validity of DFA of HRV to Determine Intensity Thresholds | Mateo-March et al. | 2022 | [Wiley](https://onlinelibrary.wiley.com/doi/10.1080/17461391.2022.2047228) |
| Heart Rate Variability-Derived Thresholds for Exercise Intensity Prescription (Systematic Review) | Kaufmann et al. | 2023 | [Springer](https://link.springer.com/article/10.1186/s40798-023-00607-2) |
