import re

PAPERS_META = {
    "Mobile Information Systems - 2020 - Pan - Human Falling Detection Algorithm Based on Multisensor Data Fusion with SVM.pdf": {
        "title": "Human Falling Detection Algorithm Based on Multisensor Data Fusion with SVM",
        "authors": "Pan et al.",
        "year": 2020,
        "topic": "IMU_fall_detection",
        "doi": "10.1155/2020/8826088"
    },
    "TROIKA_A_General_Framework_for_Heart_Rate_Monitoring_Using_Wrist-Type_Photoplethysmographic_Signals_During_Intensive_Physical_Exercise.pdf": {
        "title": "TROIKA: A General Framework for Heart Rate Monitoring",
        "authors": "Zhang et al.",
        "year": 2015,
        "topic": "PPG_motion_artifact",
        "doi": "10.1109/TBME.2014.2359372"
    },
    "Validity of detrended fluctuation analysis of heart rate variability to determine intensity thresholds in elite cyclists.pdf": {
        "title": "Validity of DFA of HRV to Determine Intensity Thresholds",
        "authors": "Mateo-March et al.",
        "year": 2022,
        "topic": "LTHR_estimation",
        "doi": "10.1080/17461391.2022.2047228"
    },
    "SPARE A Spectral Peak Recovery Algorithm for PPG Signals Pulsewave Reconstruction in Multimodal Wearable Devices.pdf": {
        "title": "SPARE: A Spectral Peak Recovery Algorithm for PPG Signals Corrupted by Motion Artifacts",
        "authors": "Masinelli et al.",
        "year": 2021,
        "topic": "PPG_motion_artifact",
        "doi": "10.3390/s21082725"
    },
    "Motion artifact removal from photoplethysmographic signals by combining temporally constrained independent component analysis and adaptive filter.pdf": {
        "title": "Motion artifact removal from photoplethysmographic signals by combining temporally constrained independent component analysis and adaptive filter",
        "authors": "Peng et al.",
        "year": 2014,
        "topic": "PPG_motion_artifact",
        "doi": "10.1186/1475-925X-13-50"
    },
    "A comparison of accuracy of fall detection algorithms (threshold-based vs. machine learning) using waist-mounted tri-axial accelerometer signals from a comprehensive set of falls and non-fall trials..pdf": {
        "title": "A comparison of accuracy of fall detection algorithms (threshold-based vs. machine learning) using waist-mounted tri-axial accelerometer signals",
        "authors": "Aziz et al.",
        "year": 2017,
        "topic": "IMU_fall_detection",
        "doi": "10.1007/s11517-016-1504-y"
    },
    "Heart rate variability-derived thresholds for exercise intensity prescription in endurance sports a systematic review of interrelations and agreement with different ventilatory and blood lactate thresholds..pdf": {
        "title": "Heart Rate Variability-Derived Thresholds for Exercise Intensity Prescription: A Systematic Review",
        "authors": "Kaufmann et al.",
        "year": 2023,
        "topic": "LTHR_estimation",
        "doi": "10.1186/s40798-023-00607-2"
    }
}

# ==========================================
# Markdown 解析與高價值區塊過濾邏輯
# ==========================================

def _categorize_header(header: str) -> str:
    """將論文中各種不同的 Markdown 標題正規化為核心 Section 分類"""
    header_lower = header.lower()
    
    if "abstract" in header_lower:
        return "abstract"
    # 方法論：涵蓋 method, system, architecture, algorithm, design, implementation
    elif any(kw in header_lower for kw in ["method", "system", "architecture", "algorithm", "design", "implementation", "proposed"]):
        return "methodology"
    # 結果與實驗：涵蓋 result, evaluation, experiment, discussion, performance
    elif any(kw in header_lower for kw in ["result", "evaluation", "experiment", "discussion", "performance", "validation"]):
        return "results"
    # 結論
    elif any(kw in header_lower for kw in ["conclusion", "future"]):
        return "conclusion"
    # 參考文獻與致謝 (需過濾)
    elif any(kw in header_lower for kw in ["reference", "bibliography", "acknowledgment"]):
        return "references"
    # 前言與背景 (需過濾)
    elif any(kw in header_lower for kw in ["introduction", "background", "related work"]):
        return "introduction"
    else:
        return "others"

def parse_markdown_sections(md_text: str) -> list:
    """
    解析 Markdown，並「只保留」高價值的 sections。
    直接排除 references, introduction, 以及未明確定義的 others。
    """
    lines = md_text.split('\n')
    sections = []
    
    # 論文最開頭通常是 Abstract，即使沒有標題
    current_section_type = "abstract" 
    current_content = []
    
    # 定義我們允許進入向量庫的高價值區塊 (白名單)
    HIGH_VALUE_SECTIONS = {"abstract", "methodology", "results", "conclusion"}

    for line in lines:
        # 偵測 Markdown 標題 (例如 ## 3. Proposed Method)
        header_match = re.match(r'^(#{1,6})\s+(.*)', line)
        
        if header_match:
            # 結算並儲存前一個區塊 (如果它屬於高價值區塊)
            if current_content and current_section_type in HIGH_VALUE_SECTIONS:
                text = "\n".join(current_content).strip()
                # 濾掉過短的無效區塊 (例如只有一行標題沒有內容)
                if len(text) > 100:  
                    sections.append({
                        "section": current_section_type,
                        "content": text
                    })
            
            # 更新為新區塊的狀態
            header_text = header_match.group(2)
            current_section_type = _categorize_header(header_text)
            # 保留標題本身，有助於 LLM 掌握上下文語意
            current_content = [line] 
        else:
            current_content.append(line)

        # metadata 的過濾機制 (假設 metadata 會用特定的 YAML frontmatter 或特定標記，在此可忽略，因為 Regex 只抓內文)

    # 處理文件最後一筆區塊
    if current_content and current_section_type in HIGH_VALUE_SECTIONS:
        text = "\n".join(current_content).strip()
        if len(text) > 100:
            sections.append({
                "section": current_section_type,
                "content": text
            })

    return sections