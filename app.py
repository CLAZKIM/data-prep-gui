import os
import re
import glob
import json
import math
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import yaml
import pandas as pd
import streamlit as st

# ==============================================================================
# 0. 환경 경로 설정 및 필수 디렉터리 자동 생성
# ==============================================================================
PRESETS_DIR = "./presets"
DATA_DIR = "./data"
DEFAULT_NAS_DIR = "./nas_archive"
NAS_ARCHIVE_DIR = os.environ.get("NAS_ARCHIVE_DIR", DEFAULT_NAS_DIR)

CATEGORY_KEYS = ["feed", "control", "state", "kpi"]


def ensure_directories() -> None:
    """로컬 및 Docker 컨테이너 환경 호환을 위해 필수 상대 경로 폴더를 자동 생성합니다."""
    for folder in [PRESETS_DIR, DATA_DIR, DEFAULT_NAS_DIR, NAS_ARCHIVE_DIR]:
        os.makedirs(folder, exist_ok=True)


def get_sample_ncc_mapping() -> Dict[str, List[Dict[str, Any]]]:
    """화학공학 엔지니어가 즉시 테스트할 수 있는 NCC 및 C2 정류 공정 예시 매핑 데이터를 반환합니다."""
    return {
        "feed": [
            {"tag": "FI-101", "name": "나프타 유량", "unit": "t/h"},
            {"tag": "TI-102", "name": "원료 예열 온도", "unit": "°C"},
        ],
        "control": [
            {"tag": "COT", "name": "열분해로 출구온도", "unit": "°C"},
            {"tag": "DSR", "name": "희석스팀비", "unit": "kg/kg"},
        ],
        "state": [
            {
                "tag": "PDI-205",
                "name": "코킹 차압",
                "unit": "bar",
                "limit_desc": "상한 0.6 bar",
            },
            {
                "tag": "TMT-210",
                "name": "튜브 표면 온도",
                "unit": "°C",
                "limit_desc": "상한 1080 °C",
            },
        ],
        "kpi": [
            {
                "tag": "C2_YIELD",
                "name": "에틸렌 수율",
                "unit": "%",
                "direction": "maximize",
                "threshold": 0.2,
            },
            {
                "tag": "STEAM_UNIT",
                "name": "톤당 스팀 원단위",
                "unit": "GJ/t",
                "direction": "minimize",
                "threshold": 0.05,
            },
        ],
    }


def create_sample_ncc_dataframe() -> pd.DataFrame:
    """
    외부 CSV가 없을 때도 Step 1~Step 3 전체 기능을 즉시 검증할 수 있도록
    NCC 및 C2 정류 공정의 시계열 운전 이력 샘플 DataFrame을 생성합니다.
    """
    timestamps = pd.date_range("2026-05-01 00:00:00", periods=16, freq="1h")
    rows = [
        # (FI-101, TI-102, COT, DSR, PDI-205, TMT-210, C2_YIELD, STEAM_UNIT)
        (45.0, 585.0, 834.0, 0.48, 0.38, 1035.0, 31.80, 2.42),
        (45.1, 585.5, 838.5, 0.50, 0.40, 1044.0, 32.25, 2.34),  # Golden Case 1 (0 -> 1)
        (45.2, 586.0, 838.5, 0.50, 0.41, 1045.0, 32.28, 2.33),
        (46.5, 582.0, 835.0, 0.47, 0.42, 1040.0, 31.65, 2.45),
        (46.4, 583.0, 840.0, 0.51, 0.44, 1051.0, 32.10, 2.36),  # Golden Case 2 (3 -> 4)
        (46.2, 583.5, 840.0, 0.51, 0.45, 1052.0, 32.12, 2.35),
        (44.0, 588.0, 836.0, 0.52, 0.46, 1046.0, 31.95, 2.40),
        (44.1, 588.5, 839.0, 0.49, 0.47, 1050.0, 32.30, 2.31),  # Golden Case 3 (6 -> 7)
        (44.3, 587.5, 838.0, 0.49, 0.48, 1048.0, 32.15, 2.33),
        (47.0, 580.0, 833.5, 0.46, 0.49, 1042.0, 31.50, 2.48),
        (47.1, 581.0, 837.5, 0.49, 0.51, 1053.0, 31.88, 2.39),  # Golden Case 4 (9 -> 10)
        (47.0, 581.5, 841.0, 0.51, 0.53, 1060.0, 32.32, 2.30),  # Golden Case 5 (10 -> 11)
        (46.8, 582.0, 840.5, 0.50, 0.54, 1058.0, 32.28, 2.31),
        (45.5, 584.0, 835.5, 0.48, 0.35, 1032.0, 31.75, 2.44),
        (45.6, 584.5, 839.5, 0.50, 0.37, 1043.0, 32.18, 2.35),  # Golden Case 6 (13 -> 14)
        (45.5, 585.0, 839.5, 0.50, 0.38, 1044.0, 32.20, 2.34),
    ]
    data = []
    for ts, r in zip(timestamps, rows):
        data.append(
            {
                "Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "FI-101": r[0],
                "TI-102": r[1],
                "COT": r[2],
                "DSR": r[3],
                "PDI-205": r[4],
                "TMT-210": r[5],
                "C2_YIELD": r[6],
                "STEAM_UNIT": r[7],
            }
        )
    return pd.DataFrame(data)


def category_items_to_df(items: List[Dict[str, Any]], category: str) -> pd.DataFrame:
    """
    카테고리별 변수 매핑 리스트(dict)를 st.data_editor 입력용 DataFrame으로 변환합니다.
    빈 테이블이거나 결측값이 있을 때도 컬럼 타입(str, float64)이 엄격히 유지되도록 보장합니다.
    """
    rows = items if isinstance(items, list) else []

    if category in ("feed", "control"):
        df = pd.DataFrame(rows, columns=["tag", "name", "unit"])
        for col in ["tag", "name", "unit"]:
            df[col] = df[col].fillna("").astype(str)
        return df

    if category == "state":
        df = pd.DataFrame(rows, columns=["tag", "name", "unit", "limit_desc"])
        for col in ["tag", "name", "unit", "limit_desc"]:
            df[col] = df[col].fillna("").astype(str)
        return df

    if category == "kpi":
        df = pd.DataFrame(
            rows, columns=["tag", "name", "unit", "direction", "threshold"]
        )
        for col in ["tag", "name", "unit"]:
            df[col] = df[col].fillna("").astype(str)
        df["direction"] = (
            df["direction"]
            .fillna("maximize")
            .astype(str)
            .apply(lambda x: x.strip() if x.strip() in ("maximize", "minimize") else "maximize")
        )
        df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce").fillna(0.1).astype("float64")
        return df

    raise ValueError(f"알 수 없는 변수 카테고리입니다: {category}")


def df_to_category_items(df: Optional[pd.DataFrame], category: str) -> List[Dict[str, Any]]:
    """
    st.data_editor에서 편집된 DataFrame을 정제된 dict 리스트로 변환합니다.
    태그명(tag)과 한글 명칭(name)이 모두 비어 있는 빈 행은 자동으로 제외합니다.
    """
    if df is None or df.empty:
        return []

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        raw_tag = row.get("tag", "")
        raw_name = row.get("name", "")
        raw_unit = row.get("unit", "")

        tag = "" if pd.isna(raw_tag) else str(raw_tag).strip()
        name = "" if pd.isna(raw_name) else str(raw_name).strip()
        unit = "" if pd.isna(raw_unit) else str(raw_unit).strip()

        # 태그명과 한글 명칭이 모두 비어 있으면 미입력 행으로 간주하여 스킵
        if not tag and not name:
            continue

        # 태그명만 입력하고 한글 명칭을 비워둔 경우 태그명을 기본 명칭으로 보완하지 않고 그대로 유지하되,
        # 둘 중 하나라도 있으면 유효 행으로 인정
        if category in ("feed", "control"):
            records.append(
                {
                    "tag": tag,
                    "name": name,
                    "unit": unit,
                }
            )
        elif category == "state":
            raw_limit = row.get("limit_desc", "")
            limit_desc = "" if pd.isna(raw_limit) else str(raw_limit).strip()
            records.append(
                {
                    "tag": tag,
                    "name": name,
                    "unit": unit,
                    "limit_desc": limit_desc,
                }
            )
        elif category == "kpi":
            raw_dir = row.get("direction", "maximize")
            direction = "" if pd.isna(raw_dir) else str(raw_dir).strip().lower()
            if direction not in ("maximize", "minimize"):
                direction = "maximize"

            raw_thresh = row.get("threshold", 0.1)
            try:
                threshold = float(raw_thresh) if not pd.isna(raw_thresh) else 0.1
            except (ValueError, TypeError):
                threshold = 0.1

            records.append(
                {
                    "tag": tag,
                    "name": name,
                    "unit": unit,
                    "direction": direction,
                    "threshold": threshold,
                }
            )

    return records


def build_mapping_base_dfs(
    var_mapping: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, pd.DataFrame]:
    """4대 변수 매핑 딕셔너리로부터 st.data_editor 초기 바인딩용 DataFrame 사전을 생성합니다."""
    return {
        cat: category_items_to_df(var_mapping.get(cat, []), cat)
        for cat in CATEGORY_KEYS
    }


def sync_editor_base_dfs_from_mapping() -> None:
    """
    프로그래밍 방식(프리셋 불러오기, 예시 채우기, 컬럼 빠른 추가, 초기화 등)으로
    st.session_state['var_mapping']이 갱신되었을 때 st.data_editor 위젯에 즉시 반영되도록
    베이스 DataFrame을 재구축하고 위젯 버전 키를 증가시킵니다.
    """
    st.session_state["mapping_base_dfs"] = build_mapping_base_dfs(
        st.session_state["var_mapping"]
    )
    st.session_state["mapping_editor_version"] = (
        st.session_state.get("mapping_editor_version", 0) + 1
    )


def init_session_state() -> None:
    """단계 간 이동, 리샘플링, 표 편집 시 데이터가 초기화되지 않도록 핵심 상태를 초기화합니다."""
    defaults = {
        "process_name": "나프타 열분해 및 C2 정류 공정",
        "source_mode": "① 내 PC에서 CSV 직접 업로드",
        "loaded_file_id": None,
        "raw_df": None,
        "processed_df": None,
        "time_col": None,
        "resample_rule": "원본 유지(집계 안 함)",
        "resample_agg": "평균값(mean)",
        "resample_summary_msg": None,
        # Phase 2: 4대 공정 변수 매핑 상태
        "var_mapping": {
            "feed": [],
            "control": [],
            "state": [],
            "kpi": [],
        },
        "mapping_editor_version": 0,
        "selected_preset_file": None,
        "preset_flash_msg": None,
        "preset_flash_type": "success",
        # Phase 3: 최적 운전 구간(Before -> After) 추출 및 보충 멘트 편집 상태
        "lookahead_rows": 1,
        "golden_base_df": None,
        "golden_cases_df": None,
        "golden_editor_version": 0,
        "golden_extraction_sig": None,
        "golden_notes_cache": {},
        "last_golden_notes_snapshot": {},
        "preview_case_select": None,
        "step3_flash_msg": None,
        "step3_flash_type": "success",
        # Phase 4: JSONL 학습/검증 분할 생성 및 dataset_info.json 자동 등록 상태
        "dataset_name": "ncc_opt_v1",
        "val_split_pct": 10,
        "last_export_result": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    if "mapping_base_dfs" not in st.session_state:
        st.session_state["mapping_base_dfs"] = build_mapping_base_dfs(
            st.session_state["var_mapping"]
        )


# ==============================================================================
# 1. 유틸리티 함수 (CSV 로드, 시간 컬럼 감지, 시계열 리샘플링, YAML 프리셋, 골든 케이스 추출/문장 생성)
# ==============================================================================
def load_csv_safely(file_source) -> pd.DataFrame:
    """인코딩(utf-8, utf-8-sig, cp949, euc-kr)을 순차 시도하여 CSV를 안전하게 로드합니다."""
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    for enc in encodings:
        try:
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            return pd.read_csv(file_source, encoding=enc)
        except Exception:
            continue
    if hasattr(file_source, "seek"):
        file_source.seek(0)
    return pd.read_csv(file_source)


def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """데이터프레임에서 Timestamp/DateTime 성격의 시간 기준 컬럼을 자동 감지합니다."""
    if df is None or df.empty:
        return None

    time_keywords = ["time", "date", "timestamp", "datetime", "일시", "시간", "시각", "일자"]
    # 1순위: 컬럼명에 키워드가 포함된 경우
    for col in df.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in time_keywords):
            return col

    # 2순위: 이미 datetime 타입인 컬럼
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col

    # 3순위: 문자열/객체 컬럼 중 상위 샘플이 날짜/시간으로 파싱 가능한 컬럼
    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().head(10)
            if len(sample) > 0:
                try:
                    pd.to_datetime(sample, errors="raise")
                    return col
                except Exception:
                    continue

    return df.columns[0] if len(df.columns) > 0 else None


def perform_resampling(
    df: pd.DataFrame,
    time_col: Optional[str],
    rule_label: str,
    agg_label: str,
) -> Tuple[pd.DataFrame, str]:
    """
    선택된 주기 및 집계 함수에 따라 시계열 데이터를 축소(Resampling/Aggregate)합니다.
    수치형 컬럼은 선택된 집계 함수(mean/last/median)를 적용하고,
    문자열/코멘트 등 비수치형 컬럼은 안전하게 'last'(결측 시 'first' 보완)로 처리합니다.
    """
    orig_count = len(df)

    rule_map = {
        "원본 유지(집계 안 함)": None,
        "1분(1min)": "1min",
        "10분(10min)": "10min",
        "30분(30min)": "30min",
        "1시간(1h)": "1h",
        "4시간(4h)": "4h",
    }
    agg_map = {
        "평균값(mean)": "mean",
        "마지막 값(last)": "last",
        "중앙값(median)": "median",
    }

    freq = rule_map.get(rule_label)
    agg_func = agg_map.get(agg_label, "mean")

    # 원본 유지 선택 시
    if freq is None:
        out_df = df.copy()
        if time_col and time_col in out_df.columns:
            parsed_dt = pd.to_datetime(out_df[time_col], errors="coerce")
            if parsed_dt.notna().sum() > 0:
                out_df[time_col] = parsed_dt
                out_df = out_df.sort_values(by=time_col).reset_index(drop=True)
        msg = f"원본 {orig_count:,}행 -> 집계 후 {len(out_df):,}행으로 유지 완료 (집계 안 함)"
        return out_df, msg

    if not time_col or time_col not in df.columns:
        raise ValueError("시계열 집계를 수행하려면 유효한 '시간 기준 컬럼'을 선택해야 합니다.")

    work_df = df.copy()
    work_df[time_col] = pd.to_datetime(work_df[time_col], errors="coerce")
    invalid_rows = work_df[time_col].isna().sum()
    work_df = work_df.dropna(subset=[time_col]).sort_values(by=time_col)

    if work_df.empty:
        raise ValueError(
            f"선택한 컬럼 '{time_col}'을 날짜/시간(Datetime) 형식으로 변환할 수 없습니다. 다른 컬럼을 선택해 주세요."
        )

    other_cols = [c for c in work_df.columns if c != time_col]
    numeric_cols = [
        c for c in other_cols if pd.api.types.is_numeric_dtype(work_df[c])
    ]
    non_numeric_cols = [c for c in other_cols if c not in numeric_cols]

    agg_dict = {}
    for c in numeric_cols:
        agg_dict[c] = agg_func
    for c in non_numeric_cols:
        # 문자열 컬럼이나 운전원 코멘트 컬럼은 안전하게 last로 처리
        agg_dict[c] = "last"

    resampled = (
        work_df.set_index(time_col)
        .resample(freq)
        .agg(agg_dict)
    )
    # 모든 수치 컬럼이 전부 비어 있는 빈 시간 구간 제거
    if numeric_cols:
        resampled = resampled.dropna(how="all", subset=numeric_cols)
    else:
        resampled = resampled.dropna(how="all")

    resampled = resampled.reset_index()
    new_count = len(resampled)

    extra_info = f" (시간 파싱 불가 {invalid_rows:,}행 제외)" if invalid_rows > 0 else ""
    msg = (
        f"원본 {orig_count:,}행 -> 집계 후 {new_count:,}행으로 축소 완료 "
        f"[주기: {rule_label} | 수치 집계: {agg_label} | 문자열/코멘트: last]{extra_info}"
    )
    return resampled, msg


def sanitize_preset_filename(process_name: str) -> str:
    """공정 명칭을 파일 시스템에 안전한 프리셋 파일명({공정명}.yaml)으로 변환합니다."""
    cleaned = (process_name or "").strip()
    if not cleaned:
        cleaned = "default_process"
    # 경로 구분자 및 파일명 금지 문자만 안전하게 치환 (한글, 영문, 숫자, 공백, 괄호, 하이픈 등은 보존)
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", cleaned)
    if cleaned.lower().endswith((".yaml", ".yml")):
        return cleaned
    return f"{cleaned}.yaml"


def list_preset_files() -> List[str]:
    """./presets 폴더 내의 .yaml 및 .yml 파일 목록을 정렬하여 반환합니다."""
    ensure_directories()
    yaml_files = glob.glob(os.path.join(PRESETS_DIR, "*.yaml")) + glob.glob(
        os.path.join(PRESETS_DIR, "*.yml")
    )
    filenames = sorted({os.path.basename(p) for p in yaml_files})
    return filenames


def save_preset_to_yaml(
    process_name: str,
    var_mapping: Dict[str, List[Dict[str, Any]]],
) -> str:
    """
    현재 설정된 공정 명칭과 4대 변수 매핑 정보를 ./presets/{공정명}.yaml 파일로 저장합니다.
    반환값은 저장된 상대 경로 문자열입니다.
    """
    ensure_directories()
    filename = sanitize_preset_filename(process_name)
    filepath = os.path.join(PRESETS_DIR, filename)

    payload = {
        "process_name": (process_name or "").strip() or "미지정 공정",
        "feed": var_mapping.get("feed", []),
        "control": var_mapping.get("control", []),
        "state": var_mapping.get("state", []),
        "kpi": var_mapping.get("kpi", []),
    }

    with open(filepath, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            payload,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )

    return filepath


def _normalize_loaded_items(raw_list: Any, category: str) -> List[Dict[str, Any]]:
    """YAML에서 읽어들인 리스트 데이터를 카테고리 스키마에 맞게 정규화합니다."""
    if not isinstance(raw_list, list):
        return []

    normalized_rows: List[Dict[str, Any]] = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        row = {
            "tag": item.get("tag", item.get("태그명", "")),
            "name": item.get("name", item.get("한글 명칭", item.get("한글명칭", ""))),
            "unit": item.get("unit", item.get("단위", "")),
        }
        if category == "state":
            row["limit_desc"] = item.get(
                "limit_desc", item.get("관리 기준 설명", item.get("관리기준", ""))
            )
        elif category == "kpi":
            row["direction"] = item.get(
                "direction", item.get("목표 방향", item.get("목표방향", "maximize"))
            )
            row["threshold"] = item.get(
                "threshold", item.get("개선 추출 기준값", item.get("기준값", 0.1))
            )
        normalized_rows.append(row)

    df = category_items_to_df(normalized_rows, category)
    return df_to_category_items(df, category)


def load_preset_from_yaml(
    filename: str,
) -> Tuple[Optional[str], Dict[str, List[Dict[str, Any]]]]:
    """
    ./presets/{filename} 파일을 읽어 (공정명, 4대 변수 매핑 사전)을 반환합니다.
    """
    filepath = os.path.join(PRESETS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"프리셋 파일을 찾을 수 없습니다: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("유효하지 않은 YAML 프리셋 형식입니다 (최상위 구조가 dict여야 합니다).")

    loaded_process_name = data.get("process_name")
    if loaded_process_name is not None:
        loaded_process_name = str(loaded_process_name).strip()

    # 최상위 키 또는 'variables' 하위 키 모두 호환 지원
    var_section = data.get("variables") if isinstance(data.get("variables"), dict) else data

    loaded_mapping: Dict[str, List[Dict[str, Any]]] = {}
    for cat in CATEGORY_KEYS:
        loaded_mapping[cat] = _normalize_loaded_items(var_section.get(cat, []), cat)

    return loaded_process_name, loaded_mapping


# ------------------------------------------------------------------------------
# Phase 3 핵심 로직: 최적 운전 구간(Before -> After) 탐색 및 한국어 문장(Prompt) 생성
# ------------------------------------------------------------------------------
def _format_num(val: Any, digits: int = 2) -> str:
    """수치를 읽기 쉬운 문자열로 포맷팅합니다."""
    if val is None or pd.isna(val):
        return "N/A"
    try:
        fval = float(val)
        if math.isnan(fval):
            return "N/A"
        if abs(fval - round(fval)) < 1e-9:
            return f"{fval:.1f}"
        return f"{fval:.{digits}f}".rstrip("0").rstrip(".")
    except (ValueError, TypeError):
        return str(val)


def _format_signed_delta(delta: float, digits: int = 2) -> str:
    """변화량(Delta)에 부호(+/-)를 붙여 포맷팅합니다."""
    if pd.isna(delta) or math.isnan(delta):
        return "0.0"
    sign = "+" if delta >= 0 else ""
    formatted = f"{delta:.{digits}f}".rstrip("0").rstrip(".")
    if formatted in ("-0", "0", ""):
        formatted = "0.0"
    return f"{sign}{formatted}"


def _format_time_value(row: pd.Series, time_col: Optional[str], row_idx: int) -> str:
    """행의 시간 컬럼 값을 사람이 읽기 좋은 문자열로 변환합니다."""
    if time_col and time_col in row.index:
        val = row[time_col]
        if pd.notna(val):
            if isinstance(val, pd.Timestamp):
                return val.strftime("%Y-%m-%d %H:%M:%S")
            return str(val)
    return f"Row #{row_idx}"


def _display_var_label(item: Dict[str, Any]) -> str:
    """변수 메타정보에서 '한글 명칭(태그명)' 형태의 레이블을 생성합니다."""
    tag = (item.get("tag") or "").strip()
    name = (item.get("name") or "").strip()
    if name and tag and name != tag:
        return f"{name}({tag})"
    return name or tag or "미지정 변수"


def _describe_lookahead_interval(lookahead_rows: int, resample_rule: str) -> str:
    """현재 집계 주기와 lookahead_rows를 결합하여 시간 간격 설명 문자열을 반환합니다."""
    minutes_map = {
        "1분(1min)": 1,
        "10분(10min)": 10,
        "30분(30min)": 30,
        "1시간(1h)": 60,
        "4시간(4h)": 240,
    }
    unit_min = minutes_map.get(resample_rule)
    if unit_min is None:
        return f"원본 데이터 기준 {lookahead_rows}행 뒤 시점"

    total_min = unit_min * lookahead_rows
    if total_min >= 60 and total_min % 60 == 0:
        hours = total_min // 60
        return f"{lookahead_rows}행 뒤 (약 {hours}시간 후, 집계 주기: {resample_rule})"
    if total_min >= 60:
        hours = total_min // 60
        rem_min = total_min % 60
        return f"{lookahead_rows}행 뒤 (약 {hours}시간 {rem_min}분 후, 집계 주기: {resample_rule})"
    return f"{lookahead_rows}행 뒤 (약 {total_min}분 후, 집계 주기: {resample_rule})"


def extract_golden_cases(
    proc_df: pd.DataFrame,
    var_mapping: Dict[str, List[Dict[str, Any]]],
    time_col: Optional[str],
    lookahead_rows: int,
    notes_cache: Optional[Dict[Tuple[int, int], Dict[str, Any]]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    설정한 모든 KPI의 threshold 조건을 동시에 만족하는 최적 운전 구간(Before 행 -> After 행)을
    자동 탐색하여 st.data_editor 편집용 DataFrame으로 반환합니다.
    """
    diagnostics: List[str] = []
    if proc_df is None or proc_df.empty:
        return pd.DataFrame(), ["집계된 데이터프레임이 비어 있습니다."]

    n_rows = len(proc_df)
    if lookahead_rows < 1 or lookahead_rows >= n_rows:
        return pd.DataFrame(), [
            f"비교 시점 간격(lookahead_rows={lookahead_rows})이 전체 데이터 행 수({n_rows}행)보다 크거나 같아 비교할 수 없습니다."
        ]

    kpi_items = [
        item
        for item in var_mapping.get("kpi", [])
        if (item.get("tag") or "").strip() in proc_df.columns
    ]
    if not kpi_items:
        return pd.DataFrame(), [
            "Step 2에서 매핑된 KPI 변수 중 현재 CSV 컬럼에 존재하는 태그가 없습니다."
        ]

    control_items = [
        item
        for item in var_mapping.get("control", [])
        if (item.get("tag") or "").strip() in proc_df.columns
    ]
    state_items = [
        item
        for item in var_mapping.get("state", [])
        if (item.get("tag") or "").strip() in proc_df.columns
    ]

    # 숫자형 시리즈 사전 변환
    numeric_cache: Dict[str, pd.Series] = {}
    for item in kpi_items + control_items + state_items:
        tag = item["tag"].strip()
        if tag not in numeric_cache:
            numeric_cache[tag] = pd.to_numeric(proc_df[tag], errors="coerce")

    notes_cache = notes_cache or {}
    golden_rows: List[Dict[str, Any]] = []
    case_seq = 1

    # 각 KPI별 최대 개선폭 추적 (조건 만족 구간이 0건일 때 엔지니어에게 힌트 제공용)
    max_improvements: Dict[str, float] = {}

    for i in range(n_rows - lookahead_rows):
        j = i + lookahead_rows
        all_kpi_passed = True
        kpi_summary_parts: List[str] = []

        for kpi in kpi_items:
            tag = kpi["tag"].strip()
            direction = (kpi.get("direction") or "maximize").strip().lower()
            threshold = float(kpi.get("threshold", 0.0) or 0.0)
            unit = (kpi.get("unit") or "").strip()
            unit_str = f" {unit}" if unit else ""
            label = (kpi.get("name") or tag).strip()

            v_before = numeric_cache[tag].iloc[i]
            v_after = numeric_cache[tag].iloc[j]

            if pd.isna(v_before) or pd.isna(v_after):
                all_kpi_passed = False
                break

            raw_delta = float(v_after - v_before)
            improvement = raw_delta if direction == "maximize" else -raw_delta

            if tag not in max_improvements or improvement > max_improvements[tag]:
                max_improvements[tag] = improvement

            # 부동소수점 오차 보정을 위해 1e-9 여유 적용
            if improvement + 1e-9 < threshold:
                all_kpi_passed = False
                break

            kpi_summary_parts.append(
                f"{label}: {_format_num(v_before)}→{_format_num(v_after)}{unit_str} ({_format_signed_delta(raw_delta)}{unit_str})"
            )

        if not all_kpi_passed:
            continue

        # 제어 변수(control) 변화 요약 생성
        ctrl_summary_parts: List[str] = []
        for ctrl in control_items:
            c_tag = ctrl["tag"].strip()
            c_unit = (ctrl.get("unit") or "").strip()
            c_unit_str = f" {c_unit}" if c_unit else ""
            c_label = (ctrl.get("name") or c_tag).strip()
            c_before = numeric_cache[c_tag].iloc[i]
            c_after = numeric_cache[c_tag].iloc[j]
            if pd.notna(c_before) and pd.notna(c_after):
                c_delta = float(c_after - c_before)
                ctrl_summary_parts.append(
                    f"{c_label}: {_format_num(c_before)}→{_format_num(c_after)}{c_unit_str} ({_format_signed_delta(c_delta)})"
                )
        ctrl_summary_str = (
            " | ".join(ctrl_summary_parts)
            if ctrl_summary_parts
            else "(매핑된 제어 변수 없음)"
        )

        # 상태/제약 변수(state) 요약 생성
        state_summary_parts: List[str] = []
        for st_item in state_items:
            s_tag = st_item["tag"].strip()
            s_unit = (st_item.get("unit") or "").strip()
            s_unit_str = f" {s_unit}" if s_unit else ""
            s_label = (st_item.get("name") or s_tag).strip()
            s_limit = (st_item.get("limit_desc") or "").strip()
            s_before = numeric_cache[s_tag].iloc[i]
            s_after = numeric_cache[s_tag].iloc[j]
            if pd.notna(s_before) and pd.notna(s_after):
                limit_note = f" [{s_limit}]" if s_limit else ""
                state_summary_parts.append(
                    f"{s_label}: {_format_num(s_before)}→{_format_num(s_after)}{s_unit_str}{limit_note}"
                )
        state_summary_str = (
            " | ".join(state_summary_parts)
            if state_summary_parts
            else "(매핑된 상태 변수 없음)"
        )

        before_row = proc_df.iloc[i]
        after_row = proc_df.iloc[j]
        before_time_str = _format_time_value(before_row, time_col, i)
        after_time_str = _format_time_value(after_row, time_col, j)

        cached = notes_cache.get((i, j), {})
        include_flag = bool(cached.get("포함 여부", True))
        expert_note = str(cached.get("Expert_Note", "") or "")
        action_reason = str(cached.get("Action_Reason", "") or "")

        golden_rows.append(
            {
                "포함 여부": include_flag,
                "케이스 ID": f"CASE-{case_seq:03d}",
                "Before 행": int(i),
                "After 행": int(j),
                "Before 시점": before_time_str,
                "After 시점": after_time_str,
                "제어 변수 변화 (Before→After)": ctrl_summary_str,
                "KPI 개선 결과 (Before→After)": " | ".join(kpi_summary_parts),
                "상태/제약 추이": state_summary_str,
                "Expert_Note": expert_note,
                "Action_Reason": action_reason,
            }
        )
        case_seq += 1

    if not golden_rows:
        for kpi in kpi_items:
            tag = kpi["tag"].strip()
            label = _display_var_label(kpi)
            direction = kpi.get("direction", "maximize")
            thresh = kpi.get("threshold", 0.0)
            unit = (kpi.get("unit") or "").strip()
            unit_str = f" {unit}" if unit else ""
            best_val = max_improvements.get(tag)
            if best_val is not None:
                dir_kor = "상승폭" if direction == "maximize" else "절감폭"
                diagnostics.append(
                    f"• {label}: 설정 기준값(threshold) = {thresh}{unit_str} | 현재 데이터 내 최대 {dir_kor} = {_format_num(best_val, 3)}{unit_str}"
                )
        return pd.DataFrame(), diagnostics

    out_df = pd.DataFrame(golden_rows)
    out_df["포함 여부"] = out_df["포함 여부"].fillna(True).astype(bool)
    out_df["Expert_Note"] = out_df["Expert_Note"].fillna("").astype(str)
    out_df["Action_Reason"] = out_df["Action_Reason"].fillna("").astype(str)
    return out_df, diagnostics


def build_default_sample_notes(
    case_row: pd.Series,
    proc_df: pd.DataFrame,
    var_mapping: Dict[str, List[Dict[str, Any]]],
) -> Tuple[str, str]:
    """
    도메인 전문가가 보충 멘트 작성 방식을 빠르게 참고할 수 있도록
    해당 케이스의 실제 수치 변화를 반영한 예시 Expert_Note 및 Action_Reason 문구를 생성합니다.
    """
    before_idx = int(case_row.get("Before 행", 0))
    after_idx = int(case_row.get("After 행", 0))
    before_s = proc_df.iloc[before_idx]
    after_s = proc_df.iloc[after_idx]

    feed_items = [
        it for it in var_mapping.get("feed", []) if (it.get("tag") or "").strip() in proc_df.columns
    ]
    ctrl_items = [
        it for it in var_mapping.get("control", []) if (it.get("tag") or "").strip() in proc_df.columns
    ]
    state_items = [
        it for it in var_mapping.get("state", []) if (it.get("tag") or "").strip() in proc_df.columns
    ]
    kpi_items = [
        it for it in var_mapping.get("kpi", []) if (it.get("tag") or "").strip() in proc_df.columns
    ]

    # Expert_Note 예시 조립
    feed_desc = ""
    if feed_items:
        f0 = feed_items[0]
        f_tag = f0["tag"].strip()
        f_name = (f0.get("name") or f_tag).strip()
        f_unit = (f0.get("unit") or "").strip()
        f_val = _format_num(before_s.get(f_tag))
        feed_desc = f"{f_name}({f_val} {f_unit}) 투입 조건에서 ".strip() + " "

    kpi_desc = "핵심 KPI가 목표 수준 대비 다소 정체된 상태임"
    if kpi_items:
        k0 = kpi_items[0]
        k_tag = k0["tag"].strip()
        k_name = (k0.get("name") or k_tag).strip()
        k_unit = (k0.get("unit") or "").strip()
        k_val = _format_num(before_s.get(k_tag))
        kpi_desc = f"초기 {k_name}이(가) {k_val} {k_unit} 수준에 머물러 열분해/분리 효율 개선 여지가 확인됨"

    sample_expert_note = (
        f"{feed_desc}{kpi_desc}. 운전 조건 미세 조정을 통한 수율 및 에너지 원단위 동시 최적화가 필요한 구간임."
    )

    # Action_Reason 예시 조립
    state_desc = "주요 공정 상태 변수가 모두 안전 관리 기준 내에 위치하여"
    if state_items:
        s0 = state_items[0]
        s_tag = s0["tag"].strip()
        s_name = (s0.get("name") or s_tag).strip()
        s_unit = (s0.get("unit") or "").strip()
        s_limit = (s0.get("limit_desc") or "안전 한계").strip()
        s_val = _format_num(after_s.get(s_tag))
        state_desc = f"{s_name}이(가) {s_val} {s_unit}로 관리 기준({s_limit}) 대비 충분한 안전 여유도를 확보하고 있어"

    ctrl_desc = "제어 변수 설정치를 최적 방향으로 조정함"
    if ctrl_items:
        c0 = ctrl_items[0]
        c_tag = c0["tag"].strip()
        c_name = (c0.get("name") or c_tag).strip()
        c_unit = (c0.get("unit") or "").strip()
        v_b = pd.to_numeric(before_s.get(c_tag), errors="coerce")
        v_a = pd.to_numeric(after_s.get(c_tag), errors="coerce")
        if pd.notna(v_b) and pd.notna(v_a):
            diff = float(v_a - v_b)
            act_word = "상향" if diff > 0 else ("하향" if diff < 0 else "유지")
            ctrl_desc = f"{c_name}을(를) {_format_num(v_b)}→{_format_num(v_a)} {c_unit}로 {act_word} 조작하여 반응 가혹도 및 열물질 평형을 개선함"

    sample_action_reason = f"{state_desc} {ctrl_desc}."
    return sample_expert_note, sample_action_reason


def generate_qa_from_golden_case(
    case_row: pd.Series,
    proc_df: pd.DataFrame,
    var_mapping: Dict[str, List[Dict[str, Any]]],
    process_name: str,
    time_col: Optional[str],
    lookahead_rows: int,
    resample_rule: str,
) -> Dict[str, str]:
    """
    외부 LLM 없이 순수 파이썬 규칙 기반 템플릿(f-string)으로
    수치 데이터(Before -> After) + 4대 변수 메타정보 + 전문가 보충 멘트(Expert_Note, Action_Reason)를
    결합하여 자연스러운 한국어 문장형 instruction, input, output 사전을 생성합니다.
    """
    proc_title = (process_name or "").strip() or "화학 공정"
    before_idx = int(case_row.get("Before 행", 0))
    after_idx = int(case_row.get("After 행", 0))

    before_s = proc_df.iloc[before_idx]
    after_s = proc_df.iloc[after_idx]

    before_time = _format_time_value(before_s, time_col, before_idx)
    after_time = _format_time_value(after_s, time_col, after_idx)
    lookahead_desc = _describe_lookahead_interval(lookahead_rows, resample_rule)

    expert_note = str(case_row.get("Expert_Note", "") or "").strip()
    action_reason = str(case_row.get("Action_Reason", "") or "").strip()

    feed_items = [
        it for it in var_mapping.get("feed", []) if (it.get("tag") or "").strip() in proc_df.columns
    ]
    ctrl_items = [
        it for it in var_mapping.get("control", []) if (it.get("tag") or "").strip() in proc_df.columns
    ]
    state_items = [
        it for it in var_mapping.get("state", []) if (it.get("tag") or "").strip() in proc_df.columns
    ]
    kpi_items = [
        it for it in var_mapping.get("kpi", []) if (it.get("tag") or "").strip() in proc_df.columns
    ]

    # 1) Instruction 조립
    kpi_goal_phrases: List[str] = []
    for kpi in kpi_items:
        label = _display_var_label(kpi)
        direction = (kpi.get("direction") or "maximize").strip().lower()
        dir_kor = "극대화(maximize)" if direction == "maximize" else "최소화(minimize)"
        kpi_goal_phrases.append(f"{label} {dir_kor}")

    kpi_goal_text = (
        ", ".join(kpi_goal_phrases)
        if kpi_goal_phrases
        else "목표 성과(KPI) 개선"
    )

    instruction = (
        f"당신은 '{proc_title}'의 운전 최적화를 담당하는 화학공학 도메인 전문가 AI입니다. "
        f"주어진 현재 운전 데이터(원료/투입 조건, 제어 변수 설정값, 공정 상태 및 안전 제약, 현재 KPI)를 종합적으로 분석하여, "
        f"설비 안전 제약 조건을 준수하면서 {kpi_goal_text}을(를) 달성할 수 있는 공정 상태 진단, "
        f"제어 변수(Control) 최적 조작 가이드, 조작 근거 및 기대되는 KPI 개선 효과를 구체적인 수치와 함께 제시하세요."
    )

    # 2) Input 조립 (Before 시점 상태 및 제약 조건 서술)
    feed_lines: List[str] = []
    for item in feed_items:
        tag = item["tag"].strip()
        label = _display_var_label(item)
        unit = (item.get("unit") or "").strip()
        unit_str = f" {unit}" if unit else ""
        val_b = _format_num(before_s.get(tag))
        feed_lines.append(f"  - {label}: {val_b}{unit_str}")
    if not feed_lines:
        feed_lines.append("  - (등록된 원료/투입 변수 없음)")

    ctrl_input_lines: List[str] = []
    for item in ctrl_items:
        tag = item["tag"].strip()
        label = _display_var_label(item)
        unit = (item.get("unit") or "").strip()
        unit_str = f" {unit}" if unit else ""
        val_b = _format_num(before_s.get(tag))
        ctrl_input_lines.append(f"  - {label}: {val_b}{unit_str}")
    if not ctrl_input_lines:
        ctrl_input_lines.append("  - (등록된 제어/조작 변수 없음)")

    state_input_lines: List[str] = []
    for item in state_items:
        tag = item["tag"].strip()
        label = _display_var_label(item)
        unit = (item.get("unit") or "").strip()
        unit_str = f" {unit}" if unit else ""
        limit_desc = (item.get("limit_desc") or "").strip()
        limit_str = f" (관리 기준: {limit_desc})" if limit_desc else ""
        val_b = _format_num(before_s.get(tag))
        state_input_lines.append(f"  - {label}: {val_b}{unit_str}{limit_str}")
    if not state_input_lines:
        state_input_lines.append("  - (등록된 상태/제약 변수 없음)")

    kpi_input_lines: List[str] = []
    for item in kpi_items:
        tag = item["tag"].strip()
        label = _display_var_label(item)
        unit = (item.get("unit") or "").strip()
        unit_str = f" {unit}" if unit else ""
        direction = (item.get("direction") or "maximize").strip().lower()
        dir_kor = "극대화(maximize)" if direction == "maximize" else "최소화(minimize)"
        thresh = _format_num(item.get("threshold", 0.0), 4)
        val_b = _format_num(before_s.get(tag))
        kpi_input_lines.append(
            f"  - {label}: {val_b}{unit_str} [목표 방향: {dir_kor}, 최소 개선 기준: {thresh}{unit_str} 이상]"
        )
    if not kpi_input_lines:
        kpi_input_lines.append("  - (등록된 목표 성과 변수 없음)")

    input_sections = [
        f"[공정 명칭] {proc_title}",
        f"[운전 기준 시점 (Before)] {before_time} (성능 비교 간격: {lookahead_desc})",
        "",
        "1. 원료/투입 조건 (Feed - 외생 전제 조건):",
        "\n".join(feed_lines),
        "",
        "2. 현재 제어/조작 변수 설정값 (Control - 조작 전):",
        "\n".join(ctrl_input_lines),
        "",
        "3. 현재 공정 상태 및 안전 제약 조건 (State & Constraints):",
        "\n".join(state_input_lines),
        "",
        "4. 현재 목표 성과 지표 (KPI - 조작 전):",
        "\n".join(kpi_input_lines),
    ]
    input_text = "\n".join(input_sections)

    # 3) Output 조립 (전문가 진단 + 제어 변수 조작 권고 + 조작 핵심 근거 + 기대 KPI 개선 효과)
    # 3-1) 공정 상태 및 비효율 원인 진단 (Expert_Note 반영)
    if expert_note:
        diagnosis_block = (
            f"  - 전문가 현장 진단(Expert_Note): {expert_note}\n"
            f"  - 기준 시점({before_time}) 운전 데이터를 점검한 결과, 원료 투입 조건 대비 주요 KPI의 추가 개선 여력이 존재하며 "
            f"상태/제약 변수가 안전 운전 범위 내에 있어 능동적인 운전 최적화 조작이 가능한 상태입니다."
        )
    else:
        diagnosis_block = (
            f"  - 기준 시점({before_time}) 운전 데이터를 분석한 결과, 현재 원료/투입 조건 하에서 "
            f"공정 상태 변수가 안전 관리 기준 내에 안정적으로 유지되고 있어, 제어 변수 미세 조정을 통해 "
            f"{kpi_goal_text}을(를) 달성할 수 있는 운전 구간으로 진단됩니다."
        )

    # 3-2) 제어 변수(Control) 최적 조작 가이드 (Before -> After 수치 변화)
    ctrl_action_lines: List[str] = []
    for item in ctrl_items:
        tag = item["tag"].strip()
        label = _display_var_label(item)
        unit = (item.get("unit") or "").strip()
        unit_str = f" {unit}" if unit else ""
        v_b = pd.to_numeric(before_s.get(tag), errors="coerce")
        v_a = pd.to_numeric(after_s.get(tag), errors="coerce")
        if pd.notna(v_b) and pd.notna(v_a):
            delta = float(v_a - v_b)
            if abs(delta) < 1e-9:
                action_verb = "현재 설정값 유지"
                delta_desc = f"변화 없음 (0.0{unit_str})"
            elif delta > 0:
                action_verb = "상향 조작"
                delta_desc = f"{_format_signed_delta(delta)}{unit_str} {action_verb}"
            else:
                action_verb = "하향 조작"
                delta_desc = f"{_format_signed_delta(delta)}{unit_str} {action_verb}"
            ctrl_action_lines.append(
                f"  - {label}: {_format_num(v_b)}{unit_str} → {_format_num(v_a)}{unit_str} ({delta_desc})"
            )
        else:
            ctrl_action_lines.append(
                f"  - {label}: {_format_num(before_s.get(tag))}{unit_str} → {_format_num(after_s.get(tag))}{unit_str}"
            )
    if not ctrl_action_lines:
        ctrl_action_lines.append("  - (등록된 제어/조작 변수가 없습니다. Step 2에서 control 변수를 매핑해 주세요.)")

    # 3-3) 제어 변수 조작 핵심 근거 및 안전 제약 검토 (Action_Reason + State 추이 반영)
    state_check_lines: List[str] = []
    for item in state_items:
        tag = item["tag"].strip()
        label = _display_var_label(item)
        unit = (item.get("unit") or "").strip()
        unit_str = f" {unit}" if unit else ""
        limit_desc = (item.get("limit_desc") or "").strip()
        limit_clause = f" [관리 기준: {limit_desc} 준수]" if limit_desc else " [안전 범위 내 유지]"
        v_b = _format_num(before_s.get(tag))
        v_a = _format_num(after_s.get(tag))
        state_check_lines.append(
            f"  - 안전 제약 확인 — {label}: {v_b}{unit_str} → {v_a}{unit_str}{limit_clause}"
        )

    reason_lines: List[str] = []
    if action_reason:
        reason_lines.append(f"  - 조작 핵심 근거(Action_Reason): {action_reason}")
    else:
        reason_lines.append(
            "  - 조작 핵심 근거: 공정 체류시간 및 열·물질 수지 응답 특성을 고려하여, 안전 제약을 침범하지 않는 범위 내에서 목표 수율 및 에너지 효율을 개선하도록 제어 변수를 조정하였습니다."
        )
    reason_lines.extend(state_check_lines)

    # 3-4) 기대되는 목표 성과(KPI) 개선 효과 (Before -> After 결과)
    kpi_effect_lines: List[str] = []
    for item in kpi_items:
        tag = item["tag"].strip()
        label = _display_var_label(item)
        unit = (item.get("unit") or "").strip()
        unit_str = f" {unit}" if unit else ""
        direction = (item.get("direction") or "maximize").strip().lower()
        thresh = _format_num(item.get("threshold", 0.0), 4)
        v_b = pd.to_numeric(before_s.get(tag), errors="coerce")
        v_a = pd.to_numeric(after_s.get(tag), errors="coerce")
        if pd.notna(v_b) and pd.notna(v_a):
            raw_delta = float(v_a - v_b)
            if direction == "maximize":
                effect_word = f"{_format_signed_delta(raw_delta)}{unit_str} 상승 (목표 극대화 달성, 기준 +{thresh}{unit_str} 충족)"
            else:
                reduction = float(v_b - v_a)
                effect_word = (
                    f"{_format_signed_delta(raw_delta)}{unit_str} "
                    f"({_format_num(reduction)}{unit_str} 절감/감소, 목표 최소화 달성, 기준 -{thresh}{unit_str} 충족)"
                )
            kpi_effect_lines.append(
                f"  - {label}: {_format_num(v_b)}{unit_str} → {_format_num(v_a)}{unit_str} [{effect_word}]"
            )
        else:
            kpi_effect_lines.append(
                f"  - {label}: {_format_num(before_s.get(tag))}{unit_str} → {_format_num(after_s.get(tag))}{unit_str}"
            )

    output_sections = [
        "1. 공정 상태 및 비효율 원인 진단:",
        diagnosis_block,
        "",
        f"2. 제어 변수(Control) 최적 조작 권고안 (Before: {before_time} → After: {after_time}):",
        "\n".join(ctrl_action_lines),
        "",
        "3. 제어 변수 조작 핵심 근거 및 안전 제약 검토:",
        "\n".join(reason_lines),
        "",
        f"4. 기대 목표 성과(KPI) 개선 효과 ({lookahead_desc} 결과):",
        "\n".join(kpi_effect_lines),
    ]
    output_text = "\n".join(output_sections)

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
    }


def sync_golden_editor_base_df(new_df: pd.DataFrame) -> None:
    """
    프로그래밍 방식으로 골든 케이스 표(golden_base_df)를 갱신할 때
    위젯 버전 키를 증가시켜 st.data_editor에 즉시 반영합니다.
    """
    st.session_state["golden_base_df"] = new_df.copy()
    st.session_state["golden_cases_df"] = new_df.copy()
    st.session_state["golden_editor_version"] = (
        st.session_state.get("golden_editor_version", 0) + 1
    )
    # 스냅샷 갱신
    snapshot: Dict[str, Tuple[str, str]] = {}
    if not new_df.empty and "케이스 ID" in new_df.columns:
        for _, r in new_df.iterrows():
            cid = str(r["케이스 ID"])
            snapshot[cid] = (
                str(r.get("Expert_Note", "") or ""),
                str(r.get("Action_Reason", "") or ""),
            )
    st.session_state["last_golden_notes_snapshot"] = snapshot


def update_notes_cache_from_df(df: Optional[pd.DataFrame]) -> None:
    """사용자가 편집한 포함 여부, Expert_Note, Action_Reason을 (Before 행, After 행) 기준으로 캐시에 보존합니다."""
    if df is None or df.empty:
        return
    cache = dict(st.session_state.get("golden_notes_cache", {}))
    for _, row in df.iterrows():
        try:
            key = (int(row["Before 행"]), int(row["After 행"]))
            cache[key] = {
                "포함 여부": bool(row.get("포함 여부", True)),
                "Expert_Note": str(row.get("Expert_Note", "") or ""),
                "Action_Reason": str(row.get("Action_Reason", "") or ""),
            }
        except Exception:
            continue
    st.session_state["golden_notes_cache"] = cache


# ------------------------------------------------------------------------------
# Phase 4 핵심 로직: JSONL 학습/검증 데이터셋 분할 생성 및 dataset_info.json 병합 등록
# ------------------------------------------------------------------------------
def sanitize_dataset_name(raw_name: str) -> str:
    """
    데이터셋 이름을 파일명 및 LLaMA-Factory dataset_info.json 키로 안전하게 사용할 수 있도록 정제합니다.
    예: 'ncc_opt_v1_train.jsonl' 등이 중복 접미사 없이 깔끔하게 생성되도록 보장합니다.
    """
    cleaned = (raw_name or "").strip()
    if not cleaned:
        return "ncc_opt_v1"
    # 확장자나 접미사를 실수로 입력한 경우 제거
    if cleaned.lower().endswith(".jsonl"):
        cleaned = cleaned[:-6].strip()
    for suffix in ("_train", "_val"):
        if cleaned.lower().endswith(suffix) and len(cleaned) > len(suffix):
            cleaned = cleaned[: -len(suffix)].strip()
    # 경로 구분자, 공백 및 금지 문자를 언더스코어로 치환
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", cleaned).strip("_")
    return cleaned or "ncc_opt_v1"


def split_train_val_records(
    records: List[Dict[str, str]],
    val_pct: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    추출된 전체 QA 레코드 리스트를 검증셋 비율(val_pct, 0~50%)에 따라
    학습용(train)과 검증용(val) 리스트로 분할합니다.
    - val_pct == 0 이면 검증셋 분할 없이 전체를 train_records로 반환하고 val_records는 빈 리스트([])를 반환합니다.
    - val_pct > 0 이고 샘플 수가 2개 이상이면 최소 1개 이상의 검증 샘플과 최소 1개 이상의 학습 샘플이 보장되도록 분할합니다.
    """
    total = len(records)
    if total == 0 or val_pct <= 0:
        return list(records), []

    if total == 1:
        # 샘플이 1건뿐인 경우 학습 데이터가 비는 것을 방지하기 위해 학습셋에 배치
        return list(records), []

    val_count = int(round(total * (val_pct / 100.0)))
    val_count = max(1, min(val_count, total - 1))
    train_count = total - val_count

    train_records = list(records[:train_count])
    val_records = list(records[train_count:])
    return train_records, val_records


def write_jsonl_file(filepath: str, records: List[Dict[str, str]]) -> None:
    """
    각 행이 {"instruction": ..., "input": ..., "output": ...} 구조인 표준 JSONL 파일을 UTF-8로 저장합니다.
    """
    ensure_directories()
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            ordered_rec = {
                "instruction": str(rec.get("instruction", "")),
                "input": str(rec.get("input", "")),
                "output": str(rec.get("output", "")),
            }
            f.write(json.dumps(ordered_rec, ensure_ascii=False) + "\n")


def update_dataset_info_json(
    dataset_entries: Dict[str, Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    """
    ./data/dataset_info.json 파일을 읽어(없거나 손상된 경우 {} 생성)
    기존에 등록된 다른 데이터셋 정보들을 온전히 보존한 채 신규 데이터셋 항목을 병합(Update)하여 저장합니다.
    """
    ensure_directories()
    info_path = os.path.join(DATA_DIR, "dataset_info.json")
    existing_data: Dict[str, Any] = {}

    if os.path.exists(info_path):
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    loaded = json.loads(content)
                    if isinstance(loaded, dict):
                        existing_data = loaded
        except Exception:
            existing_data = {}

    for key, val in dataset_entries.items():
        existing_data[key] = val

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return info_path, existing_data


# ==============================================================================
# 2. UI 렌더링 컴포넌트
# ==============================================================================
def render_onboarding_guide() -> None:
    """상단 온보딩 가이드 및 4대 변수 개념 가이드를 렌더링합니다."""
    with st.expander(
        "📘 처음 오셨나요? 전체 작업 순서 및 4대 변수 개념 가이드", expanded=False
    ):
        st.markdown(
            """
            ### 🔄 전체 4단계 워크플로우 요약
            이 도구는 플랜트 **RTDB(실시간 데이터베이스) 이력 CSV 파일**로부터 우수 운전 구간(Golden Case)을 추출하여,
            **LLaMA-Factory 학습용 고품질 한국어 지시문 데이터셋(JSONL)**으로 변환하는 웹 GUI입니다.

            1. **[Step 1] CSV 로드 및 시계열 집계(Resampling)**: RTDB CSV를 불러오고 공정 체류시간에 맞게 초/분 단위 노이즈를 평균·대푯값으로 압축합니다.
            2. **[Step 2] 4대 공정 변수 매핑 및 프리셋 관리**: CSV 태그들을 역할별 4대 변수(`feed`, `control`, `state`, `kpi`)로 분류하고 `.yaml` 프리셋으로 저장/불러옵니다.
            3. **[Step 3] 최적 운전 구간(Before → After) 추출 및 전문가 코멘트 보강**: KPI가 실질적으로 개선된 운전 구간을 자동 탐색하고 운전 노하우(`Expert_Note`, `Action_Reason`)를 덧붙여 실시간 문장을 확인합니다.
            4. **[Step 4] JSONL 데이터셋 분할 생성 및 LLaMA-Factory 자동 등록**: 학습/검증용 `.jsonl` 파일을 생성하고 `./data/dataset_info.json`에 즉시 등록합니다.

            ---

            ### 🧪 왜 공정 태그를 4가지(`feed`, `control`, `state`, `kpi`)로 나누나요?
            수백 개의 RTDB 센서 태그를 구분 없이 한꺼번에 LLM에 넣으면, 모델이 **"단순히 외생으로 주어진 원료 조건"**과 **"운전원이 직접 밸브/설정치(SV)를 조작한 변수"**, 그리고 **"공정 안전상 절대 넘으면 안 되는 제약 조건"**을 혼동하여 비현실적인 운전 가이드를 생성하게 됩니다.
            화학공학 엔지니어의 실제 운전 의사결정 구조와 동일하게 4가지 역할로 분리하면 모델이 **인과관계(원인 → 조작 → 상태 변화 → 성과)**를 정확히 학습합니다.

            | 구분 코드 | 변수 분류 명칭 | 화학공학적 의미 및 쉬운 비유 | 대표 태그 예시 (NCC / 정류 공정) | LLM 프롬프트 내 역할 |
            | :--- | :--- | :--- | :--- | :--- |
            | **① `feed`** | **원료 / 투입 변수**<br>*(Disturbance / Feed)* | **"오늘 들어온 식재료의 양과 상태"**<br>운전원이 임의로 바꾸기 어렵고 상류 공정에서 결정되어 들어오는 유량·조성·온도 조건 | 나프타 투입 유량(`FI-101`), 원료 비중/파라핀 함량, 피드 예열 온도 | 운전 상황의 **전제 조건(Given Context)** 제시 |
            | **② `control`** | **제어 / 조작 변수**<br>*(Manipulated Variable, MV/SV)* | **"셰프가 직접 조절하는 화력과 조리 밸브"**<br>운전원이 수율 개선이나 에너지 절감을 위해 직접 설정값(SV)을 변경하는 조작 변수 | 열분해로 출구온도(`COT`), 희석스팀비(`DSR`), 정류탑 환류비(`Reflux Ratio`), 리보일러 스팀량 | **최적 조작 권고안(Action)** 도출 대상 |
            | **③ `state`** | **상태 / 제약 변수**<br>*(Process State & Constraint)* | **"압력솥의 내부 압력과 냄비 바닥 눌어붙음"**<br>직접 조작하진 않지만 공정 안전·설비 보호를 위해 **상/하한(Limit) 내에 묶어두어야 하는 내부 상태** | 퀀치 보일러 코킹 차압(`PDI-205`, 상한 0.6 bar), 탑 상/하부 차압, 압축기 흡입 압력, 튜브 표면 온도(`TMT`) | 운전 가능 여부를 판단하는 **안전 제약(Safety Boundary)** |
            | **④ `kpi`** | **목표 성과 변수**<br>*(Key Performance Indicator)* | **"완성된 요리의 수율과 가스비 영수증"**<br>운전 최적화를 통해 최종적으로 **극대화(`maximize`)**하거나 **최소화(`minimize`)**하려는 핵심 성과 지표 | 에틸렌 수율(`%`, 극대화), 프로필렌 수율(`%`), 톤당 스팀 원단위(`minimize`), 탑저 에틸렌 손실률 | 우수 운전 구간(Golden Case) **자동 필터링 기준 및 기대 효과** |
            """
        )


def render_step1() -> None:
    """[Step 1] CSV 파일 로더, 공정 기본 정보 입력, 시계열 리샘플링 UI를 렌더링합니다."""
    st.subheader("Step 1. CSV 데이터 로드 및 시계열 집계 (Resampling)")

    # 1) 공정 기본 정보 및 파일 소스 선택
    col_info, col_src = st.columns([1, 1])
    with col_info:
        process_name = st.text_input(
            "🏭 공정 명칭 (예: 나프타 열분해 및 C2 정류 공정)",
            value=st.session_state["process_name"],
            help="프롬프트 지시문과 프리셋 파일명(./presets/{공정명}.yaml)의 기준이 되는 공정 명칭을 입력합니다.",
        )
        st.session_state["process_name"] = process_name

    with col_src:
        source_mode = st.radio(
            "📂 CSV 파일 불러오기 방식",
            options=["① 내 PC에서 CSV 직접 업로드", "② NAS 아카이브 폴더에서 선택"],
            horizontal=True,
        )
        st.session_state["source_mode"] = source_mode

    # 2) 파일 로더 처리
    loaded_df: Optional[pd.DataFrame] = None
    current_file_id: Optional[str] = None

    if source_mode == "① 내 PC에서 CSV 직접 업로드":
        uploaded_file = st.file_uploader(
            "RTDB 이력 CSV 파일을 업로드하세요 (.csv)",
            type=["csv"],
        )
        if uploaded_file is not None:
            current_file_id = f"upload::{uploaded_file.name}::{uploaded_file.size}"
            if st.session_state["loaded_file_id"] != current_file_id:
                try:
                    loaded_df = load_csv_safely(uploaded_file)
                except Exception as exc:
                    st.error(f"CSV 파일 로드 중 오류가 발생했습니다: {exc}")
    else:
        nas_csv_files = sorted(glob.glob(os.path.join(NAS_ARCHIVE_DIR, "*.csv")))
        if not nas_csv_files:
            st.info(
                f"현재 `{NAS_ARCHIVE_DIR}` 폴더 내에 `.csv` 파일이 없습니다. "
                f"해당 폴더에 CSV 파일을 넣거나 '① 내 PC에서 CSV 직접 업로드'를 이용해 주세요."
            )
        else:
            file_options = ["-- 파일을 선택하세요 --"] + [
                os.path.basename(p) for p in nas_csv_files
            ]
            selected_name = st.selectbox(
                f"`{NAS_ARCHIVE_DIR}` 내 CSV 파일 선택",
                options=file_options,
            )
            if selected_name != "-- 파일을 선택하세요 --":
                selected_path = os.path.join(NAS_ARCHIVE_DIR, selected_name)
                mtime = os.path.getmtime(selected_path) if os.path.exists(selected_path) else 0
                current_file_id = f"nas::{selected_path}::{mtime}"
                if st.session_state["loaded_file_id"] != current_file_id:
                    try:
                        loaded_df = load_csv_safely(selected_path)
                    except Exception as exc:
                        st.error(f"NAS CSV 파일 로드 중 오류가 발생했습니다: {exc}")

    # 새 파일이 로드되었을 때 세션 상태 갱신
    if loaded_df is not None and current_file_id is not None:
        st.session_state["raw_df"] = loaded_df
        st.session_state["loaded_file_id"] = current_file_id
        detected_col = detect_time_column(loaded_df)
        st.session_state["time_col"] = detected_col
        # 기본 상태(원본 유지)로 processed_df 초기화
        proc_df, msg = perform_resampling(
            loaded_df,
            detected_col,
            st.session_state["resample_rule"],
            st.session_state["resample_agg"],
        )
        st.session_state["processed_df"] = proc_df
        st.session_state["resample_summary_msg"] = msg

    # 3) 시계열 데이터 포인트 축소 (Resampling / Aggregate) UI
    st.markdown("---")
    header_col, help_col = st.columns([3, 1])
    with header_col:
        st.markdown("#### ⏱️ 시계열 데이터 포인트 축소 (Resampling / Aggregate)")
    with help_col:
        with st.popover("❓ 왜 데이터 포인트를 줄이나요?", use_container_width=True):
            st.markdown(
                """
                **1. 초/분 단위 계측 노이즈(Noise) 제거**
                - RTDB에서 1초~1분 간격으로 수집된 원시 데이터에는 유량계 출렁임, 밸브 헌팅(Hunting), 일시적 센서 노이즈가 다수 포함되어 있습니다.
                - 일정 시간 구간(예: 10분, 30분, 1시간)의 **평균값(`mean`)**이나 **중앙값(`median`)**으로 집계하면 일시적 노이즈가 상쇄되고 실제 공정의 추세(Trend)만 깔끔하게 남습니다.

                **2. 화학공정의 체류시간(Residence Time) 및 열적 평형 반영**
                - 열분해로나 대형 정류탑에서 운전원이 온도(`COT`)나 환류비를 조작하더라도, 그 효과가 탑저 제품 수율(`KPI`)에 온전히 나타나기까지는 수십 분~수 시간의 **체류시간 및 응답 지연(Dead Time)**이 존재합니다.
                - 공정 특성에 맞는 시간 단위로 데이터를 압축해야 **'운전 조작(Before) → 공정 안정화 및 성과 개선(After)'**의 인과관계를 왜곡 없이 추출할 수 있습니다.

                **3. 학습 데이터 중복 방지 및 연산 속도 향상**
                - 수십만 행의 원시 데이터를 그대로 비교하면 거의 동일한 상태의 중복 샘플이 대량 생성됩니다. 적절한 주기로 축소하면 대표성 높은 핵심 골든 케이스만 빠르게 선별할 수 있습니다.
                """
            )

    raw_df = st.session_state.get("raw_df")
    if raw_df is None:
        st.warning("👆 먼저 위에서 분석할 RTDB CSV 파일을 업로드하거나 NAS 폴더에서 선택해 주세요.")
        return

    all_cols = list(raw_df.columns)
    default_time_col = st.session_state.get("time_col")
    default_time_idx = (
        all_cols.index(default_time_col)
        if default_time_col in all_cols
        else 0
    )

    r_col1, r_col2, r_col3, r_col4 = st.columns([1.3, 1.2, 1.2, 0.8])
    with r_col1:
        selected_time_col = st.selectbox(
            "🕒 시간 기준 컬럼 (Timestamp/DateTime)",
            options=all_cols,
            index=default_time_idx,
            help="시간 기준 컬럼을 자동으로 감지합니다. 다른 컬럼이 시간 컬럼이라면 드롭다운에서 직접 선택하세요.",
        )

    rule_options = [
        "원본 유지(집계 안 함)",
        "1분(1min)",
        "10분(10min)",
        "30분(30min)",
        "1시간(1h)",
        "4시간(4h)",
    ]
    current_rule = st.session_state.get("resample_rule", "원본 유지(집계 안 함)")
    rule_idx = rule_options.index(current_rule) if current_rule in rule_options else 0

    with r_col2:
        selected_rule = st.selectbox(
            "📏 집계 주기 선택",
            options=rule_options,
            index=rule_idx,
        )

    agg_options = [
        "평균값(mean)",
        "마지막 값(last)",
        "중앙값(median)",
    ]
    current_agg = st.session_state.get("resample_agg", "평균값(mean)")
    agg_idx = agg_options.index(current_agg) if current_agg in agg_options else 0

    with r_col3:
        selected_agg = st.selectbox(
            "🧮 집계 함수 선택 (문자열은 last 자동 적용)",
            options=agg_options,
            index=agg_idx,
        )

    with r_col4:
        st.write("")
        st.write("")
        apply_clicked = st.button(
            "🔄 집계 적용",
            type="primary",
            use_container_width=True,
        )

    # 사용자가 옵션을 변경했거나 [집계 적용] 버튼을 눌렀을 때 즉시 반영
    options_changed = (
        selected_time_col != st.session_state.get("time_col")
        or selected_rule != st.session_state.get("resample_rule")
        or selected_agg != st.session_state.get("resample_agg")
    )

    if apply_clicked or options_changed or st.session_state.get("processed_df") is None:
        st.session_state["time_col"] = selected_time_col
        st.session_state["resample_rule"] = selected_rule
        st.session_state["resample_agg"] = selected_agg
        try:
            proc_df, summary_msg = perform_resampling(
                raw_df,
                selected_time_col,
                selected_rule,
                selected_agg,
            )
            st.session_state["processed_df"] = proc_df
            st.session_state["resample_summary_msg"] = summary_msg
        except Exception as exc:
            st.error(f"시계열 집계 중 오류가 발생했습니다: {exc}")

    # 결과 요약 메시지 및 상위 5행 미리보기 출력
    if st.session_state.get("resample_summary_msg"):
        st.success(f"✅ **{st.session_state['resample_summary_msg']}**")

    proc_df = st.session_state.get("processed_df")
    if proc_df is not None:
        st.markdown("##### 👀 집계 결과 데이터 상위 5행 미리보기")
        st.dataframe(proc_df.head(5), use_container_width=True)


def render_step2() -> None:
    """[Step 2] 4대 공정 변수 매핑 UI 및 YAML 프리셋 저장/불러오기 기능을 렌더링합니다."""
    st.subheader("Step 2. 4대 공정 변수 매핑 및 YAML 프리셋 관리")
    st.caption(
        "공정 태그(CSV 컬럼)를 역할에 따라 4가지 변수(`feed`, `control`, `state`, `kpi`)로 분류하고 한글 명칭·단위·관리 기준을 설정합니다. "
        "작성한 설정은 `./presets/{공정명}.yaml` 프리셋으로 저장하여 언제든 재사용할 수 있습니다."
    )

    # 직전 액션(저장/불러오기 등)의 플래시 알림 메시지가 있으면 상단에 표시 후 초기화
    flash_msg = st.session_state.get("preset_flash_msg")
    if flash_msg:
        flash_type = st.session_state.get("preset_flash_type", "success")
        if flash_type == "error":
            st.error(flash_msg)
        elif flash_type == "warning":
            st.warning(flash_msg)
        else:
            st.success(flash_msg)
        st.session_state["preset_flash_msg"] = None

    # --------------------------------------------------------------------------
    # 1) 프리셋 저장 / 불러오기 및 빠른 도우미 툴바
    # --------------------------------------------------------------------------
    current_process_name = st.session_state.get("process_name", "나프타 열분해 및 C2 정류 공정")
    target_filename = sanitize_preset_filename(current_process_name)
    target_rel_path = os.path.join(PRESETS_DIR, target_filename)

    preset_files = list_preset_files()
    preset_options = ["-- 프리셋 파일 선택 --"] + preset_files

    selected_preset = st.session_state.get("selected_preset_file")
    default_preset_idx = (
        preset_options.index(selected_preset)
        if selected_preset in preset_options
        else 0
    )

    with st.container(border=True):
        st.markdown("##### 🗂️ 공정 변수 프리셋 저장 / 불러오기 (`./presets`)")
        p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns([1.6, 1.0, 1.6, 1.0, 1.1])

        with p_col1:
            st.text_input(
                "💾 저장될 프리셋 경로 (Step 1 공정 명칭 기준)",
                value=target_rel_path,
                disabled=True,
                help="Step 1의 '공정 명칭' 입력값에 따라 `./presets/{공정명}.yaml` 파일명이 자동 지정됩니다.",
            )

        with p_col2:
            st.write("")
            st.write("")
            save_preset_clicked = st.button(
                "💾 프리셋 저장",
                type="primary",
                use_container_width=True,
                help=f"현재 아래 표에 입력된 4대 변수 매핑 정보를 `{target_rel_path}` 파일로 저장합니다.",
            )

        with p_col3:
            chosen_preset_file = st.selectbox(
                "📂 `./presets` 내 저장된 프리셋 선택",
                options=preset_options,
                index=default_preset_idx,
            )

        with p_col4:
            st.write("")
            st.write("")
            load_preset_clicked = st.button(
                "📂 기존 프리셋 불러오기",
                use_container_width=True,
                disabled=(chosen_preset_file == "-- 프리셋 파일 선택 --"),
                help="선택한 YAML 프리셋 파일을 읽어 아래 4대 변수 매핑 표에 즉시 반영합니다.",
            )

        with p_col5:
            st.write("")
            st.write("")
            btn_sub1, btn_sub2 = st.columns(2)
            with btn_sub1:
                fill_sample_clicked = st.button(
                    "✨ 예시 채우기",
                    use_container_width=True,
                    help="NCC 및 C2 정류 공정의 대표 태그(COT, DSR, 코킹 차압, 에틸렌 수율 등) 예시를 표에 자동으로 채웁니다.",
                )
            with btn_sub2:
                clear_all_clicked = st.button(
                    "🧹 초기화",
                    use_container_width=True,
                    help="4대 변수 매핑 표의 모든 행을 비웁니다.",
                )

    # [기존 프리셋 불러오기] 처리 (st.data_editor 렌더링 전에 수행하여 즉시 반영)
    if load_preset_clicked and chosen_preset_file != "-- 프리셋 파일 선택 --":
        try:
            loaded_proc_name, loaded_mapping = load_preset_from_yaml(chosen_preset_file)
            st.session_state["var_mapping"] = loaded_mapping
            st.session_state["selected_preset_file"] = chosen_preset_file
            if loaded_proc_name:
                st.session_state["process_name"] = loaded_proc_name
            sync_editor_base_dfs_from_mapping()
            counts_str = ", ".join(
                f"{k}: {len(loaded_mapping.get(k, []))}개" for k in CATEGORY_KEYS
            )
            st.session_state["preset_flash_type"] = "success"
            st.session_state["preset_flash_msg"] = (
                f"✅ 프리셋 `{os.path.join(PRESETS_DIR, chosen_preset_file)}`을(를) 성공적으로 불러왔습니다! ({counts_str})"
            )
            st.rerun()
        except Exception as exc:
            st.error(f"프리셋 불러오기 중 오류가 발생했습니다: {exc}")

    # [예시 채우기] 처리
    if fill_sample_clicked:
        st.session_state["var_mapping"] = get_sample_ncc_mapping()
        sync_editor_base_dfs_from_mapping()
        st.session_state["preset_flash_type"] = "success"
        st.session_state["preset_flash_msg"] = (
            "✅ NCC 및 C2 정류 공정 대표 예시 변수(나프타 유량, COT, DSR, 코킹 차압, 에틸렌 수율 등)가 표에 입력되었습니다."
        )
        st.rerun()

    # [초기화] 처리
    if clear_all_clicked:
        st.session_state["var_mapping"] = {cat: [] for cat in CATEGORY_KEYS}
        sync_editor_base_dfs_from_mapping()
        st.session_state["preset_flash_type"] = "warning"
        st.session_state["preset_flash_msg"] = "🧹 모든 변수 매핑 표가 초기화되었습니다."
        st.rerun()

    # --------------------------------------------------------------------------
    # 2) CSV 로드 시: 컬럼 빠른 분류 도우미 (선택 사항)
    # --------------------------------------------------------------------------
    proc_df = st.session_state.get("processed_df")
    time_col = st.session_state.get("time_col")
    candidate_csv_cols: List[str] = []

    if proc_df is not None and not proc_df.empty:
        candidate_csv_cols = [str(c) for c in proc_df.columns if c != time_col]
        mapped_tags = {
            item.get("tag", "").strip()
            for cat in CATEGORY_KEYS
            for item in st.session_state["var_mapping"].get(cat, [])
            if item.get("tag", "").strip()
        }
        unmapped_cols = [c for c in candidate_csv_cols if c not in mapped_tags]

        with st.expander(
            f"⚡ CSV 컬럼 빠른 분류 도우미 (전체 후보 컬럼 {len(candidate_csv_cols)}개 중 미분류 {len(unmapped_cols)}개)",
            expanded=False,
        ):
            st.caption(
                "현재 로드된 CSV 파일의 컬럼을 선택하여 원하는 변수 카테고리 표에 한 번에 추가할 수 있습니다. "
                "(물론 아래 표에서 직접 `+` 버튼을 눌러 태그명을 타이핑하셔도 됩니다.)"
            )
            q_col1, q_col2, q_col3 = st.columns([2.2, 1.2, 0.8])
            with q_col1:
                cols_to_add = st.multiselect(
                    "추가할 CSV 컬럼 선택",
                    options=candidate_csv_cols,
                    default=[],
                    placeholder="CSV 컬럼을 하나 이상 선택하세요...",
                )
            with q_col2:
                target_cat_label = st.selectbox(
                    "할당할 변수 카테고리",
                    options=[
                        "① feed (원료/투입)",
                        "② control (제어/조작)",
                        "③ state (상태/제약)",
                        "④ kpi (목표 성과)",
                    ],
                )
            with q_col3:
                st.write("")
                st.write("")
                quick_add_clicked = st.button(
                    "➕ 표에 추가",
                    use_container_width=True,
                    disabled=(len(cols_to_add) == 0),
                )

            if quick_add_clicked and cols_to_add:
                cat_map = {
                    "① feed (원료/투입)": "feed",
                    "② control (제어/조작)": "control",
                    "③ state (상태/제약)": "state",
                    "④ kpi (목표 성과)": "kpi",
                }
                target_cat = cat_map[target_cat_label]
                existing_items = list(st.session_state["var_mapping"].get(target_cat, []))
                existing_tags = {item.get("tag", "") for item in existing_items}

                added_count = 0
                for col_name in cols_to_add:
                    if col_name not in existing_tags:
                        new_entry: Dict[str, Any] = {
                            "tag": col_name,
                            "name": col_name,
                            "unit": "",
                        }
                        if target_cat == "state":
                            new_entry["limit_desc"] = ""
                        elif target_cat == "kpi":
                            new_entry["direction"] = "maximize"
                            new_entry["threshold"] = 0.1
                        existing_items.append(new_entry)
                        added_count += 1

                st.session_state["var_mapping"][target_cat] = existing_items
                sync_editor_base_dfs_from_mapping()
                st.session_state["preset_flash_type"] = "success"
                st.session_state["preset_flash_msg"] = (
                    f"✅ 선택한 컬럼 {added_count}개를 `{target_cat}` 카테고리에 추가했습니다. 한글 명칭과 단위를 입력해 주세요."
                )
                st.rerun()

    # --------------------------------------------------------------------------
    # 3) 4대 변수 카테고리별 도움말 팝오버(st.popover) 및 설정 표(st.data_editor)
    # --------------------------------------------------------------------------
    editor_version = st.session_state.get("mapping_editor_version", 0)
    base_dfs = st.session_state["mapping_base_dfs"]

    row1_col1, row1_col2 = st.columns(2)

    # ---- ① 원료/투입 변수 (feed) ----
    with row1_col1:
        with st.container(border=True):
            h_col, p_col = st.columns([1.8, 1.2])
            with h_col:
                st.markdown("#### ① 원료/투입 변수 (`feed`)")
            with p_col:
                with st.popover("❓ 이 변수에는 무엇을 넣나요?", use_container_width=True):
                    st.markdown(
                        """
                        **① 원료/투입 변수 (`feed` - Feed & Disturbance)**
                        - 운전원이 직접 밸브를 조작해 바꾸기 어렵고, 상류 공정이나 원료 탱크에서 결정되어 들어오는 **외생 투입 조건**입니다.
                        - LLM 프롬프트에서는 **"현재 주어진 운전 전제 조건(Given Context)"**으로 서술됩니다.

                        **📌 구체적 입력 예시**
                        | 태그명 (`tag`) | 한글 명칭 (`name`) | 단위 (`unit`) |
                        | :--- | :--- | :--- |
                        | `FI-101` | 나프타 유량 | `t/h` |
                        | `TI-102` | 원료 예열기 출구온도 | `°C` |
                        | `AI-105` | 원료 파라핀 함량 | `wt%` |
                        | `FI-301` | C2 정류탑 피드 유량 | `t/h` |
                        """
                    )
            st.caption("투입 원료 유량·온도·조성 등 운전 전제 조건이 되는 태그를 입력하세요.")
            edited_feed_df = st.data_editor(
                base_dfs["feed"],
                key=f"editor_feed_{editor_version}",
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                column_config={
                    "tag": st.column_config.TextColumn(
                        "태그명 (tag)",
                        help="CSV 컬럼명 또는 RTDB 태그명 (예: FI-101)",
                        required=True,
                    ),
                    "name": st.column_config.TextColumn(
                        "한글 명칭 (name)",
                        help="엔지니어가 이해하기 쉬운 한글 변수명 (예: 나프타 유량)",
                        required=True,
                    ),
                    "unit": st.column_config.TextColumn(
                        "단위 (unit)",
                        help="공정 공학 단위 (예: t/h, kg/h, °C, wt%)",
                    ),
                },
            )

    # ---- ② 제어/조작 변수 (control) ----
    with row1_col2:
        with st.container(border=True):
            h_col, p_col = st.columns([1.8, 1.2])
            with h_col:
                st.markdown("#### ② 제어/조작 변수 (`control`)")
            with p_col:
                with st.popover("❓ 이 변수에는 무엇을 넣나요?", use_container_width=True):
                    st.markdown(
                        """
                        **② 제어/조작 변수 (`control` - Manipulated Variable, MV/SV)**
                        - 운전원이 수율 개선, 품질 확보, 에너지 절감을 위해 **직접 설정치(SV)나 밸브를 조작하는 변수**입니다.
                        - Step 3에서 **Before → After 조작 변화량**을 계산하여 LLM이 **"구체적으로 어떤 변수를 얼마나 조정해야 하는지"** 답변을 생성하는 핵심 항목입니다.

                        **📌 구체적 입력 예시**
                        | 태그명 (`tag`) | 한글 명칭 (`name`) | 단위 (`unit`) |
                        | :--- | :--- | :--- |
                        | `COT` | 열분해로 출구온도 | `°C` |
                        | `DSR` | 희석스팀비 | `kg/kg` |
                        | `FIC-305` | 정류탑 환류 유량 (Reflux) | `t/h` |
                        | `FIC-310` | 리보일러 스팀 투입량 | `t/h` |
                        """
                    )
            st.caption("COT, DSR, 환류비 등 운전원이 직접 설정값을 조절하는 조작 변수를 입력하세요.")
            edited_control_df = st.data_editor(
                base_dfs["control"],
                key=f"editor_control_{editor_version}",
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                column_config={
                    "tag": st.column_config.TextColumn(
                        "태그명 (tag)",
                        help="CSV 컬럼명 또는 RTDB 태그명 (예: COT, DSR)",
                        required=True,
                    ),
                    "name": st.column_config.TextColumn(
                        "한글 명칭 (name)",
                        help="한글 변수명 (예: 열분해로 출구온도, 희석스팀비)",
                        required=True,
                    ),
                    "unit": st.column_config.TextColumn(
                        "단위 (unit)",
                        help="공정 공학 단위 (예: °C, kg/kg, t/h)",
                    ),
                },
            )

    row2_col1, row2_col2 = st.columns(2)

    # ---- ③ 상태/제약 변수 (state) ----
    with row2_col1:
        with st.container(border=True):
            h_col, p_col = st.columns([1.8, 1.2])
            with h_col:
                st.markdown("#### ③ 상태/제약 변수 (`state`)")
            with p_col:
                with st.popover("❓ 이 변수에는 무엇을 넣나요?", use_container_width=True):
                    st.markdown(
                        """
                        **③ 상태/제약 변수 (`state` - Process State & Safety Constraint)**
                        - 직접 조작하진 않지만 운전 조작에 따라 변하며, **설비 보호·코킹 방지·플러딩(Flooding) 방지를 위해 상/하한 관리 기준(`limit_desc`) 내에 묶어두어야 하는 공정 상태 변수**입니다.
                        - `관리 기준 설명(limit_desc)`을 적어주시면 LLM이 안전 제약을 침범하지 않는 범위 내에서 최적 조작을 권고하도록 학습됩니다.

                        **📌 구체적 입력 예시**
                        | 태그명 (`tag`) | 한글 명칭 (`name`) | 단위 (`unit`) | 관리 기준 설명 (`limit_desc`) |
                        | :--- | :--- | :--- | :--- |
                        | `PDI-205` | 코킹 차압 | `bar` | `상한 0.6 bar` |
                        | `TMT-210` | 튜브 표면 온도(TMT) | `°C` | `상한 1080 °C` |
                        | `PDI-302` | 정류탑 상/하부 차압 | `mmH2O` | `상한 450 mmH2O` |
                        | `PI-401` | 압축기 흡입 압력 | `kg/cm²G` | `하한 0.15 kg/cm²G` |
                        """
                    )
            st.caption("코킹 차압, 튜브 표면 온도 등 운전 안전상 상/하한 관리 기준이 있는 상태 변수를 입력하세요.")
            edited_state_df = st.data_editor(
                base_dfs["state"],
                key=f"editor_state_{editor_version}",
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                column_config={
                    "tag": st.column_config.TextColumn(
                        "태그명 (tag)",
                        help="CSV 컬럼명 또는 RTDB 태그명 (예: PDI-205)",
                        required=True,
                    ),
                    "name": st.column_config.TextColumn(
                        "한글 명칭 (name)",
                        help="한글 변수명 (예: 코킹 차압)",
                        required=True,
                    ),
                    "unit": st.column_config.TextColumn(
                        "단위 (unit)",
                        help="공정 공학 단위 (예: bar, °C, mmH2O)",
                    ),
                    "limit_desc": st.column_config.TextColumn(
                        "관리 기준 설명 (limit_desc)",
                        help="운전 상/하한 제약 조건 설명 (예: 상한 0.6 bar)",
                    ),
                },
            )

    # ---- ④ 목표 성과 변수 (kpi) ----
    with row2_col2:
        with st.container(border=True):
            h_col, p_col = st.columns([1.8, 1.2])
            with h_col:
                st.markdown("#### ④ 목표 성과 변수 (`kpi`)")
            with p_col:
                with st.popover("❓ 이 변수에는 무엇을 넣나요?", use_container_width=True):
                    st.markdown(
                        """
                        **④ 목표 성과 변수 (`kpi` - Key Performance Indicator)**
                        - 운전 최적화를 통해 최종적으로 달성하고자 하는 **핵심 성과 지표(수율 극대화, 에너지 원단위 최소화, 손실 최소화 등)**입니다.
                        - **목표 방향 (`direction`)**: 수치가 커질수록 좋으면 `maximize`, 작아질수록 좋으면 `minimize`를 선택하세요.
                        - **개선 추출 기준값 (`threshold`)**: Step 3에서 `Before → After` 구간을 비교할 때, 최소 이 기준값 이상 개선된 우수 운전 구간(Golden Case)만 자동으로 골라냅니다.

                        **📌 구체적 입력 예시**
                        | 태그명 (`tag`) | 한글 명칭 (`name`) | 단위 (`unit`) | 목표 방향 (`direction`) | 개선 추출 기준값 (`threshold`) |
                        | :--- | :--- | :--- | :--- | :--- |
                        | `C2_YIELD` | 에틸렌 수율 | `%` | `maximize` | `0.2` *(+0.2%p 이상 상승 시 추출)* |
                        | `C3_YIELD` | 프로필렌 수율 | `%` | `maximize` | `0.15` *(+0.15%p 이상 상승 시 추출)* |
                        | `STEAM_UNIT` | 톤당 스팀 원단위 | `GJ/t` | `minimize` | `0.05` *(-0.05 GJ/t 이상 절감 시 추출)* |
                        | `C2_LOSS` | 탑저 에틸렌 손실 | `ppm` | `minimize` | `10.0` *(-10 ppm 이상 감소 시 추출)* |
                        """
                    )
            st.caption("에틸렌 수율, 스팀 원단위 등 극대화(maximize)/최소화(minimize)할 목표 지표를 입력하세요.")
            edited_kpi_df = st.data_editor(
                base_dfs["kpi"],
                key=f"editor_kpi_{editor_version}",
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                column_config={
                    "tag": st.column_config.TextColumn(
                        "태그명 (tag)",
                        help="CSV 컬럼명 또는 RTDB 태그명 (예: C2_YIELD)",
                        required=True,
                    ),
                    "name": st.column_config.TextColumn(
                        "한글 명칭 (name)",
                        help="한글 변수명 (예: 에틸렌 수율)",
                        required=True,
                    ),
                    "unit": st.column_config.TextColumn(
                        "단위 (unit)",
                        help="공정 공학 단위 (예: %, GJ/t, ppm)",
                    ),
                    "direction": st.column_config.SelectboxColumn(
                        "목표 방향 (direction)",
                        help="높을수록 좋으면 maximize, 낮을수록 좋으면 minimize 선택",
                        options=["maximize", "minimize"],
                        default="maximize",
                        required=True,
                    ),
                    "threshold": st.column_config.NumberColumn(
                        "개선 추출 기준값 (threshold)",
                        help="Step 3에서 골든 케이스로 추출할 최소 개선 폭 (예: 0.2)",
                        min_value=0.0,
                        step=0.05,
                        format="%.4g",
                        default=0.1,
                        required=True,
                    ),
                },
            )

    # --------------------------------------------------------------------------
    # 4) 편집 결과를 st.session_state["var_mapping"]에 실시간 동기화
    # --------------------------------------------------------------------------
    current_mapping = {
        "feed": df_to_category_items(edited_feed_df, "feed"),
        "control": df_to_category_items(edited_control_df, "control"),
        "state": df_to_category_items(edited_state_df, "state"),
        "kpi": df_to_category_items(edited_kpi_df, "kpi"),
    }
    st.session_state["var_mapping"] = current_mapping

    # [프리셋 저장] 버튼이 클릭된 경우, 최신 current_mapping을 파일로 저장 후 즉시 리런하여 드롭다운 갱신
    if save_preset_clicked:
        total_vars = sum(len(v) for v in current_mapping.values())
        if total_vars == 0:
            st.warning("⚠️ 저장할 변수 매핑 정보가 없습니다. 먼저 표에 하나 이상의 변수를 입력하거나 [✨ 예시 채우기]를 눌러주세요.")
        else:
            try:
                saved_path = save_preset_to_yaml(current_process_name, current_mapping)
                saved_filename = os.path.basename(saved_path)
                st.session_state["selected_preset_file"] = saved_filename
                sync_editor_base_dfs_from_mapping()
                counts_str = ", ".join(
                    f"{k}: {len(current_mapping.get(k, []))}개" for k in CATEGORY_KEYS
                )
                st.session_state["preset_flash_type"] = "success"
                st.session_state["preset_flash_msg"] = (
                    f"💾 프리셋이 `{saved_path}`에 성공적으로 저장되었습니다! ({counts_str})"
                )
                st.rerun()
            except Exception as exc:
                st.error(f"프리셋 저장 중 오류가 발생했습니다: {exc}")

    # --------------------------------------------------------------------------
    # 5) 매핑 현황 요약 및 CSV 컬럼 일치 여부 안내
    # --------------------------------------------------------------------------
    feed_cnt = len(current_mapping["feed"])
    ctrl_cnt = len(current_mapping["control"])
    state_cnt = len(current_mapping["state"])
    kpi_cnt = len(current_mapping["kpi"])
    total_cnt = feed_cnt + ctrl_cnt + state_cnt + kpi_cnt

    if total_cnt > 0:
        st.info(
            f"📊 **현재 매핑된 공정 변수 요약** — 총 **{total_cnt}개** "
            f"(① `feed` **{feed_cnt}개** | ② `control` **{ctrl_cnt}개** | "
            f"③ `state` **{state_cnt}개** | ④ `kpi` **{kpi_cnt}개**)"
        )
        if proc_df is not None and not proc_df.empty:
            csv_col_set = {str(c) for c in proc_df.columns}
            missing_tags = [
                f"{item['tag']} ({cat})"
                for cat in CATEGORY_KEYS
                for item in current_mapping[cat]
                if item.get("tag") and item["tag"] not in csv_col_set
            ]
            if missing_tags:
                st.warning(
                    "⚠️ 현재 매핑된 태그 중 로드된 CSV 파일의 컬럼명과 일치하지 않는 태그가 있습니다: "
                    + ", ".join(f"`{t}`" for t in missing_tags)
                )


def render_step3() -> None:
    """
    [Step 3] 최적 운전 구간(Before -> After) 추출, 도메인 전문가 보충 멘트 편집기
    및 규칙 기반 한국어 문장(instruction, input, output) 실시간 미리보기 UI를 렌더링합니다.
    """
    st.subheader("Step 3. 최적 운전 구간(Before → After) 추출 및 실시간 문장 미리보기")
    st.caption(
        "Step 2에서 설정한 모든 KPI의 개선 기준(`threshold`)을 만족하는 우수 운전 구간(Golden Case)을 자동 추출하고, "
        "도메인 전문가의 진단(`Expert_Note`)과 조작 근거(`Action_Reason`)를 보강하여 학습용 문장을 실시간으로 검증합니다."
    )

    # 플래시 메시지 출력
    flash_msg = st.session_state.get("step3_flash_msg")
    if flash_msg:
        flash_type = st.session_state.get("step3_flash_type", "success")
        if flash_type == "error":
            st.error(flash_msg)
        elif flash_type == "warning":
            st.warning(flash_msg)
        else:
            st.success(flash_msg)
        st.session_state["step3_flash_msg"] = None

    # --------------------------------------------------------------------------
    # 1) Lookahead 설정 영역 및 [❓ 비교 시점 간격(Lookahead)이란?] 도움말 팝오버
    # --------------------------------------------------------------------------
    proc_df: Optional[pd.DataFrame] = st.session_state.get("processed_df")
    var_mapping: Dict[str, List[Dict[str, Any]]] = st.session_state.get("var_mapping", {})
    time_col: Optional[str] = st.session_state.get("time_col")
    resample_rule: str = st.session_state.get("resample_rule", "원본 유지(집계 안 함)")
    process_name: str = st.session_state.get("process_name", "나프타 열분해 및 C2 정류 공정")

    max_lookahead = max(1, (len(proc_df) - 1) if (proc_df is not None and len(proc_df) > 1) else 100)
    current_lookahead = min(
        max(1, int(st.session_state.get("lookahead_rows", 1))),
        max_lookahead,
    )

    with st.container(border=True):
        top_h_col, top_help_col = st.columns([3, 1])
        with top_h_col:
            st.markdown("##### ⏭️ 비교 시점 간격 (`lookahead_rows`) 설정 및 골든 케이스 자동 탐색")
        with top_help_col:
            with st.popover("❓ 비교 시점 간격(Lookahead)이란?", use_container_width=True):
                st.markdown(
                    """
                    **⏱️ 비교 시점 간격 (`lookahead_rows`)의 화학공학적 의미**
                    - 운전원이 제어 변수(`control`, 예: 분해로 출구온도 `COT`, 정류탑 환류비 등)를 조작한 시점(**Before 행, $t$**) 이후, 공정이 열적·물질적 평형에 도달하여 목표 성과(`kpi`, 예: 에틸렌 수율, 스팀 원단위)가 실제로 개선되기까지는 **공정 체류시간(Residence Time) 및 응답 지연(Dead Time)**이 소요됩니다.
                    - `lookahead_rows`는 **집계된 데이터(Step 1 결과) 기준으로 몇 행 뒤($t + \\text{lookahead\\_rows}$)의 결과를 After 시점으로 삼아 비교할지** 결정하는 값입니다.

                    **📌 설정 예시**
                    - Step 1 집계 주기가 **`10분(10min)`**이고 공정 응답 시간이 약 **30분**이라면 👉 `lookahead_rows = 3` (3행 뒤 = 30분 후 비교)
                    - Step 1 집계 주기가 **`1시간(1h)`**이고 **1시간 뒤** 안정화 상태를 비교한다면 👉 `lookahead_rows = 1` (1행 뒤 = 1시간 후 비교)
                    """
                )

        l_col1, l_col2, l_col3 = st.columns([1.3, 1.9, 1.0])
        with l_col1:
            lookahead_val = st.number_input(
                "비교 시점 간격 (lookahead_rows, 집계된 행 기준)",
                min_value=1,
                max_value=max_lookahead,
                value=current_lookahead,
                step=1,
                help="Before 행(i)과 비교할 After 행(i + lookahead_rows) 사이의 행 간격을 입력합니다.",
            )
            st.session_state["lookahead_rows"] = int(lookahead_val)

        with l_col2:
            interval_desc = _describe_lookahead_interval(int(lookahead_val), resample_rule)
            kpi_list = var_mapping.get("kpi", [])
            if kpi_list:
                kpi_cond_badges = []
                for k in kpi_list:
                    k_label = _display_var_label(k)
                    k_dir = "▲극대화" if k.get("direction") == "maximize" else "▼최소화"
                    k_thr = _format_num(k.get("threshold", 0.0), 4)
                    k_unit = (k.get("unit") or "").strip()
                    kpi_cond_badges.append(f"`{k_label}` ({k_dir} ≥ {k_thr}{(' ' + k_unit) if k_unit else ''})")
                kpi_cond_text = " **AND** ".join(kpi_cond_badges)
            else:
                kpi_cond_text = "*(Step 2에서 설정된 KPI 없음)*"

            st.info(
                f"🕒 **비교 시점**: Before($t$) 대비 **{interval_desc}** After($t+{int(lookahead_val)}$) 비교\n\n"
                f"🎯 **동시 만족 필터 조건**: {kpi_cond_text}"
            )

        with l_col3:
            st.write("")
            extract_clicked = st.button(
                "🔍 개선 구간 재탐색",
                type="primary",
                use_container_width=True,
                help="현재 설정된 lookahead_rows와 모든 KPI threshold 조건을 만족하는 최적 운전 구간을 다시 탐색합니다.",
            )
            sample_load_clicked = st.button(
                "🧪 샘플 데이터로 즉시 체험",
                use_container_width=True,
                help="외부 CSV 파일이 없어도 NCC 공정 샘플 시계열 데이터(16행)와 4대 변수 매핑을 한 번에 불러와 Step 3를 즉시 테스트합니다.",
            )

    # [샘플 데이터로 즉시 체험] 버튼 처리
    if sample_load_clicked:
        sample_df = create_sample_ncc_dataframe()
        st.session_state["raw_df"] = sample_df
        st.session_state["loaded_file_id"] = "builtin_sample_ncc_csv"
        st.session_state["time_col"] = "Timestamp"
        st.session_state["resample_rule"] = "원본 유지(집계 안 함)"
        st.session_state["resample_agg"] = "평균값(mean)"
        proc_sample, msg = perform_resampling(
            sample_df, "Timestamp", "원본 유지(집계 안 함)", "평균값(mean)"
        )
        st.session_state["processed_df"] = proc_sample
        st.session_state["resample_summary_msg"] = msg
        st.session_state["var_mapping"] = get_sample_ncc_mapping()
        sync_editor_base_dfs_from_mapping()
        st.session_state["lookahead_rows"] = 1
        st.session_state["golden_extraction_sig"] = None
        st.session_state["step3_flash_type"] = "success"
        st.session_state["step3_flash_msg"] = (
            "✅ 테스트용 NCC 운전 이력 데이터(16행)와 4대 공정 변수 매핑이 로드되어 최적 운전 구간이 추출되었습니다!"
        )
        st.rerun()

    # 사전 조건 검증 (CSV 데이터 및 KPI 매핑 유무)
    if proc_df is None or proc_df.empty:
        st.warning(
            "👆 먼저 **Step 1**에서 CSV 파일을 업로드하거나, 우측 상단의 **[🧪 샘플 데이터로 즉시 체험]** 버튼을 눌러주세요."
        )
        return

    valid_kpis = [
        item
        for item in var_mapping.get("kpi", [])
        if (item.get("tag") or "").strip() in proc_df.columns
    ]
    if not valid_kpis:
        st.warning(
            "⚠️ **Step 2**의 **④ 목표 성과 변수 (`kpi`)** 표에 현재 CSV 컬럼과 일치하는 KPI 태그가 최소 1개 이상 설정되어야 개선 구간을 탐색할 수 있습니다."
        )
        return

    # --------------------------------------------------------------------------
    # 2) 모든 KPI threshold 만족 구간 자동 탐색 및 동기화
    # --------------------------------------------------------------------------
    current_sig = json.dumps(
        {
            "file_id": st.session_state.get("loaded_file_id"),
            "rows": len(proc_df),
            "resample_rule": resample_rule,
            "resample_agg": st.session_state.get("resample_agg"),
            "time_col": str(time_col),
            "lookahead": int(lookahead_val),
            "kpi": var_mapping.get("kpi", []),
            "control_tags": [x.get("tag") for x in var_mapping.get("control", [])],
            "state_tags": [x.get("tag") for x in var_mapping.get("state", [])],
        },
        ensure_ascii=False,
         sort_keys=True,
    )

    need_reextract = (
        extract_clicked
        or st.session_state.get("golden_base_df") is None
        or st.session_state.get("golden_extraction_sig") != current_sig
    )

    if need_reextract:
        # 기존 편집 코멘트 캐시 보존
        update_notes_cache_from_df(st.session_state.get("golden_cases_df"))
        extracted_df, diag_msgs = extract_golden_cases(
            proc_df=proc_df,
            var_mapping=var_mapping,
            time_col=time_col,
            lookahead_rows=int(lookahead_val),
            notes_cache=st.session_state.get("golden_notes_cache", {}),
        )
        st.session_state["golden_extraction_sig"] = current_sig
        st.session_state["golden_diag_msgs"] = diag_msgs
        sync_golden_editor_base_df(extracted_df)

    golden_base_df: Optional[pd.DataFrame] = st.session_state.get("golden_base_df")
    diag_msgs: List[str] = st.session_state.get("golden_diag_msgs", [])

    if golden_base_df is None or golden_base_df.empty:
        st.warning(
            f"⚠️ 설정한 비교 간격(`lookahead_rows = {int(lookahead_val)}`)과 모든 KPI의 `threshold` 조건을 동시에 만족하는 운전 구간(Before → After)이 없습니다."
        )
        if diag_msgs:
            with st.container(border=True):
                st.markdown("**💡 데이터 내 KPI 최대 변화폭 진단 (Step 2 `threshold` 조정 참고용)**")
                for d_msg in diag_msgs:
                    st.markdown(d_msg)
        return

    # --------------------------------------------------------------------------
    # 3) 추출된 골든 케이스 편집 표 (st.data_editor) 및 도메인 전문가 보충 멘트 입력
    # --------------------------------------------------------------------------
    st.markdown("---")
    tb_col1, tb_col2, tb_col3, tb_col4 = st.columns([2.2, 1.0, 0.9, 0.9])
    with tb_col1:
        st.markdown(
            f"#### 📝 추출된 최적 운전 구간(Golden Cases) 및 전문가 코멘트 편집기 (총 `{len(golden_base_df)}`건)"
        )
    with tb_col2:
        fill_notes_clicked = st.button(
            "✨ 예시 보충 멘트 채우기",
            use_container_width=True,
            help="비어 있는 Expert_Note와 Action_Reason 컬럼에 실제 수치 기반의 도메인 전문가 예시 멘트를 자동으로 채웁니다.",
        )
    with tb_col3:
        select_all_clicked = st.button(
            "☑️ 전체 선택/해제",
            use_container_width=True,
            help="모든 골든 케이스의 '포함 여부' 체크박스를 일괄 선택하거나 해제합니다.",
        )
    with tb_col4:
        clear_notes_clicked = st.button(
            "🧹 코멘트 초기화",
            use_container_width=True,
            help="입력된 Expert_Note와 Action_Reason 보충 멘트를 모두 비웁니다.",
        )

    # 툴바 버튼 액션 처리 (st.data_editor 렌더링 이전에 처리하여 즉시 반영)
    if fill_notes_clicked:
        working_df = (
            st.session_state["golden_cases_df"].copy()
            if st.session_state.get("golden_cases_df") is not None
            else golden_base_df.copy()
        )
        for idx, row in working_df.iterrows():
            sample_exp, sample_act = build_default_sample_notes(row, proc_df, var_mapping)
            if not str(row.get("Expert_Note", "")).strip():
                working_df.at[idx, "Expert_Note"] = sample_exp
            if not str(row.get("Action_Reason", "")).strip():
                working_df.at[idx, "Action_Reason"] = sample_act
        update_notes_cache_from_df(working_df)
        sync_golden_editor_base_df(working_df)
        st.session_state["step3_flash_type"] = "success"
        st.session_state["step3_flash_msg"] = (
            "✨ 모든 골든 케이스에 수치 기반 도메인 전문가 예시 보충 멘트(`Expert_Note`, `Action_Reason`)가 채워졌습니다. 표에서 자유롭게 수정해 보세요!"
        )
        st.rerun()

    if select_all_clicked:
        working_df = (
            st.session_state["golden_cases_df"].copy()
            if st.session_state.get("golden_cases_df") is not None
            else golden_base_df.copy()
        )
        all_checked = bool(working_df["포함 여부"].all())
        working_df["포함 여부"] = not all_checked
        update_notes_cache_from_df(working_df)
        sync_golden_editor_base_df(working_df)
        st.rerun()

    if clear_notes_clicked:
        working_df = (
            st.session_state["golden_cases_df"].copy()
            if st.session_state.get("golden_cases_df") is not None
            else golden_base_df.copy()
        )
        working_df["Expert_Note"] = ""
        working_df["Action_Reason"] = ""
        update_notes_cache_from_df(working_df)
        sync_golden_editor_base_df(working_df)
        st.session_state["step3_flash_type"] = "warning"
        st.session_state["step3_flash_msg"] = "🧹 모든 케이스의 `Expert_Note`와 `Action_Reason`이 초기화되었습니다."
        st.rerun()

    st.caption(
        "💡 아래 표에서 **`포함 여부`** 체크박스를 선택하고, **`Expert_Note`**(공정 상태 및 비효율 원인 진단)와 "
        "**`Action_Reason`**(제어 변수 조작 핵심 근거) 셀을 더블클릭하여 직접 타이핑하면 하단 **실시간 문장 미리보기**에 즉시 반영됩니다."
    )

    golden_editor_version = st.session_state.get("golden_editor_version", 0)
    edited_golden_df = st.data_editor(
        st.session_state["golden_base_df"],
        key=f"golden_cases_editor_{golden_editor_version}",
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        column_order=[
            "포함 여부",
            "케이스 ID",
            "Before 시점",
            "After 시점",
            "제어 변수 변화 (Before→After)",
            "KPI 개선 결과 (Before→After)",
            "상태/제약 추이",
            "Expert_Note",
            "Action_Reason",
        ],
        disabled=[
            "케이스 ID",
            "Before 행",
            "After 행",
            "Before 시점",
            "After 시점",
            "제어 변수 변화 (Before→After)",
            "KPI 개선 결과 (Before→After)",
            "상태/제약 추이",
        ],
        column_config={
            "포함 여부": st.column_config.CheckboxColumn(
                "포함 여부",
                help="체크된 케이스만 Step 4에서 최종 학습 데이터(JSONL)로 생성됩니다.",
                default=True,
                width="small",
            ),
            "케이스 ID": st.column_config.TextColumn(
                "케이스 ID",
                help="추출된 골든 케이스 고유 식별자",
                width="small",
            ),
            "Before 시점": st.column_config.TextColumn(
                "Before 시점 (조작 전)",
                width="medium",
            ),
            "After 시점": st.column_config.TextColumn(
                "After 시점 (조작 후)",
                width="medium",
            ),
            "제어 변수 변화 (Before→After)": st.column_config.TextColumn(
                "제어 변수 변화 (Control)",
                help="Before 시점 대비 After 시점의 제어 변수(control) 수치 변화 요약",
                width="large",
            ),
            "KPI 개선 결과 (Before→After)": st.column_config.TextColumn(
                "KPI 개선 결과 (KPI)",
                help="설정한 모든 KPI의 Before → After 수치 변화 요약",
                width="large",
            ),
            "상태/제약 추이": st.column_config.TextColumn(
                "상태/제약 추이 (State)",
                help="공정 상태 및 안전 제약 변수의 Before → After 변화",
                width="medium",
            ),
            "Expert_Note": st.column_config.TextColumn(
                "✏️ Expert_Note (공정 상태 및 비효율 원인 진단 보충 멘트)",
                help="도메인 전문가가 진단한 공정 상태 및 비효율 원인을 직접 타이핑하세요. 수정 즉시 아래 미리보기에 반영됩니다.",
                width="large",
            ),
            "Action_Reason": st.column_config.TextColumn(
                "✏️ Action_Reason (제어 변수 조작 핵심 근거 보충 멘트)",
                help="제어 변수를 해당 방향으로 조작한 공학적/현장적 핵심 근거를 직접 타이핑하세요. 수정 즉시 아래 미리보기에 반영됩니다.",
                width="large",
            ),
        },
    )

    # 편집 결과를 세션 상태 및 노트에 즉시 동기화
    st.session_state["golden_cases_df"] = edited_golden_df
    update_notes_cache_from_df(edited_golden_df)

    # 사용자가 방금 특정 행의 Expert_Note 또는 Action_Reason을 수정했다면,
    # 실시간 미리보기 드롭다운이 해당 행을 즉시 가리키도록 자동 포커스 전환
    prev_snapshot: Dict[str, Tuple[str, str]] = st.session_state.get(
        "last_golden_notes_snapshot", {}
    )
    newly_edited_case_id: Optional[str] = None
    new_snapshot: Dict[str, Tuple[str, str]] = {}

    for _, row in edited_golden_df.iterrows():
        cid = str(row["케이스 ID"])
        cur_pair = (
            str(row.get("Expert_Note", "") or ""),
            str(row.get("Action_Reason", "") or ""),
        )
        new_snapshot[cid] = cur_pair
        if cid in prev_snapshot and prev_snapshot[cid] != cur_pair:
            newly_edited_case_id = cid

    st.session_state["last_golden_notes_snapshot"] = new_snapshot

    included_count = int(edited_golden_df["포함 여부"].sum())
    note_filled_count = int(
        (
            (edited_golden_df["Expert_Note"].str.strip() != "")
            | (edited_golden_df["Action_Reason"].str.strip() != "")
        ).sum()
    )
    st.info(
        f"✅ **추출된 골든 케이스 총 `{len(edited_golden_df)}`건** 중 "
        f"**학습 포함 선택: `{included_count}`건** | "
        f"**전문가 보충 멘트 입력 완료: `{note_filled_count}`건**"
    )

    # --------------------------------------------------------------------------
    # 4) 순수 파이썬 규칙 기반(f-string) 문장 생성 및 실시간 미리보기 (Preview) 박스
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 🔎 실시간 학습 문장 미리보기 (Live Prompt Preview)")
    st.caption(
        "외부 LLM 없이 순수 파이썬 규칙 기반 템플릿(f-string)으로 **수치 데이터(Before→After) + 4대 변수 메타정보 + 전문가 보충 멘트(`Expert_Note`, `Action_Reason`)**를 "
        "결합하여 생성된 한국어 `instruction`, `input`, `output` 문장입니다."
    )

    case_ids = [str(cid) for cid in edited_golden_df["케이스 ID"].tolist()]
    case_label_map: Dict[str, str] = {}
    for _, r in edited_golden_df.iterrows():
        cid = str(r["케이스 ID"])
        inc_mark = "🟢[포함]" if bool(r["포함 여부"]) else "⚪[제외]"
        has_note = (
            "✏️멘트입력됨"
            if (
                str(r.get("Expert_Note", "")).strip()
                or str(r.get("Action_Reason", "")).strip()
            )
            else "기본템플릿"
        )
        case_label_map[cid] = (
            f"{cid} {inc_mark} ({r['Before 시점']} → {r['After 시점']}) | {r['KPI 개선 결과 (Before→After)']} [{has_note}]"
        )

    if newly_edited_case_id and newly_edited_case_id in case_ids:
        st.session_state["preview_case_select"] = newly_edited_case_id
    elif st.session_state.get("preview_case_select") not in case_ids:
        st.session_state["preview_case_select"] = case_ids[0]

    selected_case_id = st.selectbox(
        "미리보기할 골든 케이스 선택 (표에서 멘트를 수정하면 해당 행으로 자동 전환됩니다)",
        options=case_ids,
        format_func=lambda x: case_label_map.get(x, x),
        key="preview_case_select",
    )

    selected_row = edited_golden_df[
        edited_golden_df["케이스 ID"] == selected_case_id
    ].iloc[0]

    qa_dict = generate_qa_from_golden_case(
        case_row=selected_row,
        proc_df=proc_df,
        var_mapping=var_mapping,
        process_name=process_name,
        time_col=time_col,
        lookahead_rows=int(lookahead_val),
        resample_rule=resample_rule,
    )

    with st.container(border=True):
        exp_val = str(selected_row.get("Expert_Note", "") or "").strip()
        act_val = str(selected_row.get("Action_Reason", "") or "").strip()

        badge_col1, badge_col2 = st.columns(2)
        with badge_col1:
            if exp_val:
                st.success(f"**🩺 반영된 Expert_Note**: {exp_val}")
            else:
                st.caption("🩺 **Expert_Note**: *(미입력 — 위 표에서 타이핑하면 즉시 아래 [1. 공정 상태 및 비효율 원인 진단]에 반영됩니다)*")
        with badge_col2:
            if act_val:
                st.success(f"**🛠️ 반영된 Action_Reason**: {act_val}")
            else:
                st.caption("🛠️ **Action_Reason**: *(미입력 — 위 표에서 타이핑하면 즉시 아래 [3. 제어 변수 조작 핵심 근거]에 반영됩니다)*")

        tab_readable, tab_json = st.tabs(
            ["📄 문장형 미리보기 (Instruction / Input / Output)", "🧩 JSONL 레코드 원문 미리보기"]
        )

        with tab_readable:
            st.markdown("##### 1️⃣ `instruction` (시스템 역할 및 작업 지시문)")
            st.info(qa_dict["instruction"])

            in_col, out_col = st.columns(2)
            with in_col:
                st.markdown("##### 2️⃣ `input` (현재 운전 조건 및 안전 제약 — Before)")
                st.code(qa_dict["input"], language="text")
            with out_col:
                st.markdown("##### 3️⃣ `output` (전문가 진단 · 최적 조작 권고 · 기대 효과 — After)")
                st.code(qa_dict["output"], language="text")

        with tab_json:
            st.code(
                json.dumps(qa_dict, ensure_ascii=False, indent=2),
                language="json",
            )


def render_step4() -> None:
    """
    [Step 4] JSONL 학습/검증 데이터셋 분할 생성 및 LLaMA-Factory dataset_info.json 자동 등록 UI를 렌더링합니다.
    """
    st.subheader("Step 4. JSONL 학습/검증 분할 생성 및 LLaMA-Factory 자동 등록")
    st.caption(
        "Step 3에서 선별·보강한 골든 케이스들을 LLaMA-Factory 표준 지시문 포맷(`instruction`, `input`, `output`)의 JSONL 파일로 변환하고, "
        "`./data/dataset_info.json`에 자동으로 병합 등록합니다."
    )

    proc_df: Optional[pd.DataFrame] = st.session_state.get("processed_df")
    golden_cases_df: Optional[pd.DataFrame] = st.session_state.get("golden_cases_df")
    var_mapping: Dict[str, List[Dict[str, Any]]] = st.session_state.get("var_mapping", {})
    process_name: str = st.session_state.get("process_name", "나프타 열분해 및 C2 정류 공정")
    time_col: Optional[str] = st.session_state.get("time_col")
    lookahead_rows: int = int(st.session_state.get("lookahead_rows", 1))
    resample_rule: str = st.session_state.get("resample_rule", "원본 유지(집계 안 함)")

    # 현재 선택된(포함 여부 == True) 골든 케이스 수 계산
    included_df = pd.DataFrame()
    if (
        proc_df is not None
        and not proc_df.empty
        and golden_cases_df is not None
        and not golden_cases_df.empty
        and "포함 여부" in golden_cases_df.columns
    ):
        included_df = golden_cases_df[golden_cases_df["포함 여부"] == True].copy()

    total_selected = len(included_df)

    # --------------------------------------------------------------------------
    # 1) 데이터셋 이름 입력창 및 학습/검증 분할 비율 슬라이더
    # --------------------------------------------------------------------------
    with st.container(border=True):
        st.markdown("##### ⚙️ 데이터셋 파일명 및 학습/검증(Train/Val) 분할 설정")
        cfg_col1, cfg_col2 = st.columns([1.3, 1.7])

        with cfg_col1:
            raw_ds_name = st.text_input(
                "🏷️ 저장할 데이터셋 이름 (예: ncc_opt_v1)",
                value=st.session_state.get("dataset_name", "ncc_opt_v1"),
                placeholder="ncc_opt_v1",
                help="입력한 이름을 기준으로 `./data/{데이터셋이름}_train.jsonl` (및 분할 시 `_val.jsonl`) 파일이 생성되고 `dataset_info.json`에 등록됩니다.",
            )
            st.session_state["dataset_name"] = raw_ds_name
            clean_ds_name = sanitize_dataset_name(raw_ds_name)

        with cfg_col2:
            val_split_pct = st.slider(
                "📊 검증 데이터(Validation) 분할 비율 (%) — 0% 선택 시 분할 없이 단일 파일(_train.jsonl)로 생성",
                min_value=0,
                max_value=50,
                value=int(st.session_state.get("val_split_pct", 10)),
                step=5,
                format="%d%%",
                help="예: 10% 선택 시 전체 샘플의 90%는 `_train.jsonl`, 10%는 `_val.jsonl`로 분할 저장됩니다. 0% 선택 시 `_train.jsonl` 단일 파일만 생성합니다.",
            )
            st.session_state["val_split_pct"] = int(val_split_pct)

        # 분할 예상 건수 및 생성 예정 파일 경로 실시간 안내
        train_Path_preview = os.path.join(DATA_DIR, f"{clean_ds_name}_train.jsonl")
        val_path_preview = os.path.join(DATA_DIR, f"{clean_ds_name}_val.jsonl")
        info_path_preview = os.path.join(DATA_DIR, "dataset_info.json")

        if total_selected == 0:
            est_train_cnt, est_val_cnt = 0, 0
        elif val_split_pct <= 0 or total_selected == 1:
            est_train_cnt, est_val_cnt = total_selected, 0
        else:
            est_val_cnt = max(1, min(int(round(total_selected * (val_split_pct / 100.0))), total_selected - 1))
            est_train_cnt = total_selected - est_val_cnt

        if val_split_pct == 0:
            split_mode_desc = (
                f"**단일 파일 모드 (검증셋 0%)** 👉 학습셋 `{train_Path_preview}` (**{est_train_cnt}건**) 단일 파일 생성"
            )
        elif total_selected == 1:
            split_mode_desc = (
                f"**학습/검증 분할 모드 ({100 - val_split_pct}% : {val_split_pct}%)** 👉 현재 선택된 샘플이 1건뿐이므로 학습셋 `{train_Path_preview}` (**1건**) 단일 파일로 보호 생성됩니다."
            )
        else:
            split_mode_desc = (
                f"**학습/검증 분할 모드 ({100 - val_split_pct}% : {val_split_pct}%)** 👉 "
                f"학습셋 `{train_Path_preview}` (**{est_train_cnt}건**) + 검증셋 `{val_path_preview}` (**{est_val_cnt}건**) 분할 생성"
            )

        st.info(
            f"📦 **Step 3 선택 샘플**: 총 **`{total_selected}`건**  |  {split_mode_desc}\n\n"
            f"🗂️ **LLaMA-Factory 자동 등록 대상**: `{info_path_preview}` (기존 등록 데이터셋 보존 및 병합 업데이트)"
        )

        # 실행 버튼
        generate_clicked = st.button(
            "🚀 학습 데이터 생성 및 LLaMA-Factory 자동 등록",
            type="primary",
            use_container_width=True,
            disabled=(total_selected == 0),
            help="선택된 골든 케이스들을 JSONL 파일로 저장하고 ./data/dataset_info.json에 자동으로 등록합니다.",
        )

        if total_selected == 0:
            st.caption(
                "⚠️ Step 3에서 추출된 골든 케이스 중 `포함 여부`가 체크된 항목이 최소 1개 이상 있어야 학습 데이터를 생성할 수 있습니다."
            )

    # --------------------------------------------------------------------------
    # 2) 버튼 클릭 시: JSONL 파일 생성 및 ./data/dataset_info.json 병합 저장
    # --------------------------------------------------------------------------
    if generate_clicked:
        if proc_df is None or proc_df.empty or included_df.empty:
            st.error("생성할 골든 케이스 데이터가 없습니다. Step 1~Step 3 설정을 먼저 확인해 주세요.")
        else:
            try:
                all_records: List[Dict[str, str]] = []
                for _, case_row in included_df.iterrows():
                    qa_item = generate_qa_from_golden_case(
                        case_row=case_row,
                        proc_df=proc_df,
                        var_mapping=var_mapping,
                        process_name=process_name,
                        time_col=time_col,
                        lookahead_rows=lookahead_rows,
                        resample_rule=resample_rule,
                    )
                    all_records.append(qa_item)

                train_records, val_records = split_train_val_records(
                    all_records, int(val_split_pct)
                )

                train_filename = f"{clean_ds_name}_train.jsonl"
                train_filepath = os.path.join(DATA_DIR, train_filename)
                write_jsonl_file(train_filepath, train_records)

                dataset_entries_to_register: Dict[str, Dict[str, Any]] = {
                    f"{clean_ds_name}_train": {
                        "file_name": train_filename,
                        "columns": {
                            "prompt": "instruction",
                            "query": "input",
                            "response": "output",
                        },
                    }
                }

                val_filename: Optional[str] = None
                val_filepath: Optional[str] = None
                if val_records:
                    val_filename = f"{clean_ds_name}_val.jsonl"
                    val_filepath = os.path.join(DATA_DIR, val_filename)
                    write_jsonl_file(val_filepath, val_records)
                    dataset_entries_to_register[f"{clean_ds_name}_val"] = {
                        "file_name": val_filename,
                        "columns": {
                            "prompt": "instruction",
                            "query": "input",
                            "response": "output",
                        },
                    }
                else:
                    # 기존에 동일 이름의 _val.jsonl로 혼동하지 않도록 단일 파일 생성 상태 기록
                    val_filepath = None

                info_path, merged_dataset_info = update_dataset_info_json(
                    dataset_entries_to_register
                )

                st.session_state["last_export_result"] = {
                    "dataset_name": clean_ds_name,
                    "process_name": process_name,
                    "val_split_pct": int(val_split_pct),
                    "total_count": len(all_records),
                    "train_count": len(train_records),
                    "val_count": len(val_records),
                    "train_filepath": train_filepath,
                    "val_filepath": val_filepath,
                    "train_key": f"{clean_ds_name}_train",
                    "val_key": f"{clean_ds_name}_val" if val_records else None,
                    "info_path": info_path,
                    "registered_entries": dataset_entries_to_register,
                    "merged_dataset_info": merged_dataset_info,
                    "sample_record": train_records[0] if train_records else None,
                }
            except Exception as exc:
                st.error(f"학습 데이터 생성 및 등록 중 오류가 발생했습니다: {exc}")

    # --------------------------------------------------------------------------
    # 3) 완료 시 생성된 샘플 수 요약 및 LLaMA-Factory 웹 UI(:7860) 학습 진행 안내 가이드 박스
    # --------------------------------------------------------------------------
    export_res = st.session_state.get("last_export_result")
    if export_res:
        st.success(
            f"🎉 **데이터셋 `{export_res['dataset_name']}` 생성 및 `{export_res['info_path']}` 자동 등록이 완료되었습니다!**"
        )

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("총 변환 샘플 수", f"{export_res['total_count']:,} 건")
        with m_col2:
            st.metric(
                "학습 데이터 (Train)",
                f"{export_res['train_count']:,} 건",
                delta=os.path.basename(export_res["train_filepath"]),
            )
        with m_col3:
            if export_res["val_filepath"]:
                st.metric(
                    "검증 데이터 (Val)",
                    f"{export_res['val_count']:,} 건",
                    delta=os.path.basename(export_res["val_filepath"]),
                )
            else:
                st.metric("검증 데이터 (Val)", "0 건 (단일 파일)", delta="분할 안 함")
        with m_col4:
            registered_keys_list = list(export_res["registered_entries"].keys())
            st.metric(
                "dataset_info.json 등록 키",
                f"{len(registered_keys_list)} 개",
                delta=", ".join(registered_keys_list),
            )

        with st.container(border=True):
            st.markdown("##### 📂 생성된 파일 경로 및 다운로드")
            file_lines = [
                f"- **학습용 JSONL (`{export_res['train_key']}`)**: `{export_res['train_filepath']}` (**{export_res['train_count']:,}행**)"
            ]
            if export_res["val_filepath"]:
                file_lines.append(
                    f"- **검증용 JSONL (`{export_res['val_key']}`)**: `{export_res['val_filepath']}` (**{export_res['val_count']:,}행**)"
                )
            file_lines.append(
                f"- **LLaMA-Factory 데이터셋 메타파일**: `{export_res['info_path']}` (총 **{len(export_res['merged_dataset_info'])}개** 데이터셋 등록됨)"
            )
            st.markdown("\n".join(file_lines))

            # 로컬 PC에서도 즉시 내려받아 확인할 수 있도록 다운로드 버튼 제공
            dl_col1, dl_col2, dl_col3 = st.columns(3)
            with dl_col1:
                if os.path.exists(export_res["train_filepath"]):
                    with open(export_res["train_filepath"], "r", encoding="utf-8") as f:
                        train_raw_text = f.read()
                    st.download_button(
                        label=f"⬇️ {os.path.basename(export_res['train_filepath'])} 다운로드",
                        data=train_raw_text,
                        file_name=os.path.basename(export_res["train_filepath"]),
                        mime="application/jsonl",
                        use_container_width=True,
                    )
            with dl_col2:
                if export_res["val_filepath"] and os.path.exists(export_res["val_filepath"]):
                    with open(export_res["val_filepath"], "r", encoding="utf-8") as f:
                        val_raw_text = f.read()
                    st.download_button(
                        label=f"⬇️ {os.path.basename(export_res['val_filepath'])} 다운로드",
                        data=val_raw_text,
                        file_name=os.path.basename(export_res["val_filepath"]),
                        mime="application/jsonl",
                        use_container_width=True,
                    )
            with dl_col3:
                if os.path.exists(export_res["info_path"]):
                    with open(export_res["info_path"], "r", encoding="utf-8") as f:
                        info_raw_text = f.read()
                    st.download_button(
                        label="⬇️ dataset_info.json 다운로드",
                        data=info_raw_text,
                        file_name="dataset_info.json",
                        mime="application/json",
                        use_container_width=True,
                    )

            with st.expander("🔍 `./data/dataset_info.json` 등록 내용 확인하기", expanded=False):
                st.code(
                    json.dumps(export_res["merged_dataset_info"], ensure_ascii=False, indent=2),
                    language="json",
                )

        # LLaMA-Factory 웹 UI (:7860) 학습 진행 안내 가이드 박스
        with st.container(border=True):
            st.markdown("#### 🦙 LLaMA-Factory 웹 UI (`:7860`) 미세조정(Fine-Tuning) 학습 진행 가이드")
            val_guide_line = (
                f"- **검증 데이터셋(Eval Dataset) 지정 시**: `{export_res['val_key']}`를 함께 활용하거나 Evaluation 탭에서 검증을 수행할 수 있습니다."
                if export_res["val_key"]
                else "- **단일 파일로 생성한 경우**: LLaMA-Factory UI의 `Val size`(검증 비율) 옵션에 `0.1` 등을 입력하여 학습 중 자체 분할 검증을 수행할 수 있습니다."
            )
            st.info(
                f"""
                생성된 학습 데이터와 `dataset_info.json`은 Docker 컨테이너 및 로컬의 `./data` 경로에 즉시 반영되었습니다.
                아래 순서에 따라 **LLaMA-Factory WebUI (`http://localhost:7860` 또는 `http://<GPU서버_IP>:7860`)**에서 학습을 시작하세요!

                1. **LLaMA-Factory WebUI 접속**: 브라우저에서 **`http://localhost:7860`** (또는 서버의 `:7860` 포트)을 엽니다.
                2. **모델 및 어댑터 설정 (상단 패널)**:
                   - **Model Name / Model Path**: 학습할 베이스 LLM (예: `Qwen2.5-7B-Instruct`, `Llama-3.1-8B-Instruct` 등)을 선택합니다.
                   - **Finetuning Method**: `lora` (권장) 또는 `full`을 선택합니다.
                3. **학습 단계 및 데이터셋 선택 (`Train` 탭)**:
                   - **Stage**: `Supervised Fine-Tuning` (SFT)을 선택합니다.
                   - **Data dir**: `./data` (기본값 `data`)로 설정되어 있는지 확인합니다.
                   - **Dataset**: 드롭다운을 클릭하여 방금 자동 등록된 **`{export_res['train_key']}`**을(를) 선택합니다. *(목록이 안 보이면 데이터셋 우측 새로고침 버튼을 클릭하세요)*
                   {val_guide_line}
                4. **주요 하이퍼파라미터 권장 설정 (도메인 특화 SFT 기준)**:
                   - **Learning Rate**: `5e-5` ~ `1e-4`  |  **Epochs**: `3.0` ~ `5.0`  |  **Cutoff Length**: `2048` 이상
                   - **LoRA Rank / Alpha**: `rank = 16`, `alpha = 32`
                5. **학습 시작**: 하단의 **`Preview Command`**로 명령어를 확인한 뒤 **`Start`** 버튼을 클릭하면 `{export_res['process_name']}` 최적화 도메인 LLM 학습이 즉시 시작됩니다!
                """
            )


# ==============================================================================
# 3. 메인 엔트리포인트
# ==============================================================================
def main() -> None:
    st.set_page_config(
        page_title="RTDB 데이터 변환 웹 GUI",
        page_icon="🏭",
        layout="wide",
    )

    ensure_directories()
    init_session_state()

    st.title("🏭 RTDB 운전 데이터 → LLaMA-Factory 학습 데이터 변환기")
    st.caption(
        "플랜트 시계열 CSV 데이터를 화학공학 도메인 구조(Feed / Control / State / KPI)에 맞춰 정제하고 고품질 지시문 데이터셋(JSONL)으로 변환합니다."
    )

    # 상단 온보딩 가이드
    render_onboarding_guide()

    st.divider()

    # Step 1 렌더링
    render_step1()

    st.divider()

    # Step 2 렌더링
    render_step2()

    st.divider()

    # Step 3 렌더링
    render_step3()

    st.divider()

    # Step 4 렌더링
    render_step4()


if __name__ == "__main__":
    main()
