import io
import re
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


HOURLY_WAGE = 1250

SHIFT_END_MAP = {
    "L": "21:00",
}

SHIFT_COLUMNS = ["画像", "日付", "出勤", "退勤"]
RESULT_COLUMNS = [
    "画像",
    "日付",
    "出勤",
    "退勤",
    "勤務時間[h]",
    "休憩時間[h]",
    "実働時間[h]",
    "給料[円]",
]


@st.cache_resource
def get_ocr_reader():
    import easyocr

    return easyocr.Reader(["ja", "en"], gpu=False)


@st.cache_data(show_spinner=False)
def read_ocr_result(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return get_ocr_reader().readtext(np.array(image))


def normalize_ocr_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    return text.translate(str.maketrans({
        "O": "0",
        "o": "0",
        "〇": "0",
        "○": "0",
    }))


def normalize_time_token(value) -> str | None:
    text = normalize_ocr_text(value).strip()
    shift_symbol = text.upper()
    if shift_symbol in SHIFT_END_MAP:
        return SHIFT_END_MAP[shift_symbol]

    match = re.search(r"(?<!\d)([0-2]?\d)\s*[:：.．。]\s*([0-5]\d)(?!\d)", text)
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour > 23 or minute > 59:
        return None

    return f"{hour:02d}:{minute:02d}"


def normalize_times_for_display(text: str) -> str:
    normalized_text = normalize_ocr_text(text)
    pattern = r"(?<!\d)([0-2]?\d)\s*[:：.．。]\s*([0-5]\d)(?!\d)"

    def replace_time(match: re.Match) -> str:
        hour = int(match.group(1))
        minute = int(match.group(2))
        if hour > 23 or minute > 59:
            return match.group(0)
        return f"{hour:02d}:{minute:02d}"

    return re.sub(pattern, replace_time, normalized_text)


def bbox_center(bbox):
    xs = [point[0] for point in bbox]
    ys = [point[1] for point in bbox]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def bbox_bounds(bbox):
    xs = [point[0] for point in bbox]
    ys = [point[1] for point in bbox]
    return min(xs), min(ys), max(xs), max(ys)


def extract_times_with_positions(text: str, bbox):
    times = extract_times_from_text(text)
    if not times:
        return []

    x1, y1, x2, y2 = bbox_bounds(bbox)
    center_x = (x1 + x2) / 2
    height = y2 - y1

    positioned_times = []
    for i, time_text in enumerate(times):
        if len(times) == 1:
            center_y = (y1 + y2) / 2
        else:
            center_y = y1 + height * (i + 1) / (len(times) + 1)

        positioned_times.append({
            "text": time_text,
            "x": center_x,
            "y": center_y,
        })

    return positioned_times


def extract_l_symbol_with_position(text: str, bbox):
    normalized_text = normalize_ocr_text(text).strip().upper()
    if normalized_text != "L":
        return None

    center_x, center_y = bbox_center(bbox)
    return {
        "text": "L",
        "x": center_x,
        "y": center_y,
    }


def estimate_row_geometry(ocr_result, image_size):
    image_width, image_height = image_size
    date_points = []

    for item in ocr_result:
        bbox, text = item[0], item[1]
        normalized_text = normalize_ocr_text(text).strip()
        if not re.fullmatch(r"\d{1,2}", normalized_text):
            continue

        date = int(normalized_text)
        if not 1 <= date <= 31:
            continue

        center_x, center_y = bbox_center(bbox)
        if center_y < image_height * 0.08:
            continue

        if center_x < image_width / 2 and date <= 15:
            date_points.append((date - 1, center_y))
        elif center_x >= image_width / 2 and date >= 16:
            date_points.append((date - 16, center_y))

    row_height = image_height * 0.0565
    if len(date_points) >= 2:
        slopes = []
        for i, (row_a, y_a) in enumerate(date_points):
            for row_b, y_b in date_points[i + 1:]:
                if row_a != row_b:
                    slopes.append(abs((y_b - y_a) / (row_b - row_a)))
        useful_slopes = [
            slope
            for slope in slopes
            if image_height * 0.035 <= slope <= image_height * 0.08
        ]
        if useful_slopes:
            row_height = float(np.median(useful_slopes))

    if date_points:
        first_row_center = float(np.median([
            center_y - row_index * row_height
            for row_index, center_y in date_points
        ]))
    else:
        first_row_center = image_height * 0.112

    return first_row_center, row_height


def get_layout_shift_side(x: float, image_width: int) -> str | None:
    left_shift_start = image_width * 0.30
    left_shift_end = image_width * 0.51
    right_shift_start = image_width * 0.69
    right_shift_end = image_width * 0.90

    if left_shift_start <= x <= left_shift_end:
        return "left"
    if right_shift_start <= x <= right_shift_end:
        return "right"
    return None


def make_shift_table_from_ocr_result(ocr_result, image_size):
    image_width, image_height = image_size
    first_row_center, row_height = estimate_row_geometry(ocr_result, image_size)
    shift_tokens = []

    for item in ocr_result:
        bbox, text = item[0], item[1]
        shift_tokens.extend(extract_times_with_positions(text, bbox))

        l_symbol = extract_l_symbol_with_position(text, bbox)
        if l_symbol is not None:
            shift_tokens.append(l_symbol)

    rows_by_date = {}
    for token in shift_tokens:
        side = get_layout_shift_side(token["x"], image_width)
        if side is None:
            continue

        row_index = round((token["y"] - first_row_center) / row_height)
        row_center_y = first_row_center + row_index * row_height
        if abs(token["y"] - row_center_y) > row_height * 0.48:
            continue

        if side == "left":
            if not 0 <= row_index <= 14:
                continue
            date = row_index + 1
        else:
            if not 0 <= row_index <= 15:
                continue
            date = row_index + 16

        rows_by_date.setdefault(date, []).append(token)

    rows = []
    for date, tokens in sorted(rows_by_date.items()):
        times = [
            token["text"]
            for token in sorted(tokens, key=lambda token: token["y"])
            if normalize_time_token(token["text"]) is not None
        ]
        symbols = {
            token["text"]
            for token in tokens
            if normalize_ocr_text(token["text"]).strip().upper() in SHIFT_END_MAP
        }

        if len(times) >= 2:
            start = times[0]
            end = times[-1]
        elif len(times) == 1 and symbols:
            start = times[0]
            end = SHIFT_END_MAP[sorted(symbols)[0]]
        elif len(times) == 1:
            start = times[0]
            end = None
        else:
            continue

        rows.append({
            "画像": None,
            "日付": date,
            "出勤": start,
            "退勤": end,
        })

    return pd.DataFrame(rows, columns=SHIFT_COLUMNS)


def time_to_minutes(t: str) -> int:
    normalized = normalize_time_token(t)
    if normalized is None:
        raise ValueError(f"Invalid time: {t}")

    h, m = map(int, normalized.split(":"))
    return h * 60 + m


def minutes_to_hours(minutes: int) -> float:
    return minutes / 60


def calc_break_minutes(work_minutes: int) -> int:
    work_hours = work_minutes / 60

    if work_hours >= 10:
        return 90
    if work_hours >= 8:
        return 60
    if work_hours >= 6:
        return 45
    return 0


def extract_times_from_text(text: str):
    """
    OCR結果から 10:00, 13.30, 1O.oo などの時刻を抽出する。
    """
    normalized_text = normalize_ocr_text(text)
    pattern = r"(?<!\d)([0-2]?\d)\s*[:：.．。]\s*([0-5]\d)(?!\d)"

    times = []
    for hour, minute in re.findall(pattern, normalized_text):
        h = int(hour)
        m = int(minute)
        if h <= 23 and m <= 59:
            times.append(f"{h:02d}:{m:02d}")

    return times


def extract_dates_from_text(text: str):
    """
    OCR結果から、単独行に出た 1-31 の数字だけを日付候補として抽出する。
    時刻の 13.30 に含まれる 13 や 30 を日付として拾わないようにする。
    """
    dates = []

    for line in text.splitlines():
        normalized_line = normalize_ocr_text(line).strip()
        if not re.fullmatch(r"\d{1,2}", normalized_line):
            continue

        value = int(normalized_line)
        if 1 <= value <= 31:
            dates.append(value)

    return dates


def make_shift_table(ocr_text: str):
    times = extract_times_from_text(ocr_text)
    dates = extract_dates_from_text(ocr_text)

    rows = []

    # 時刻は2個ずつ「出勤・退勤」として仮に割り当てる。
    pair_count = len(times) // 2

    for i in range(pair_count):
        date = dates[i] if i < len(dates) else None
        start = times[2 * i]
        end = times[2 * i + 1]

        rows.append({
            "画像": None,
            "日付": date,
            "出勤": start,
            "退勤": end,
        })

    return pd.DataFrame(rows, columns=SHIFT_COLUMNS)


def calculate_salary(df: pd.DataFrame, hourly_wage: int):
    result_rows = []

    for _, row in df.iterrows():
        source_image = row.get("画像")
        date = row.get("日付")
        start = normalize_time_token(row.get("出勤"))
        end = normalize_time_token(row.get("退勤"))

        if start is None or end is None:
            continue

        start_min = time_to_minutes(start)
        end_min = time_to_minutes(end)

        if end_min <= start_min:
            end_min += 24 * 60

        work_minutes = end_min - start_min
        break_minutes = calc_break_minutes(work_minutes)
        actual_minutes = work_minutes - break_minutes

        if actual_minutes <= 0:
            continue

        pay = actual_minutes / 60 * hourly_wage

        result_rows.append({
            "画像": source_image,
            "日付": date,
            "出勤": start,
            "退勤": end,
            "勤務時間[h]": round(minutes_to_hours(work_minutes), 2),
            "休憩時間[h]": round(minutes_to_hours(break_minutes), 2),
            "実働時間[h]": round(minutes_to_hours(actual_minutes), 2),
            "給料[円]": round(pay),
        })

    result_df = pd.DataFrame(result_rows, columns=RESULT_COLUMNS)

    total_hours = result_df["実働時間[h]"].sum()
    total_pay = result_df["給料[円]"].sum()

    return result_df, total_hours, total_pay


st.title("シフト表 給料計算アプリ")

st.write("シフト表の画像をアップロードすると、勤務時間・休憩時間・給料を計算する。")

hourly_wage = st.number_input(
    "時給[円]",
    min_value=0,
    value=HOURLY_WAGE,
    step=50,
)

uploaded_file = st.file_uploader(
    "シフト表画像をアップロード（複数選択可）",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_file:
    parsed_tables = []

    st.subheader("アップロード画像とOCR読み取り結果")
    st.caption("O/o の 0 誤読、13.30 のような時刻表記は自動補正しています。")

    for index, file in enumerate(uploaded_file):
        file_bytes = file.getvalue()
        image = Image.open(io.BytesIO(file_bytes))

        with st.expander(file.name, expanded=len(uploaded_file) == 1):
            st.image(image, caption=file.name, width="stretch")

            try:
                with st.spinner("OCRで読み取り中..."):
                    ocr_result = read_ocr_result(file_bytes)
            except Exception as exc:
                st.error("OCRの読み取り中にエラーが発生しました。")
                st.exception(exc)
                continue

            raw_ocr_text = "\n".join([item[1] for item in ocr_result])
            normalized_ocr_text = normalize_times_for_display(raw_ocr_text)
            edited_ocr_text = st.text_area(
                "読み取った文字（修正可）",
                normalized_ocr_text,
                height=200,
                key=f"ocr_text_{index}_{file.name}",
            )

            image_df = make_shift_table_from_ocr_result(ocr_result, image.size)
            if image_df.empty:
                image_df = make_shift_table(edited_ocr_text)
            image_df["画像"] = file.name
            parsed_tables.append(image_df[SHIFT_COLUMNS])

    if parsed_tables:
        df = pd.concat(parsed_tables, ignore_index=True)
    else:
        df = pd.DataFrame(columns=SHIFT_COLUMNS)

    st.subheader("読み取り後のシフト表")
    st.write("画像ごとの読み取り結果です。必要に応じて日付・出勤・退勤を修正する。")

    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        width="stretch",
    )

    if st.button("給料を計算"):
        result_df, total_hours, total_pay = calculate_salary(
            edited_df,
            hourly_wage,
        )

        st.subheader("計算結果")

        if result_df.empty:
            st.warning("計算できるシフトがありません。出勤・退勤が 10:00 の形式になっているか確認してください。")
        else:
            st.dataframe(result_df, width="stretch")

        st.metric("実働時間合計", f"{total_hours:.2f} 時間")
        st.metric("給料合計", f"{total_pay:,.0f} 円")
