import re
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
from PIL import Image
import easyocr


HOURLY_WAGE = 1250

# シフト記号の退勤時刻
SHIFT_END_MAP = {
    "L": "21:00",
}


def time_to_minutes(t: str) -> int:
    h, m = map(int, t.split(":"))
    return h * 60 + m


def minutes_to_hours(minutes: int) -> float:
    return minutes / 60


def calc_break_minutes(work_minutes: int) -> int:
    work_hours = work_minutes / 60

    if work_hours >= 10:
        return 90
    elif work_hours >= 8:
        return 60
    elif work_hours >= 6:
        return 45
    else:
        return 0


def extract_times_from_text(text: str):
    """
    OCR結果から 10:00, 13:30 などの時刻を抽出する
    """
    pattern = r"\b\d{1,2}[:：]\d{2}\b"
    times = re.findall(pattern, text)
    times = [t.replace("：", ":") for t in times]
    return times


def extract_dates_from_text(text: str):
    """
    OCR結果から日付らしき数字を抽出する
    """
    nums = re.findall(r"\b\d{1,2}\b", text)
    dates = []

    for n in nums:
        value = int(n)
        if 1 <= value <= 31:
            dates.append(value)

    return dates


def make_shift_table(ocr_text: str):
    times = extract_times_from_text(ocr_text)
    dates = extract_dates_from_text(ocr_text)

    rows = []

    # 時刻は2個ずつ「出勤・退勤」として仮に割り当てる
    pair_count = len(times) // 2

    for i in range(pair_count):
        date = dates[i] if i < len(dates) else None
        start = times[2 * i]
        end = times[2 * i + 1]

        rows.append({
            "日付": date,
            "出勤": start,
            "退勤": end,
        })

    return pd.DataFrame(rows)


def calculate_salary(df: pd.DataFrame, hourly_wage: int):
    result_rows = []

    for _, row in df.iterrows():
        date = row["日付"]
        start = str(row["出勤"])
        end = str(row["退勤"])

        if not start or not end or start == "nan" or end == "nan":
            continue

        start_min = time_to_minutes(start)
        end_min = time_to_minutes(end)

        work_minutes = end_min - start_min
        break_minutes = calc_break_minutes(work_minutes)
        actual_minutes = work_minutes - break_minutes

        pay = actual_minutes / 60 * hourly_wage

        result_rows.append({
            "日付": date,
            "出勤": start,
            "退勤": end,
            "勤務時間[h]": round(minutes_to_hours(work_minutes), 2),
            "休憩時間[h]": round(minutes_to_hours(break_minutes), 2),
            "実働時間[h]": round(minutes_to_hours(actual_minutes), 2),
            "給料[円]": round(pay),
        })

    result_df = pd.DataFrame(result_rows)

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
    "シフト表画像をアップロード",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロード画像", use_container_width=True)

    reader = easyocr.Reader(["ja", "en"], gpu=False)
    ocr_result = reader.readtext(image)

    ocr_text = "\n".join([item[1] for item in ocr_result])

    st.subheader("OCR読み取り結果")
    st.text_area("読み取った文字", ocr_text, height=200)

    df = make_shift_table(ocr_text)

    st.subheader("読み取り後のシフト表")
    st.write("OCRは誤読する場合があるため、必要に応じて修正する。")

    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
    )

    if st.button("給料を計算"):
        result_df, total_hours, total_pay = calculate_salary(
            edited_df,
            hourly_wage,
        )

        st.subheader("計算結果")
        st.dataframe(result_df, use_container_width=True)

        st.metric("実働時間合計", f"{total_hours:.2f} 時間")
        st.metric("給料合計", f"{total_pay:,} 円")