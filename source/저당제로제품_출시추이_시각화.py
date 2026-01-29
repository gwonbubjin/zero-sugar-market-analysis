import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
# =========================
# 0) 한글 폰트 설정 (Windows)
# =========================
plt.rcParams["axes.unicode_minus"] = False
try:
    font_path = "C:/Windows/Fonts/malgun.ttf"
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = font_prop.get_name()
except:
    pass  # 폰트 설정 실패해도 실행은 되게

# =========================
# 1) 엑셀 불러오기 (⭐ 이 파일은 header=0이 맞음)
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "제로저당식품 (2).xlsx")

df = pd.read_excel(DATA_PATH)

# 컬럼/문자열 정리
df.columns = df.columns.astype(str).str.replace("\n", "", regex=False).str.strip()
for c in df.columns:
    if df[c].dtype == "object":
        df[c] = df[c].astype(str).str.strip()

# =========================
# 2) 날짜 처리: 출시월 있으면 월별, 없으면 출시연도 기준으로 연도별
# =========================
use_monthly = "출시월" in df.columns

if use_monthly:
    df["출시월"] = pd.to_datetime(df["출시월"], format="%Y-%m", errors="coerce")
    df = df.dropna(subset=["출시월"]).copy()
    df["연도"] = df["출시월"].dt.year
    df["월"] = df["출시월"].dt.month

    group_cols = ["연도", "월"]
    # x축 라벨
    # (연-월 정렬을 위해 월도 포함)
else:
    # 이 파일은 '출시연도'만 있음
    if "출시연도" not in df.columns:
        raise KeyError("파일에 '출시월'도 없고 '출시연도'도 없어요. 컬럼명을 확인해줘!")
    df["연도"] = pd.to_numeric(df["출시연도"], errors="coerce")
    df = df.dropna(subset=["연도"]).copy()
    df["연도"] = df["연도"].astype(int)

    group_cols = ["연도"]
df["대분류"] = (
    df["대분류"]
    .astype(str)
    .str.replace("\u00a0", " ", regex=False)  # 특수 공백
    .str.replace("\r", "", regex=False)
    .str.replace("\n", "", regex=False)
    .str.strip()
)

# 혹시 남아있을 요거트 강제 제거/통합
df.loc[df["대분류"].str.contains("요거트", na=False), "대분류"] = "유제품"

# =========================
# 3) 출시건수 집계
# =========================
monthly = (
    df.groupby(group_cols)
      .size()
      .reset_index(name="출시건수")
      .sort_values(group_cols)
      .reset_index(drop=True)
)

if use_monthly:
    monthly["연월"] = monthly["연도"].astype(str) + "-" + monthly["월"].astype(str).str.zfill(2)
    x_labels = monthly["연월"]
else:
    monthly["연월"] = monthly["연도"].astype(str)  # 그냥 x축 라벨용
    x_labels = monthly["연월"]

# =========================
# 4) 기간별 제품유형(대분류) 요약
# =========================
category_map = (
    df.groupby(group_cols)["대분류"]
      .apply(lambda x: "+".join(sorted(pd.unique(x.dropna()))))
      .reset_index(name="제품유형요약")
)

main_cat = (
    df.groupby(group_cols)["대분류"]
      .agg(lambda s: s.value_counts().idxmax() if len(s.dropna()) else "기타")
      .reset_index(name="대표대분류")
)

monthly = monthly.merge(category_map, on=group_cols, how="left").merge(main_cat, on=group_cols, how="left")

# =========================
# 5) 시각화
# =========================
x = np.arange(len(monthly))
y = monthly["출시건수"].to_numpy()

plt.figure(figsize=(22, 10))
plt.plot(x, y, marker="o", linewidth=2)

# y축 여유
plt.ylim(y.min() - 0.8, y.max() + 1.2)

'''for idx, row in monthly.iterrows():
    cats = str(row["제품유형요약"]).split("+") if pd.notna(row["제품유형요약"]) else [""]

    label = ",\n".join([c for c in cats if c][:10])

    year_val = int(row["연도"])

    # ✅ 글자만 아래로
    if year_val >= 2024:
        y_pos = y[idx] - 0.6   # ← 이 숫자만 조절
        va = "top"
    else:
        y_pos = y[idx] + 0.6
        va = "bottom"

    plt.text(
        x[idx],
        y_pos,
        label,
        ha="center",
        va=va,
        fontsize=10,
        fontweight="bold",
        color="black"
    )'''


plt.xticks(x, x_labels, rotation=45, ha="right", fontsize=9)
plt.xlabel("연-월 (YYYY-MM)" if use_monthly else "연도 (YYYY)", fontsize=10, fontweight="bold")
plt.ylabel("출시 건수", fontsize=10, fontweight="bold")
plt.title("기간별 제로/저당 제품 출시 추이", fontsize=15)
plt.grid(axis="y", alpha=0.3)
plt.subplots_adjust(left=0.12)
plt.show()

print(monthly)   
