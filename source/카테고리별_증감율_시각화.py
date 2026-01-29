import os
import pandas as pd
import matplotlib.pyplot as plt
import unicodedata
import re

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 1) 데이터 로드
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "저당.제로식품_대분류_카테고리추가.xlsx")

df = pd.read_excel(DATA_PATH)

# =========================
# 2) 필수 컬럼 체크
# =========================
required_cols = ["출시연도", "분류"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"필수 컬럼이 없습니다: {missing}\n현재 컬럼: {list(df.columns)}")

# =========================
# 3) 분류 정리 + 오타 통합(가능성 -> 기능성)
# =========================
def normalize_category(s):
    if pd.isna(s):
        return ""
    s = str(s)

    # 유니코드 정규화 + 보이지 않는 문자 제거
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in s if not unicodedata.category(ch).startswith("C"))

    # 구분자 통일(. · 등 -> /)
    s = s.replace("·", "/").replace(".", "/").replace("／", "/").replace("∕", "/").replace("⁄", "/")
    s = re.sub(r"\s*/\s*", "/", s)

    # 공백 정리
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()

    return s

df["분류_정리"] = df["분류"].apply(normalize_category)

# 오타 강제 통일
df["분류_정리"] = df["분류_정리"].replace({
    "건강/가능성군": "건강/기능성군",
    "건강가능성군": "건강/기능성군",
    "건강기능성군": "건강/기능성군",
})

# =========================
# 4) 연도 전처리
# =========================
df["출시연도"] = pd.to_numeric(df["출시연도"], errors="coerce")
df = df.dropna(subset=["출시연도", "분류_정리"])
df["출시연도"] = df["출시연도"].astype(int)

# =========================
# 5) 연도 × 카테고리 "출시건수" 집계
#    (전년 대비 증감율은 '연간 출시건수'를 기준으로 계산하는 게 가장 직관적)
# =========================
year_cnt = (
    df.groupby(["출시연도", "분류_정리"])
      .size()
      .reset_index(name="연간출시건수")
      .sort_values(["분류_정리", "출시연도"])
)

# =========================
# 6) 전년 대비 증감율(%) 계산: (올해-전년)/전년 * 100
# =========================
year_cnt["전년대비증감율(%)"] = (
    year_cnt.groupby("분류_정리")["연간출시건수"]
            .pct_change() * 100
)

# ✅ 선택지
# A) 정석: 2021년은 NaN 그대로 두기 (추천)
# (아무것도 안 하면 됨)

# B) 그래프용: 2021년을 0%로 표시하고 싶으면 아래 한 줄 켜기
# year_cnt["전년대비증감율(%)"] = year_cnt["전년대비증감율(%)"].fillna(0)

# =========================
# 7) 그래프: 2021년부터 연도별 전년 대비 증감율
# =========================
plt.figure(figsize=(12, 6))

for cat in year_cnt["분류_정리"].unique():
    tmp = year_cnt[year_cnt["분류_정리"] == cat]
    plt.plot(tmp["출시연도"], tmp["전년대비증감율(%)"], marker="o", label=cat)

plt.title("카테고리별 연간 출시건수 전년 대비 증감율(%) (2021년부터)")
plt.xlabel("출시연도")
plt.ylabel("증감율 (%)")
plt.grid(alpha=0.3)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

# =========================
# 8) (옵션) 결과 테이블 확인
# =========================
print(year_cnt.head(20))
