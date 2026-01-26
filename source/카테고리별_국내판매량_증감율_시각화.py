import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import math

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 1) 카테고리-감미료 매핑 (너가 만든 가설)
# =========================
category_terms = {
    "소스/식사 대용군": ["알룰로스", "사카린나트륨"],
    "디저트/베이커리군": ["에리스리톨", "말티톨시럽"],
    "유제품/빙과군": ["알룰로스", "에리스리톨", "아스파탐"],
    "주류/기호음료군": ["스테비올배당체", "수크랄로스", "에리스리톨", "아스파탐", "효소처리스테비아"],
    "건강/기능성군": ["자일리톨", "이소말트", "폴리글리시톨시럽", "D-소비톨"],
}

# =========================
# 2) 데이터 로드
# =========================
trade_path = "C:\\Users\\melon\\Documents\\카카오톡 받은 파일\\수출입 실적(품목별)_20260122.xlsx"  # 업로드 파일 경로
trade = pd.read_excel(trade_path)

# 컬럼명 공백 정리
trade.columns = [str(c).strip() for c in trade.columns]

# =========================
# 3) 연도 만들기: '기간' -> '연도'
# =========================
if "기간" not in trade.columns:
    raise ValueError(f"'기간' 컬럼이 없습니다. 현재 컬럼: {trade.columns.tolist()}")

trade["연도"] = pd.to_numeric(trade["기간"], errors="coerce")
trade = trade.dropna(subset=["연도"])
trade["연도"] = trade["연도"].astype(int)

# =========================
# 4) 국내판매량 확보
# =========================
if "국내판매량" in trade.columns:
    trade["국내판매량_사용"] = pd.to_numeric(trade["국내판매량"], errors="coerce").fillna(0)
else:
    in_col = "수입 중량(T)"
    out_col = "수출 중량(T)"
    if in_col not in trade.columns or out_col not in trade.columns:
        raise ValueError(f"국내판매량/수입/수출 중량 컬럼을 찾을 수 없습니다.\n현재 컬럼: {trade.columns.tolist()}")
    trade[in_col] = pd.to_numeric(trade[in_col], errors="coerce").fillna(0)
    trade[out_col] = pd.to_numeric(trade[out_col], errors="coerce").fillna(0)
    trade["국내판매량_사용"] = trade[in_col] - trade[out_col]

# 품목명 체크
if "품목명" not in trade.columns:
    raise ValueError(f"'품목명' 컬럼이 없습니다. 현재 컬럼: {trade.columns.tolist()}")

# =========================
# 5) (카테고리, 감미료, 연도)별 국내판매량 + 증감율 만들기
# =========================
rows = []

for cat, terms in category_terms.items():
    for term in terms:
        tmp = trade[trade["품목명"].astype(str).str.contains(re.escape(term), na=False)].copy()
        if tmp.empty:
            continue

        yr = tmp.groupby("연도", as_index=False)["국내판매량_사용"].sum()
        yr = yr.sort_values("연도")
        yr["증감율(%)"] = yr["국내판매량_사용"].pct_change() * 100
        yr["카테고리"] = cat
        yr["감미료"] = term
        rows.append(yr)

df_plot = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

if df_plot.empty:
    raise ValueError("category_terms로 매칭되는 품목이 없습니다. (품목명에 키워드가 포함되는지 확인 필요)")


df_plot = df_plot[df_plot["연도"] >= 2021]

# =========================
# 6) 카테고리별 subplot 그리기
# =========================
categories = list(category_terms.keys())
n = len(categories)

ncols = 2
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 9), sharex=True)
axes = np.array(axes).reshape(-1)

fig.suptitle("식품 카테고리별 주요 감미료 국내판매량 증감율(%) 추이", fontsize=18, fontweight="bold")
years = sorted(df_plot["연도"].unique())
for i, cat in enumerate(categories):
    ax = axes[i]
    sub = df_plot[df_plot["카테고리"] == cat].copy()

    # 카테고리에서 매칭된 감미료만 라인으로
    for sweet in sub["감미료"].unique():
        t = sub[sub["감미료"] == sweet].sort_values("연도")
        ax.plot(t["연도"], t["증감율(%)"], marker="o", label=sweet)

    ax.axhline(0, color="steelblue", linewidth=1)  # 0% 기준선
    ax.set_title(f"[{cat}]", fontsize=13, fontweight="bold")
    ax.set_ylabel("증감율(%)")
    ax.set_xticks(years)

    ax.set_xticklabels(years)
    
    ax.tick_params(axis="x", labelbottom=True)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

# 남는 subplot(빈 칸) 숨김
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
