#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


df = pd.read_excel('data11/kor_zero.xlsx')
df.info()


# In[10]:


# 1. 윈도우 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 전처리 및 증감율 계산 (2020년 0% 기점)
df['연도'] = pd.to_numeric(df['연도'], errors='coerce')
df['국내판매량(T)'] = pd.to_numeric(df['국내판매량(T)'].replace('-', None), errors='coerce')
df = df.sort_values(['품목명', '연도'])
df['판매량_증감율'] = df.groupby('품목명')['국내판매량(T)'].pct_change() * 100
df['판매량_증감율'] = df['판매량_증감율'].fillna(0)

# 2020~2024년 데이터 필터링
df_filtered = df[(df['연도'] >= 2020) & (df['연도'] <= 2024)].copy()

# 품목별로 최대/최소 증감율을 확인하여 그룹 나누기
# 모든 연도에서 증감율이 -100% ~ 100% 사이인 품목들만 추출
item_stats = df_filtered.groupby('품목명')['판매량_증감율'].agg(['max', 'min'])
normal_items = item_stats[(item_stats['max'] <= 100) & (item_stats['min'] >= -100)].index
extreme_items = item_stats[(item_stats['max'] > 100) | (item_stats['min'] < -100)].index

# 2. 2행 1열 시각화
fig, axes = plt.subplots(2, 1, figsize=(15, 16))
sns.set_style("whitegrid")
plt.rc('font', family='Malgun Gothic')

# [위쪽 그래프] 일반 성장 품목 (-100% ~ 100% 이내)
df_normal = df_filtered[df_filtered['품목명'].isin(normal_items)]
sns.lineplot(data=df_normal, x='연도', y='판매량_증감율', hue='품목명', marker='o', ax=axes[0], palette='tab20')
axes[0].set_title('1. 일반 변동 품목 (증감율 -100% ~ 100% 범위)', fontsize=16, pad=15)
axes[0].set_ylabel('판매량 증감율 (%)', fontsize=12)
axes[0].axhline(0, color='red', linestyle='--', alpha=0.5)
axes[0].legend(title='품목명', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)

# [아래쪽 그래프] 급변동 품목 (100% 초과 또는 -100% 미만 포함)
df_extreme = df_filtered[df_filtered['품목명'].isin(extreme_items)]
sns.lineplot(data=df_extreme, x='연도', y='판매량_증감율', hue='품목명', marker='o', ax=axes[1], palette='Set1')
axes[1].set_title('2. 급변동 품목 (수치가 커서 따로 분리)', fontsize=16, pad=15)
axes[1].set_ylabel('판매량 증감율 (%)', fontsize=12)
axes[1].axhline(0, color='red', linestyle='--', alpha=0.5)
axes[1].legend(title='품목명', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)

# 공통 설정
for ax in axes:
    ax.set_xticks(range(2020, 2025))
    ax.set_xlabel('연도', fontsize=12)

plt.tight_layout()
plt.show()


# In[16]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 윈도우 한글 폰트 설정 (맑은 고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# [데이터 전처리]
# 요청하신 특정 품목 리스트 (데이터상 존재하는 이름 기준)
selected_items = [
    '에리스리톨', '말티톨시럽', '효소처리스테비아', 
    '스테비올배당체', '아스파탐', '수크랄로스', '자일리톨', '이소말트'
]

# 수치형 변환 및 필터링
df_sub = df[df['품목명'].isin(selected_items)].copy()
df_sub['연도'] = pd.to_numeric(df_sub['연도'], errors='coerce')
df_sub['국내판매량(T)'] = pd.to_numeric(df_sub['국내판매량(T)'].replace('-', '0'), errors='coerce').fillna(0)

# 증감율 계산 (품목별 그룹화 -> 2020년 0% 기점)
df_sub = df_sub.sort_values(['품목명', '연도'])
df_sub['판매량_증감율'] = df_sub.groupby('품목명')['국내판매량(T)'].pct_change() * 100
df_sub['판매량_증감율'] = df_sub['판매량_증감율'].replace([float('inf'), -float('inf')], 0).fillna(0)

# 2020~2024년 데이터 확정
df_plot = df_sub[(df_sub['연도'] >= 2020) & (df_sub['연도'] <= 2024)].copy()

# 2. 가독성 극대화를 위한 개별 분할 그래프(FacetGrid) 시각화
sns.set_theme(style="whitegrid", font='Malgun Gothic')

# 품목별로 칸을 나누어 그리기 (한 줄에 4개씩 배치)
g = sns.FacetGrid(
    df_plot, 
    col="품목명", 
    col_wrap=4, 
    height=4, 
    aspect=1.2, 
    hue="품목명", 
    palette="viridis",
    sharey=False # 품목마다 수치 폭이 다르므로 각자 최적화된 스케일 적용 (매우 중요)
)

# 메인 그래프: 0% 기준선 + 꺾은선 + 마커
g.map(plt.axhline, y=0, color='black', ls='--', lw=1, alpha=0.3)
g.map(plt.plot, "연도", "판매량_증감율", marker="o", markersize=8, linewidth=3)

# [핵심] 각 포인트마다 증감율(%) 수치 표시
def annotate(data, **kwargs):
    for i, row in data.iterrows():
        if row['연도'] == 2020: continue # 시작점인 2020년은 제외
        val = row['판매량_증감율']
        # 수치가 배경과 겹치지 않게 흰색 박스로 감싸서 표시
        plt.text(row['연도'], val, f"{val:,.1f}%", 
                 ha='center', va='bottom', fontsize=10, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

g.map_dataframe(annotate)

# 디자인 디테일 및 제목 설정
g.set_titles("{col_name}", size=15, fontweight='bold', pad=15)
g.set_axis_labels("연도", "판매량 증감율 (%)", fontsize=12)
g.set(xticks=[2020, 2021, 2022, 2023, 2024])

# 상단 대제목 추가
plt.subplots_adjust(top=0.88, hspace=0.4)
g.fig.suptitle('✨ 주요 품목별 국내 판매량 증감율 추이 (2020-2024)', fontsize=22, fontweight='bold')

plt.show()


# In[17]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 윈도우 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 대상 품목 리스트 필터링
selected_items = [
    '에리스리톨', '말티톨시럽', 'D-말티톨(없음)', 
    '효소처리스테비아', '스테비올배당체', '아스파탐', 
    '수크랄로스', '자일리톨', '이소말트'
]
df_sub = df[df['품목명'].isin(selected_items)].copy()

# 데이터 전처리 및 수치형 변환
df_sub['연도'] = pd.to_numeric(df_sub['연도'], errors='coerce')
df_sub['국내판매량(T)'] = pd.to_numeric(df_sub['국내판매량(T)'].replace('-', None), errors='coerce')

# 품목별/연도별 정렬 후 판매량 증감율 계산 (2020년 0% 기점)
df_sub = df_sub.sort_values(['품목명', '연도'])
df_sub['판매량_증감율'] = df_sub.groupby('품목명')['국내판매량(T)'].pct_change() * 100
df_sub['판매량_증감율'] = df_sub['판매량_증감율'].fillna(0)

# 2020~2024년 데이터 필터링
df_filtered = df_sub[(df_sub['연도'] >= 2020) & (df_sub['연도'] <= 2024)].copy()

# 품목 분류 (일반 vs 급변동)
item_stats = df_filtered.groupby('품목명')['판매량_증감율'].agg(['max', 'min'])
normal_items = item_stats[(item_stats['max'] <= 100) & (item_stats['min'] >= -100)].index
extreme_items = item_stats[(item_stats['max'] > 100) | (item_stats['min'] < -100)].index

# 2. 시각화 (상단 통합 / 하단 2열 구성)
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(2, 2)
sns.set_style("whitegrid")
plt.rc('font', family='Malgun Gothic')

# [상단 전체]
ax1 = fig.add_subplot(gs[0, :])
sns.lineplot(data=df_filtered, x='연도', y='판매량_증감율', hue='품목명', marker='o', markersize=8, linewidth=2, palette='tab10', ax=ax1)
ax1.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax1.set_title('선택 품목 국내 판매량(T) 증감율 추이 (2020년 0% 기준)', fontsize=20, pad=20)
ax1.set_ylabel('판매량 증감율 (%)', fontsize=12)
ax1.set_xticks(range(2020, 2025))
ax1.legend(title='품목명', bbox_to_anchor=(1.01, 1), loc='upper left')

# [하단 왼쪽 - 일반 품목]
ax2 = fig.add_subplot(gs[1, 0])
df_normal = df_filtered[df_filtered['품목명'].isin(normal_items)]
if not df_normal.empty:
    sns.lineplot(data=df_normal, x='연도', y='판매량_증감율', hue='품목명', marker='o', ax=ax2, palette='tab10')
ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
ax2.set_title('1. 일반 변동 품목 (-100% ~ 100%)', fontsize=16)
ax2.set_ylabel('판매량 증감율 (%)', fontsize=12)
ax2.set_xticks(range(2020, 2025))
ax2.legend(title='품목명', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)

# [하단 오른쪽 - 급변동 품목]
ax3 = fig.add_subplot(gs[1, 1])
df_extreme = df_filtered[df_filtered['품목명'].isin(extreme_items)]
if not df_extreme.empty:
    sns.lineplot(data=df_extreme, x='연도', y='판매량_증감율', hue='품목명', marker='o', ax=ax3, palette='Set1')
ax3.axhline(0, color='red', linestyle='--', alpha=0.5)
ax3.set_title('2. 급변동 품목 (수치 과다)', fontsize=16)
ax3.set_ylabel('판매량 증감율 (%)', fontsize=12)
ax3.set_xticks(range(2020, 2025))
ax3.legend(title='품목명', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)

plt.tight_layout()
plt.show()

