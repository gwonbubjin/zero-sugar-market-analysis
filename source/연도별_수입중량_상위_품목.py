#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


df = pd.read_excel('data12/수출입 실적(품목별)_20260122.xlsx')
df.info()


# In[9]:


# 1. 데이터 전처리
df_plot = df[df['기간'] != '총계'].copy()
df_plot['기간'] = pd.to_numeric(df_plot['기간'])
df_plot['수입 중량(T)'] = pd.to_numeric(df_plot['수입 중량(T)'])

# 연도별로 수입 중량 기준 상위 3개 품목 추출
top_3_df = df_plot.sort_values(['기간', '수입 중량(T)'], ascending=[True, False]).groupby('기간').head(3)

# 2. 시각화 설정 (윈도우 한글 폰트)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='Malgun Gothic')

plt.figure(figsize=(14, 8))

# 3. 수동 위치 지정을 통한 막대 그래프 생성 (빈 공간 제거 핵심)
unique_items = top_3_df['품목명'].unique()
colors = sns.color_palette('Set2', len(unique_items))
color_map = dict(zip(unique_items, colors)) # 품목별 고유 색상 매핑

years = sorted(top_3_df['기간'].unique())
x_base = np.arange(len(years))
width = 0.25  # 막대 너비
ax = plt.gca()

for i, year in enumerate(years):
    # 해당 연도 데이터만 추출하여 중량순 정렬
    year_data = top_3_df[top_3_df['기간'] == year].sort_values('수입 중량(T)', ascending=False).reset_index()

    for j, row in year_data.iterrows():
        # j(0,1,2 순위)에 따라 x 좌표를 계산하여 막대를 밀착시킴
        pos = x_base[i] + (j - 1) * width

        ax.bar(
            pos, 
            row['수입 중량(T)'], 
            width=width, 
            color=color_map[row['품목명']], 
            label=row['품목명'],
            edgecolor='white'
        )

        # 막대 위 수치 표시
        ax.text(pos, row['수입 중량(T)'] + 5000, f"{row['수입 중량(T)']:,.0f}", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# 범례 중복 제거 및 설정
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax.legend(unique_labels.values(), unique_labels.keys(), title='품목명', bbox_to_anchor=(1.02, 1), loc='upper left')

# 축 및 제목 설정
ax.set_xticks(x_base)
ax.set_xticklabels([f"{int(y)}년" for y in years], fontsize=11)
plt.title('연도별 수입 중량 상위 3개 품목 현황 (품목명 개별 유지)', fontsize=18, pad=20, fontweight='bold')
plt.xlabel('연도', fontsize=13, fontweight='bold')
plt.ylabel('수입 중량 (T)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()


# In[18]:


# 1. 데이터 전처리
df_clean = df[df['기간'] != '총계'].copy()
df_clean['기간'] = pd.to_numeric(df_clean['기간'])
df_clean['수입 중량(T)'] = pd.to_numeric(df_clean['수입 중량(T)'])

# 증감율 계산을 위해 모든 연도/품목 조합 생성 (0으로 채움)
all_years = sorted(df_clean['기간'].unique())
all_items = df_clean['품목명'].unique()
index = pd.MultiIndex.from_product([all_years, all_items], names=['기간', '품목명'])
df_full = df_clean.set_index(['기간', '품목명']).reindex(index, fill_value=0).reset_index()

# 품목별 전년 대비 증감율(%) 계산: (현재 - 이전) / 이전 * 100
df_full = df_full.sort_values(['품목명', '기간'])
df_full['수입 중량 증감율(%)'] = df_full.groupby('품목명')['수입 중량(T)'].pct_change() * 100

# 2020년 제외 및 무한대(inf, 전년도가 0이었던 경우) 값 제외 후 상위 3개 추출
df_rate = df_full[df_full['기간'] > 2020].copy()
df_rate = df_rate[np.isfinite(df_rate['수입 중량 증감율(%)'])]
top_3_rate = df_rate.sort_values(['기간', '수입 중량 증감율(%)'], ascending=[True, False]).groupby('기간').head(3)

# 2. 시각화 설정 (윈도우 한글 폰트)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='Malgun Gothic')

plt.figure(figsize=(15, 8))

# 품목별 고유 색상 매핑
unique_items = top_3_rate['품목명'].unique()
colors = sns.color_palette('husl', len(unique_items))
color_map = dict(zip(unique_items, colors))

years = sorted(top_3_rate['기간'].unique())
x_base = np.arange(len(years))
width = 0.25
ax = plt.gca()

# 3. 수동 위치 지정을 통한 막대 그래프 생성 (빈 공간 제거)
for i, year in enumerate(years):
    year_data = top_3_rate[top_3_rate['기간'] == year].sort_values('수입 중량 증감율(%)', ascending=False).reset_index()

    for j, row in year_data.iterrows():
        # 순위(j)에 따라 좌표를 밀착시켜 빈 공간 제거
        pos = x_base[i] + (j - 1) * width

        ax.bar(
            pos, 
            row['수입 중량 증감율(%)'], 
            width=width, 
            color=color_map[row['품목명']], 
            label=row['품목명'],
            edgecolor='white'
        )

        # 막대 위 % 수치 표시
        y_label_pos = row['수입 중량 증감율(%)'] + (top_3_rate['수입 중량 증감율(%)'].max() * 0.02)
        ax.text(pos, y_label_pos, f"{row['수입 중량 증감율(%)']:.1f}%", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# 범례 중복 제거
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax.legend(unique_labels.values(), unique_labels.keys(), title='품목명', bbox_to_anchor=(1.02, 1), loc='upper left')

# 축 및 제목 설정
ax.set_xticks(x_base)
ax.set_xticklabels([f"{int(y)}년" for y in years], fontsize=11)
plt.title('연도별 수입 중량 증감율(%) 상위 3개 품목', fontsize=18, pad=25, fontweight='bold')
plt.xlabel('연도 (전년 대비 성장률)', fontsize=13, fontweight='bold')
plt.ylabel('수입 중량 증감율 (%)', fontsize=13, fontweight='bold')

# Y축 상단 여유 공간 추가
plt.ylim(0, top_3_rate['수입 중량 증감율(%)'].max() * 1.15)

plt.tight_layout()
plt.show()


# In[19]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 준비 (아스파탐 추출 및 증감율 계산)
df_aspartame = df[(df['품목명'] == '아스파탐') & (df['기간'] != '총계')].copy()
df_aspartame['기간'] = pd.to_numeric(df_aspartame['기간'])
df_aspartame['수입 중량(T)'] = pd.to_numeric(df_aspartame['수입 중량(T)'])
df_aspartame = df_aspartame.sort_values('기간')
df_aspartame['증감율(%)'] = df_aspartame['수입 중량(T)'].pct_change() * 100
df_plot = df_aspartame.dropna(subset=['증감율(%)'])

# 2. 시각화 스타일 설정 (한글 깨짐 방지 핵심)
sns.set_style("white") # 스타일을 먼저 설정합니다.

# 윈도우 환경 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(10, 7))

# 3. 메인 그래프 (Bar + Line)
colors = ['#FF7675' if x > 0 else '#74B9FF' for x in df_plot['증감율(%)']]
bars = ax.bar(df_plot['기간'].astype(str), df_plot['증감율(%)'], color=colors, alpha=0.8, width=0.5)
ax.plot(df_plot['기간'].astype(str), df_plot['증감율(%)'], color='#2D3436', marker='o', 
        markersize=10, markerfacecolor='white', markeredgewidth=2, linestyle='-', linewidth=2.5)

# 4. 레이아웃 및 텍스트 최적화
y_min, y_max = df_plot['증감율(%)'].min(), df_plot['증감율(%)'].max()
ax.set_ylim(y_min - 10, y_max + 15)
ax.axhline(0, color='#2D3436', linewidth=1.5, alpha=0.8)

for bar in bars:
    height = bar.get_height()
    text_color = '#D63031' if height > 0 else '#0984E3'
    va = 'bottom' if height > 0 else 'top'
    offset = 1.5 if height > 0 else -1.5

    ax.text(bar.get_x() + bar.get_width()/2., height + offset,
             f'{height:.1f}%', ha='center', va=va,
             fontsize=13, fontweight='bold', color=text_color)

# 축 레이블 및 제목 (이때 한글이 적용됩니다)
plt.title('아스파탐 수입 증감율 추이 (2021-2024)', fontsize=20, pad=35, fontweight='bold', color='#2D3436')
ax.set_ylabel('전년 대비 증감율 (%)', fontsize=14, labelpad=15, fontweight='bold', color='#636E72')
ax.set_xlabel('연도', fontsize=14, labelpad=15, fontweight='bold', color='#636E72')

sns.despine(left=True, bottom=False)
ax.yaxis.grid(True, linestyle='--', alpha=0.2)

plt.tight_layout()
plt.show()


# In[31]:


# 1. 데이터 전처리 및 증감율 계산
df_clean = df[df['기간'] != '총계'].copy()
df_clean['기간'] = pd.to_numeric(df_clean['기간'])
df_clean['수입 중량(T)'] = pd.to_numeric(df_clean['수입 중량(T)'])

# 모든 연도/품목 조합 생성 및 증감율 계산
all_years = sorted(df_clean['기간'].unique())
all_items_list = df_clean['품목명'].unique()
index = pd.MultiIndex.from_product([all_years, all_items_list], names=['기간', '품목명'])
df_full = df_clean.set_index(['기간', '품목명']).reindex(index, fill_value=0).reset_index()

df_full = df_full.sort_values(['품목명', '기간'])
df_full['증감률(%)'] = df_full.groupby('품목명')['수입 중량(T)'].pct_change() * 100
df_rate = df_full[(df_full['기간'] > 2020) & (np.isfinite(df_full['증감률(%)']))].copy()

# 연도별 상위 3개 품목 추출
years_list = [2021, 2022, 2023, 2024]
top_items_per_year = {y: df_rate[df_rate['기간'] == y].sort_values('증감률(%)', ascending=False).head(3)['품목명'].tolist() for y in years_list}

# 상위 3위에 한 번이라도 들었던 모든 품목 (전체 차트 및 색상 고정용)
all_top_items = sorted(list(set().union(*top_items_per_year.values())))
df_top_all = df_rate[df_rate['품목명'].isin(all_top_items)]

# 2. 시각화 설정 (한글 깨짐 방지)
sns.set_style("white") 
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# [핵심] 모든 차트에서 품목별로 동일한 색상을 사용하기 위한 글로벌 컬러 맵 생성
global_palette = sns.color_palette('husl', len(all_top_items))
global_color_map = dict(zip(all_top_items, global_palette))

fig = plt.figure(figsize=(22, 26))
gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], wspace=0.45, hspace=0.5)

# --- [상단: 전체 트렌드] ---
ax_top = fig.add_subplot(gs[0, :])
for item in all_top_items:
    data = df_top_all[df_top_all['품목명'] == item]
    ax_top.plot(data['기간'], data['증감률(%)'], marker='o', markersize=10, linewidth=3, 
                label=item, color=global_color_map[item], alpha=0.8)

# 상단 수치 표시 (겹침 방지)
for year in years_list:
    year_data = df_top_all[df_top_all['기간'] == year].sort_values('증감률(%)', ascending=False)
    for idx, (i, row) in enumerate(year_data.iterrows()):
        offset = 4 + (idx * 5)
        ax_top.text(row['기간'], row['증감률(%)'] + offset, f"{row['증감률(%)']:.1f}%", 
                    fontsize=11, ha='center', fontweight='bold', color=global_color_map[row['품목명']],
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

ax_top.set_title('전체 상위 품목(Top 3 경험) 수입 증감률 추이', fontsize=26, pad=40, fontweight='bold')
ax_top.set_xticks(years_list)
ax_top.set_xticklabels([f"{int(y)}년" for y in years_list], fontsize=13)
ax_top.set_ylabel('증감률 (%)', fontsize=16, fontweight='bold')
ax_top.legend(title='품목명', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12, frameon=True, shadow=True)
ax_top.grid(axis='y', linestyle='--', alpha=0.3)

# --- [하단: 연도별 상세 추이 4개] ---
axes_bottom = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), 
               fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]

for i, year in enumerate(years_list):
    ax = axes_bottom[i]
    items = top_items_per_year[year]
    df_sub = df_rate[df_rate['품목명'].isin(items)]

    # 글로벌 컬러 맵을 사용하여 모든 하단 그래프에서도 색상을 일관되게 적용
    for item in items:
        data = df_sub[df_sub['품목명'] == item]
        ax.plot(data['기간'], data['증감률(%)'], marker='o', markersize=10, linewidth=3, 
                label=item, color=global_color_map[item])

    # 숫자 표시 (겹침 방지 및 색상 매칭)
    for y_p in years_list:
        y_d = df_sub[df_sub['기간'] == y_p].sort_values('증감률(%)', ascending=False)
        for idx, (_, row) in enumerate(y_d.iterrows()):
            offset = 2.5 + (idx * 4.5)
            ax.text(row['기간'], row['증감률(%)'] + offset, f"{row['증감률(%)']:.1f}%", 
                    fontsize=11, ha='center', fontweight='bold', color=global_color_map[row['품목명']])

    ax.set_title(f'[{year}년 상위 3개 품목]의 흐름', fontsize=20, fontweight='bold', pad=25)
    ax.set_xticks(years_list)
    ax.set_xticklabels([f"{int(y)}년" for y in years_list], fontsize=11)
    ax.set_ylabel('증감률 (%)', fontsize=13)
    ax.legend(title='품목명', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11, frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    sns.despine(ax=ax)

plt.tight_layout()
plt.subplots_adjust(right=0.85) 
plt.show()


# In[34]:


df_clean = df[df['기간'] != '총계'].copy()
df_clean['기간'] = pd.to_numeric(df_clean['기간'])
df_clean['수입 중량(T)'] = pd.to_numeric(df_clean['수입 중량(T)'])

# 모든 연도/품목 조합 생성 및 증감률 계산
all_years = sorted(df_clean['기간'].unique())
all_items_list = df_clean['품목명'].unique()
index = pd.MultiIndex.from_product([all_years, all_items_list], names=['기간', '품목명'])
df_full = df_clean.set_index(['기간', '품목명']).reindex(index, fill_value=0).reset_index()

df_full = df_full.sort_values(['품목명', '기간'])
df_full['증감률(%)'] = df_full.groupby('품목명')['수입 중량(T)'].pct_change() * 100
df_rate = df_full[(df_full['기간'] > 2020) & (np.isfinite(df_full['증감률(%)']))].copy()

# 카테고리 정의 (데이터셋의 품목명과 매칭)
categories = {
    '[소스/식사 대용군]': ['알룰로스,말티톨시럽'],
    '[디저트/베이커리군]': ['에리스리톨,자일리톨', '알룰로스,말티톨시럽', '스테비올배당체,효소처리스테비아,글리실리진산이나트륨'],
    '[유제품/빙과군]': ['알룰로스,말티톨시럽', '에리스리톨,자일리톨', '아스파탐'],
    '[주류/기호음료군]': ['스테비올배당체,효소처리스테비아,글리실리진산이나트륨', '수크랄로스', '에리스리톨,자일리톨'],
    '[건강/기능성군]': ['에리스리톨,자일리톨', 'D-자일로오스,락티톨,이소말트']
}

# 모든 관련 품목 추출 (색상 고정용)
all_relevant_items = sorted(list(set([item for sublist in categories.values() for item in sublist])))

# 2. 시각화 설정 (윈도우 맑은 고딕)
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 글로벌 컬러 맵 생성 (품목별 색상 고정)
global_palette = sns.color_palette('husl', len(all_relevant_items))
global_color_map = dict(zip(all_relevant_items, global_palette))

fig, axes = plt.subplots(3, 2, figsize=(20, 24))
axes = axes.flatten()

# 5개 카테고리 시각화
years_list = [2021, 2022, 2023, 2024]
for i, (cat_name, items) in enumerate(categories.items()):
    ax = axes[i]
    cat_df = df_rate[df_rate['품목명'].isin(items)]

    for idx, item in enumerate(items):
        item_df = cat_df[cat_df['품목명'] == item]
        if not item_df.empty:
            ax.plot(item_df['기간'], item_df['증감률(%)'], marker='o', markersize=8, linewidth=2.5, 
                    label=item, color=global_color_map[item])

            # 수치 표시 (겹침 방지를 위해 오프셋 조절)
            for _, row in item_df.iterrows():
                ax.text(row['기간'], row['증감률(%)'] + 1.5 + (idx * 2), f"{row['증감률(%)']:.1f}%", 
                        fontsize=10, ha='center', fontweight='bold', color=global_color_map[item])

    ax.set_title(cat_name, fontsize=18, fontweight='bold', pad=15)
    ax.set_xticks(years_list)
    ax.set_xticklabels([f"{int(y)}년" for y in years_list])
    ax.set_ylabel('증감률 (%)', fontsize=12)
    ax.legend(title='품목명', loc='upper left', fontsize=9, bbox_to_anchor=(1, 1))

# 빈 서브플롯 제거
fig.delaxes(axes[5])

plt.suptitle('식품 카테고리별 주요 감미료 수입 증감률(%) 추이', fontsize=24, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 0.9, 0.96])
plt.show()


# In[37]:


# 1. 데이터 전처리 및 증감률 계산
df_clean = df[df['기간'] != '총계'].copy()
df_clean['기간'] = pd.to_numeric(df_clean['기간'])
df_clean['수입 중량(T)'] = pd.to_numeric(df_clean['수입 중량(T)'])

all_years = sorted(df_clean['기간'].unique())
all_items_list = df_clean['품목명'].unique()
index = pd.MultiIndex.from_product([all_years, all_items_list], names=['기간', '품목명'])
df_full = df_clean.set_index(['기간', '품목명']).reindex(index, fill_value=0).reset_index()

df_full = df_full.sort_values(['품목명', '기간'])
df_full['증감률(%)'] = df_full.groupby('품목명')['수입 중량(T)'].pct_change() * 100
df_rate = df_full[(df_full['기간'] > 2020) & (np.isfinite(df_full['증감률(%)']))].copy()

# 카테고리 정의
categories = {
    '[소스/식사 대용군]': ['알룰로스,말티톨시럽'],
    '[디저트/베이커리군]': ['에리스리톨,자일리톨', '알룰로스,말티톨시럽', '스테비올배당체,효소처리스테비아,글리실리진산이나트륨'],
    '[유제품/빙과군]': ['알룰로스,말티톨시럽', '에리스리톨,자일리톨', '아스파탐'],
    '[주류/기호음료군]': ['스테비올배당체,효소처리스테비아,글리실리진산이나트륨', '수크랄로스', '에리스리톨,자일리톨'],
    '[건강/기능성군]': ['에리스리톨,자일리톨', 'D-자일로오스,락티톨,이소말트']
}

all_relevant_items = sorted(list(set([item for sublist in categories.values() for item in sublist])))

# 2. 시각화 설정
sns.set_style("white") 
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 글로벌 컬러 맵 (색상 고정)
global_color_map = dict(zip(all_relevant_items, sns.color_palette('husl', len(all_relevant_items))))

fig, axes = plt.subplots(3, 2, figsize=(20, 24))
axes_flat = axes.flatten()

years_list = [2021, 2022, 2023, 2024]

for i, (cat_name, items) in enumerate(categories.items()):
    ax = axes_flat[i]
    ax.axhline(0, color='#2D3436', linestyle='-', linewidth=1, alpha=0.3) # 0선 강조

    for idx, item in enumerate(items):
        item_df = df_rate[df_rate['품목명'] == item]
        if not item_df.empty:
            ax.plot(item_df['기간'], item_df['증감률(%)'], marker='o', markersize=10, 
                    linewidth=3, label=item, color=global_color_map[item], alpha=0.85)

            # 수치 표시 최적화
            for _, row in item_df.iterrows():
                y_offset = 3 + (idx * 4) # 품목별 텍스트 높이 조절
                ax.text(row['기간'], row['증감률(%)'] + y_offset, f"{row['증감률(%)']:.1f}%", 
                        fontsize=11, ha='center', fontweight='bold', color=global_color_map[item],
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

    # 차트 디자인
    ax.set_title(cat_name, fontsize=20, fontweight='bold', pad=25, color='#2D3436')
    ax.set_xticks(years_list)
    ax.set_xticklabels([f"{int(y)}년" for y in years_list], fontsize=12)
    ax.set_ylabel('증감률 (%)', fontsize=14, fontweight='bold')
    ax.legend(title='품목명', loc='upper left', fontsize=10, bbox_to_anchor=(1.02, 1), frameon=True, shadow=True)

    # 깔끔한 축 정리
    sns.despine(ax=ax, left=True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)

# 마지막 6번째 비어있는 서브플롯 숨기기
fig.delaxes(axes_flat[5])

plt.suptitle('식품 카테고리별 주요 감미료 수입 증감률(%) 추이 분석', fontsize=26, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 0.9, 0.96])
plt.show()


# In[42]:


df.head(13)


# In[46]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 데이터 전처리 및 증감률 계산
df_clean = df[df['기간'] != '총계'].copy()
df_clean['기간'] = pd.to_numeric(df_clean['기간'])
df_clean['수입 중량(T)'] = pd.to_numeric(df_clean['수입 중량(T)'])

# 모든 연도/품목 조합 생성 및 증감률 계산
all_years = sorted(df_clean['기간'].unique())
all_items_list = df_clean['품목명'].unique()
index = pd.MultiIndex.from_product([all_years, all_items_list], names=['기간', '품목명'])
df_full = df_clean.set_index(['기간', '품목명']).reindex(index, fill_value=0).reset_index()

df_full = df_full.sort_values(['품목명', '기간'])
df_full['증감률(%)'] = df_full.groupby('품목명')['수입 중량(T)'].pct_change() * 100
df_rate = df_full[(df_full['기간'] > 2020) & (np.isfinite(df_full['증감률(%)']))].copy()

# 명칭 매칭 (가독성 최적화)
display_name_map = {
    '알룰로스,말티톨시럽': '알룰로스',
    '스테비올배당체,효소처리스테비아,글리실리진산이나트륨': '스테비아',
    '에리스리톨,자일리톨': '에리스리톨/자일리톨',
    'D-자일로오스,락티톨,이소말트': '이소말트/기타'
}

# 5대 카테고리 구성 (요청하신 전 품목 포함)
categories = {
    '[소스/식사 대용군]': ['알룰로스,말티톨시럽', '사카린나트륨', '사카린나트륨제제'],
    '[주류/기호음료군]': ['스테비올배당체,효소처리스테비아,글리실리진산이나트륨', '수크랄로스', '에리스리톨,자일리톨'],
    '[유제품/빙과군]': ['알룰로스,말티톨시럽', '에리스리톨,자일리톨', '아스파탐'],
    '[디저트/베이커리군]': ['에리스리톨,자일리톨', '알룰로스,말티톨시럽', '스테비올배당체,효소처리스테비아,글리실리진산이나트륨'],
    '[건강/기능성군]': ['에리스리톨,자일리톨', 'D-자일로오스,락티톨,이소말트', '폴리글리시톨시럽', 'D-소비톨']
}

all_relevant_items = sorted(list(set([item for sublist in categories.values() for item in sublist])))

# 2. 시각화 설정
sns.set_style("white") 
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 글로벌 컬러 맵 (색상 고정)
global_color_map = dict(zip(all_relevant_items, sns.color_palette('husl', len(all_relevant_items))))

fig, axes = plt.subplots(3, 2, figsize=(20, 26))
axes_flat = axes.flatten()

years_list = [2021, 2022, 2023, 2024]

for i, (cat_name, items) in enumerate(categories.items()):
    ax = axes_flat[i]
    ax.axhline(0, color='#2D3436', linestyle='-', linewidth=1, alpha=0.3)

    for idx, item in enumerate(items):
        item_df = df_rate[df_rate['품목명'] == item]
        if not item_df.empty:
            label_name = display_name_map.get(item, item)
            ax.plot(item_df['기간'], item_df['증감률(%)'], marker='o', markersize=9, 
                    linewidth=2.5, label=label_name, color=global_color_map[item], alpha=0.8)

            # 수치 표시 밀착 조절 (y_offset을 최소화하여 마커에 붙임)
            for _, row in item_df.iterrows():
                # 품목 간 겹침 방지를 위해 미세한 차이만 부여 (밀착도 상승)
                y_offset = 1.5 + (idx * 2.5) 
                ax.text(row['기간'], row['증감률(%)'] + y_offset, f"{row['증감률(%)']:.1f}%", 
                        fontsize=10, ha='center', fontweight='bold', color=global_color_map[item],
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.5))

    ax.set_title(cat_name, fontsize=20, fontweight='bold', pad=25)
    ax.set_xticks(years_list)
    ax.set_xticklabels([f"{int(y)}년" for y in years_list], fontsize=12)
    ax.set_ylabel('증감률 (%)', fontsize=14, fontweight='bold')
    ax.legend(title='품목명', loc='upper left', fontsize=10, bbox_to_anchor=(1.01, 1), frameon=True)

    sns.despine(ax=ax, left=True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.2)

fig.delaxes(axes_flat[5]) # 빈 공간 제거

plt.suptitle('식품 카테고리별 주요 감미료 수입 증감률(%) 통합 분석', fontsize=26, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 0.9, 0.96])
plt.show()

