#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


# In[4]:


df = pd.read_excel('data10/식품첨가물.xlsx')


# In[7]:


df.info()


# In[12]:


df['생산량(T)'] = pd.to_numeric(df['생산량(T)'].replace('-', '0'), errors='coerce').fillna(0)
df['연도'] = df['연도'].astype(int)

# 3. 2023년과 2024년 데이터만 추출하여 피벗
df_filtered = df[df['연도'].isin([2023, 2024])]
pivot_df = df_filtered.pivot_table(index='품목명', columns='연도', values='생산량(T)', aggfunc='sum').reset_index()

# 4. 증감율(%) 계산 함수
def calculate_growth_rate(row):
    p2023 = row[2023]
    p2024 = row[2024]
    if p2023 == 0:
        return 100.0 if p2024 > 0 else 0.0
    return ((p2024 - p2023) / p2023) * 100

pivot_df['증감율(%)'] = pivot_df.apply(calculate_growth_rate, axis=1)

# 5. '아스파탐' 제외 및 정렬
pivot_df_clean = pivot_df[pivot_df['품목명'] != '아스파탐'].sort_values(by='증감율(%)', ascending=False)

# 6. 시각화
plt.figure(figsize=(12, 10))

# 증가/감소에 따른 색상 설정 (파랑/빨강)
colors = ['#3498db' if x >= 0 else '#e74c3c' for x in pivot_df_clean['증감율(%)']]

ax = sns.barplot(data=pivot_df_clean, x='증감율(%)', y='품목명', palette=colors)

# 그래프 장식
plt.title('2023년 대비 2024년 생산량 증감율 (%)', fontsize=16, pad=20)
plt.xlabel('증감율 (%)', fontsize=12)
plt.ylabel('품목명', fontsize=12)
plt.axvline(0, color='black', linewidth=1.5) # 0 기준선 강조
plt.grid(axis='x', linestyle='--', alpha=0.5)

# 막대 끝에 수치 표시
for i, v in enumerate(pivot_df_clean['증감율(%)']):
    ax.text(v + (0.5 if v >= 0 else -0.5), i, f'{v:.1f}%', 
            va='center', ha='left' if v >= 0 else 'right', fontsize=10)

plt.tight_layout()
plt.show()

