#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd

# 1. 파일 불러오기 (K-stat 파일은 보통 skiprows=4 정도가 적당합니다)
# 파일명이 s.xls, a.xls 이므로 각각 불러옵니다.
try:
    s_df = pd.read_excel('data6/s.xls', skiprows=3)
    a_df = pd.read_excel('data6/a.xls', skiprows=3)

    # 2. 필요한 컬럼만 선택 (예: 품목명, 수입금액, 수입증감률 등)
    # 컬럼 이름은 엑셀을 열었을 때 보이는 것과 동일하게 맞춰야 합니다.
    print("--- 설탕 데이터 (s.xls) ---")
    print(s_df.head())

    print("\n--- 알룰로스 데이터 (a.xls) ---")
    print(a_df.head())

except Exception as e:
    print(f"파일을 읽는 중 오류가 발생했습니다. 라이브러리 설치 확인이 필요할 수 있습니다: {e}")


# In[1]:


get_ipython().system('pip install xlrd')


# In[15]:


import pandas as pd
import matplotlib.pyplot as plt

# 2. 데이터 정제 함수 (연도 추출 및 숫자 변환)
def clean_trade_data(df):
    # 필요한 컬럼만 선택 (년월, 수입금액, 수입증감률)
    # 컬럼 위치는 파일 구조에 따라 다를 수 있으니 iloc으로 지정하는 것이 안전합니다.
    df = df.iloc[:, [0, 5, 6]].copy() 
    df.columns = ['Year', 'Imp_Amt', 'Imp_Rate']

    # '년' 글자 제거 및 숫자로 변환
    df['Year'] = df['Year'].str.replace('년', '').astype(int)
    df['Imp_Amt'] = pd.to_numeric(df['Imp_Amt'], errors='coerce')
    df['Imp_Rate'] = pd.to_numeric(df['Imp_Rate'], errors='coerce')

    # 연도 순으로 정렬
    return df.sort_values('Year')

s_clean = clean_trade_data(s_df)
a_clean = clean_trade_data(a_df)

# 3. 그래프 그리기
plt.figure(figsize=(12, 7))
plt.rc('font', family='Malgun Gothic') # 한글 깨짐 방지 (윈도우 기준)

# 첫 번째 축: 설탕 (수입금액)
ax1 = plt.gca()
line1 = ax1.plot(s_clean['Year'], s_clean['Imp_Amt'], color='red', marker='o', label='설탕 수입금액', linewidth=2)
ax1.set_ylabel('설탕 수입금액 (천불)', color='red', fontsize=12)
ax1.tick_params(axis='y', labelcolor='red')

# 두 번째 축: 알룰로스 (수입금액) - 단위 차이가 커서 보조축 사용
ax2 = ax1.twinx()
line2 = ax2.plot(a_clean['Year'], a_clean['Imp_Amt'], color='blue', marker='s', label='알룰로스 수입금액', linewidth=2)
ax2.set_ylabel('알룰로스 수입금액 (천불)', color='blue', fontsize=12)
ax2.tick_params(axis='y', labelcolor='blue')

# 제목 및 범례 설정
plt.title('연도별 설탕 vs 알룰로스 수입 추이 비교', fontsize=15, pad=20)
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

# 격자 추가
ax1.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


# In[19]:


import pandas as pd
import matplotlib.pyplot as plt

# 1. 데이터 로드 및 정제
# 'data6/s.xls' 등 실제 경로에 맞춰 수정하세요.
s_df = pd.read_excel('data6/s.xls', skiprows=3)
a_df = pd.read_excel('data6/a.xls', skiprows=3)

def process_growth_data(df):
    # 년월(0번)과 수입증감률(6번) 추출
    df = df.iloc[:, [0, 6]].copy()
    df.columns = ['Year', 'Growth']
    df['Year'] = df['Year'].str.extract('(\d+)').astype(int)
    df['Growth'] = pd.to_numeric(df['Growth'], errors='coerce')
    return df[df['Year'] >= 2000].sort_values('Year')

s_clean = process_growth_data(s_df)
a_clean = process_growth_data(a_df)

# 2. 한 화면에 두 개의 그래프 그리기 (1열 2행)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 16))
plt.rc('font', family='Malgun Gothic') # 한글 설정

# --- 첫 번째 그래프: 설탕 증감률 ---
ax1.plot(s_clean['Year'], s_clean['Growth'], marker='o', color='red', linewidth=2, label='설탕 증감률(%)')
ax1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5) # 0% 기준선
ax1.set_title('설탕 연도별 수입 증감률 추이 (2000-2025)', fontsize=20, pad=20)
ax1.set_xticks(range(2000, 2026, 1))
ax1.grid(True, linestyle=':', alpha=0.4)

for i, (idx, row) in enumerate(s_clean.iterrows()):
    offset = 18 if i % 2 == 0 else -28
    va = 'bottom' if offset > 0 else 'top'
    ax1.annotate(f"{row['Growth']}%", (row['Year'], row['Growth']),
                 textcoords="offset points", xytext=(0, offset), ha='center', va=va,
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec='red', alpha=0.7))

# --- 두 번째 그래프: 알룰로스 증감률 ---
ax2.plot(a_clean['Year'], a_clean['Growth'], marker='o', color='blue', linewidth=2, label='알룰로스 증감률(%)')
ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax2.set_title('알룰로스 연도별 수입 증감률 추이 (2000-2025)', fontsize=20, pad=20)
ax2.set_xticks(range(2000, 2026, 1))
ax2.grid(True, linestyle=':', alpha=0.4)

for i, (idx, row) in enumerate(a_clean.iterrows()):
    offset = 18 if i % 2 == 0 else -28
    va = 'bottom' if offset > 0 else 'top'
    ax2.annotate(f"{row['Growth']}%", (row['Year'], row['Growth']),
                 textcoords="offset points", xytext=(0, offset), ha='center', va=va,
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec='blue', alpha=0.7))

plt.tight_layout()
plt.show()


# In[18]:


import pandas as pd
import matplotlib.pyplot as plt

# 1. 데이터 로드 및 정제
s_df = pd.read_excel('data6/s.xls', skiprows=3)
a_df = pd.read_excel('data6/a.xls', skiprows=3)

def process_data(df):
    df = df.iloc[:, [0, 5, 6]].copy() # 년월, 수입액, 증감률 추출
    df.columns = ['Year', 'Amount', 'Growth']
    df['Year'] = df['Year'].str.extract('(\d+)').astype(int)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df['Growth'] = pd.to_numeric(df['Growth'], errors='coerce')
    return df[df['Year'] >= 2000].sort_values('Year')

s_clean = process_data(s_df)
a_clean = process_data(a_df)

# 2. 한 화면에 두 개의 그래프 그리기 (1열 2행 구성)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 16)) # 세로로 긴 도화지 준비
plt.rc('font', family='Malgun Gothic')

# --- 첫 번째 그래프: 설탕 수입액 ---
ax1.plot(s_clean['Year'], s_clean['Amount'], marker='o', color='red', linewidth=2, label='설탕 수입액(천불)')
ax1.set_title('설탕 연도별 수입액 추이 (2000-2025)', fontsize=20, pad=20)
ax1.set_xticks(range(2000, 2026, 1))
ax1.grid(True, linestyle=':', alpha=0.5)

for i, (idx, row) in enumerate(s_clean.iterrows()):
    offset = 20 if i % 2 == 0 else -30
    ax1.annotate(f"{int(row['Amount']):,}", (row['Year'], row['Amount']),
                 textcoords="offset points", xytext=(0, offset), ha='center', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec='red', alpha=0.7))

# --- 두 번째 그래프: 알룰로스 수입액 ---
ax2.plot(a_clean['Year'], a_clean['Amount'], marker='o', color='blue', linewidth=2, label='알룰로스 수입액(천불)')
ax2.set_title('알룰로스 연도별 수입액 추이 (2000-2025)', fontsize=20, pad=20)
ax2.set_xticks(range(2000, 2026, 1))
ax2.grid(True, linestyle=':', alpha=0.5)

for i, (idx, row) in enumerate(a_clean.iterrows()):
    offset = 20 if i % 2 == 0 else -30
    ax2.annotate(f"{int(row['Amount']):,}", (row['Year'], row['Amount']),
                 textcoords="offset points", xytext=(0, offset), ha='center', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec='blue', alpha=0.7))

plt.tight_layout() # 그래프 간 간격 자동 조정
plt.show()


# In[16]:


import pandas as pd
import matplotlib.pyplot as plt

# 1. 데이터 불러오기 (사용자님의 파일 경로에 맞게 수정하세요)
s_df_raw = pd.read_excel('data6/s.xls', skiprows=3)
a_df_raw = pd.read_excel('data6/a.xls', skiprows=3)

# 2. 데이터 전처리 함수 (연도와 수입액 추출)
def preprocess_data(df, name):
    # 년월(0번)과 수입액(5번) 컬럼 선택
    df = df.iloc[:, [0, 5]].copy()
    df.columns = ['Year', f'Amt_{name}']
    # '2025년'에서 숫자만 추출
    df['Year'] = df['Year'].str.extract('(\d+)').astype(int)
    # 수입액을 숫자형으로 변환
    df[f'Amt_{name}'] = pd.to_numeric(df[f'Amt_{name}'], errors='coerce')
    return df

# 설탕과 알룰로스 데이터 각각 전처리
s_data = preprocess_data(s_df_raw, 'Sugar')
a_data = preprocess_data(a_df_raw, 'Allulose')

# 3. 데이터 통합 및 대체율 계산
merged = pd.merge(s_data, a_data, on='Year')
merged = merged[merged['Year'] >= 2000].sort_values('Year') # 2000년부터 필터링

# [중요] 대체율(점유율) 계산 공식
# 전체 시장 대비 알룰로스가 차지하는 비중(%)
merged['Total'] = merged['Amt_Sugar'] + merged['Amt_Allulose']
merged['Share'] = (merged['Amt_Allulose'] / merged['Total']) * 100

# 4. 시각화 설정
plt.figure(figsize=(20, 10))
plt.rc('font', family='Malgun Gothic') # 한글 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# 배경 그리드 추가
plt.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

# 막대 그래프: 연도별 점유율 수치
bars = plt.bar(merged['Year'], merged['Share'], color='skyblue', 
               edgecolor='navy', alpha=0.7, label='알룰로스 시장 점유율(%)', zorder=3)

# 선 그래프: 성장 추세 강조
plt.plot(merged['Year'], merged['Share'], color='royalblue', 
         marker='o', markersize=8, linewidth=3, label='교체 추세선', zorder=4)

# 5. 수치 라벨링 및 강조
for i, (idx, row) in enumerate(merged.iterrows()):
    # 2025년 데이터는 특별히 빨간색으로 강조
    color = 'red' if row['Year'] == 2025 else 'black'
    weight = 'bold' if row['Year'] == 2025 else 'normal'

    plt.text(row['Year'], row['Share'] + 0.15, f"{row['Share']:.2f}%", 
             ha='center', va='bottom', fontsize=11, color=color, fontweight=weight)

# 그래프 제목 및 축 설정
plt.title('설탕 대비 알룰로스의 시장 대체율 추이 (2000-2025)', fontsize=22, pad=30, fontweight='bold')
plt.xlabel('연도', fontsize=15, labelpad=15)
plt.ylabel('알룰로스 점유율 (전체 대비 %)', fontsize=15, labelpad=15)
plt.xticks(range(2000, 2026, 1), fontsize=12) # 1년 단위 눈금

# 화살표 및 설명 주석 (2025년 급등 강조)
plt.annotate('2025년 역대 최고 비중(8.05%) 달성', 
             xy=(2025, 8.05), xytext=(2018, 8.5),
             arrowprops=dict(facecolor='black', shrink=0.05, width=2),
             fontsize=14, color='red', fontweight='bold')

plt.legend(fontsize=13, loc='upper left')
plt.tight_layout()

# 6. 이미지 저장
plt.show()

