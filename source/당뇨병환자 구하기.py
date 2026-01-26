#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt


# In[9]:


import pandas as pd
import os
import warnings

# 1. 보기 싫은 경고(UserWarning) 메시지 끄기
warnings.filterwarnings('ignore', category=UserWarning)

# 2. 경로 설정
folder_name = 'data7'
diabetes_files = ['당뇨1.xlsx', '당뇨2.xlsx', '당뇨3.xlsx', '당뇨4.xlsx', '당뇨5.xlsx']

total_diabetes = None

print("--- 데이터 통합 시작 ---")

for f in diabetes_files:
    file_path = os.path.join(folder_name, f)

    if os.path.exists(file_path):
        # 엑셀 읽기 (4행부터 데이터 시작)
        df = pd.read_excel(file_path, skiprows=3)

        # '계' - '계' 행 필터링 (가장 상단에 있는 전체 합계 데이터)
        # iloc[:, 1]은 성별구분, iloc[:, 2]는 입원외래구분
        df_sum = df[(df.iloc[:, 1] == '계') & (df.iloc[:, 2] == '계')].copy()

        if not df_sum.empty:
            counts = []
            # 2010년 환자수(인덱스 3)부터 2024년까지 5칸씩 건너뛰며 수집
            for i in range(3, len(df_sum.columns), 5):
                if len(counts) < 15:
                    val = df_sum.iloc[0, i]
                    # 쉼표 제거 및 숫자 변환
                    clean_val = int(str(val).replace(',', '')) if pd.notnull(val) else 0
                    counts.append(clean_val)

            temp_df = pd.DataFrame({'연도': range(2010, 2025), '환자수': counts})

            if total_diabetes is None:
                total_diabetes = temp_df
            else:
                total_diabetes['환자수'] += temp_df['환자수']

            print(f"✅ {f} 합산 완료")
    else:
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")

# 3. 최종 결과 확인
if total_diabetes is not None:
    print("\n--- [최종 결과] 2010-2024 통합 당뇨 환자 수 ---")
    print(total_diabetes)
    # 결과를 엑셀로 저장해두면 나중에 쓰기 편합니다.
    total_diabetes.to_excel('통합_당뇨환자_데이터.xlsx', index=False)
    print("\n'통합_당뇨환자_데이터.xlsx' 파일로 저장되었습니다.")


# In[18]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# 1. 한글 폰트 설정 (윈도우: Malgun Gothic, 맥: AppleGothic)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 2. 경로 및 파일 설정
folder_path = 'data7' # 파일들이 들어있는 폴더 이름
diabetes_files = ['당뇨1.xlsx', '당뇨2.xlsx', '당뇨3.xlsx', '당뇨4.xlsx', '당뇨5.xlsx']
new_hira_file = '건강보험심사평가원_시군구별 성별 연령별 당뇨병 진료 통계 2024.csv'

# 연도 범위 (2010~2024)
years = list(range(2010, 2025))
total_counts = {year: 0 for year in years}

# 3. 기존 당뇨 파일 5개 합산 작업
for f in diabetes_files:
    file_path = os.path.join(folder_path, f)
    if os.path.exists(file_path):
        # 엑셀 읽기 (4행부터 데이터 시작이므로 skiprows=3)
        df = pd.read_excel(file_path, skiprows=3)
        # 성별 '계', 입원외래 '계'인 전체 합계 행만 추출
        row_total = df[(df.iloc[:, 1] == '계') & (df.iloc[:, 2] == '계')]

        if not row_total.empty:
            for idx, year in enumerate(years):
                col_idx = 3 + (idx * 5) # 환자수 컬럼의 위치
                if col_idx < len(row_total.columns):
                    val = row_total.iloc[0, col_idx]
                    # 콤마 제거 후 숫자로 변환
                    clean_val = int(str(val).replace(',', '')) if pd.notnull(val) else 0
                    total_counts[year] += clean_val

df_trend = pd.DataFrame(list(total_counts.items()), columns=['Year', 'Count'])

# 4. 최신 심평원 2024 데이터 불러오기 (검증용)
hira_path = os.path.join(folder_path, new_hira_file)
total_2024_new = 0
if os.path.exists(hira_path):
    # 한글 깨짐 방지를 위해 encoding='cp949' 추가
    df_new = pd.read_csv(hira_path, encoding='cp949')
    df_new['환자수_num'] = pd.to_numeric(df_new['환자수'].astype(str).str.replace(',', '').str.strip(), errors='coerce').fillna(0)
    total_2024_new = df_new['환자수_num'].sum()

# 5. 그래프 시각화
plt.figure(figsize=(14, 8))

# [추세선] 기존 5종 합계 데이터
plt.plot(df_trend['Year'], df_trend['Count'], marker='o', color='#d62728', linewidth=2.5, label='당뇨병 진료 추세 (E10~E14)')

# [비교 포인트] 최신 심평원 통합 데이터
if total_2024_new > 0:
    plt.scatter(2024, total_2024_new, color='#1f77b4', s=300, marker='*', label='2024 심평원 통합 통계', zorder=5)
    # 데이터 포인트 위에 숫자 표시
    plt.text(2024, total_2024_new + 100000, f'{total_2024_new:,.0f}명', 
             ha='center', color='blue', fontweight='bold', fontsize=12)

# 💡 가독성 핵심: Y축 단위를 '만 명'으로 변경
ax = plt.gca()
ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x/10000):,}만'))

# 그래프 스타일링
plt.title('대한민국 당뇨병 환자 추이 및 데이터 교차 검증 (2010-2024)', fontsize=18, pad=20)
plt.xlabel('연도', fontsize=12)
plt.ylabel('환자 수 (만 명)', fontsize=12)
plt.xticks(years)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper left', fontsize=11)

plt.tight_layout()
plt.show()


# In[20]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# 1. 한글 폰트 설정 (윈도우: Malgun Gothic)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 2. 경로 및 파일 설정
folder_path = 'data7'
trend_files = ['당뇨1.xlsx', '당뇨2.xlsx', '당뇨3.xlsx', '당뇨4.xlsx', '당뇨5.xlsx']
hira_2024_file = '건강보험심사평가원_시군구별 성별 연령별 당뇨병 진료 통계 2024.csv'

# 3. 데이터 로드 및 전처리 (2010-2024)
years = list(range(2010, 2025))
total_counts = {year: 0 for year in years}

for f in trend_files:
    path = os.path.join(folder_path, f)
    if os.path.exists(path):
        df = pd.read_excel(path, skiprows=3)
        row = df[(df.iloc[:, 1] == '계') & (df.iloc[:, 2] == '계')]
        if not row.empty:
            for idx, yr in enumerate(years):
                col_idx = 3 + (idx * 5)
                if col_idx < len(row.columns):
                    val = row.iloc[0, col_idx]
                    total_counts[yr] += int(str(val).replace(',', '')) if pd.notnull(val) else 0

df_final = pd.DataFrame(list(total_counts.items()), columns=['Year', 'Count'])

# [데이터 보정] 2024년은 가장 정확한 심평원 전수 조사 데이터로 교체
hira_path = os.path.join(folder_path, hira_2024_file)
if os.path.exists(hira_path):
    df_hira = pd.read_csv(hira_path, encoding='cp949')
    total_2024 = pd.to_numeric(df_hira['환자수'].astype(str).str.replace(',', ''), errors='coerce').sum()
    df_final.loc[df_final['Year'] == 2024, 'Count'] = total_2024

# 4. 전년 대비 증감률(YoY) 계산
df_final['YoY_Rate'] = df_final['Count'].pct_change() * 100

# 5. 고퀄리티 그래프 시각화
fig, ax = plt.subplots(figsize=(16, 9))

# 메인 추세선 및 영역 채우기
ax.plot(df_final['Year'], df_final['Count'], marker='o', markersize=8, color='#C1121F', 
        linewidth=3.5, label='당뇨병 확진 환자수 (심평원)')
ax.fill_between(df_final['Year'], df_final['Count'], color='#C1121F', alpha=0.08)

# 데이터 라벨링 (인원수 & 증감률)
for i, row in df_final.iterrows():
    y, x = row['Count'], row['Year']

    # 1. 환자수 라벨 (상단)
    ax.text(x, y + 120000, f"{int(y/10000):,}만", ha='center', fontweight='bold', 
            fontsize=11, color='#003049')

    # 2. 증감률 라벨 (하단) - 첫 해 제외
    if pd.notnull(row['YoY_Rate']):
        ax.text(x, y - 250000, f"▲{row['YoY_Rate']:.1f}%", ha='center', 
                fontsize=10, color='#780000', fontweight='semibold')

# 그래프 디테일 설정
ax.set_title('대한민국 당뇨병 환자수 변화 추이 (2010-2024 확정 데이터)', fontsize=22, pad=35, fontweight='black')
ax.set_ylabel('총 환자 수 (단위: 만 명)', fontsize=14, labelpad=15)
ax.set_xlabel('연도', fontsize=14, labelpad=10)

# Y축 단위를 '만 명'으로 포맷팅
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x/10000):,}만'))

# 스타일링 (격자 및 테두리 제거)
ax.grid(True, axis='y', linestyle=':', alpha=0.5)
ax.spines[['top', 'right']].set_visible(False)
ax.set_xticks(df_final['Year'])

plt.legend(loc='upper left', fontsize=12, frameon=False)
plt.tight_layout()
plt.show()


# In[ ]:





# In[34]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# 1. 한글 폰트 설정 (윈도우 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 2. 파일 찾기 및 로드 설정
# data7 폴더 내의 파일 후보군 (실제 파일명에 맞춰 수정 가능)
target_files = {
    'Sugar': ['s.xls', 'sugar.xls', '설탕.xls'],
    'Allulose': ['allulose.xls', '알룰로스.xls', 'K-stat 무역통계 - 한국무역협회.xls'],
    'Erythritol': ['erythritol.xls', '에리스리톨.xls', 'K-stat 무역통계 - 한국무역협회 (1).xls'],
    'Stevia': ['stevia.xls', '스테비아.xls', 'K-stat 무역통계 - 한국무역협회 (2).xls']
}

def load_and_clean(label, candidates):
    folder = 'data7'
    for cand in candidates:
        path = os.path.join(folder, cand)
        if os.path.exists(path):
            # 엑셀 파일 읽기 (상단 3행 메타데이터 제외)
            df = pd.read_excel(path, skiprows=3)

            # 컬럼 선택: 0(년월), 7(수입중량 kg), 8(증감률 %)
            df_clean = df.iloc[:, [0, 7, 8]].copy()
            df_clean.columns = ['Year', 'Weight_kg', 'Growth_Rate']

            # 년도 숫자만 추출 ('2024년' -> 2024)
            df_clean['Year'] = df_clean['Year'].astype(str).str.extract('(\d+)').astype(float)
            df_clean = df_clean.dropna(subset=['Year']).astype({'Year': int})

            # 숫자형 변환 및 단위 변환 (kg -> Ton)
            df_clean['Weight_kg'] = pd.to_numeric(df_clean['Weight_kg'], errors='coerce').fillna(0)
            df_clean['Growth_Rate'] = pd.to_numeric(df_clean['Growth_Rate'], errors='coerce').fillna(0)
            df_clean['Weight_T'] = df_clean['Weight_kg'] / 1000

            # 2010~2024 필터링
            return df_clean[(df_clean['Year'] >= 2010) & (df_clean['Year'] <= 2024)].sort_values('Year')

    print(f"경고: {label} 파일을 {folder} 폴더에서 찾을 수 없습니다.")
    return None

# 데이터 로드 실행
dfs = {label: load_and_clean(label, cands) for label, cands in target_files.items()}

# 3. 2x2 그리드 시각화
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 시각화 구성 (라벨, 색상, 위치, 제목)
plot_configs = [
    ('Sugar', '#E63946', axes[0, 0], '설탕 (Sugar - HS 1701)'),
    ('Allulose', '#FFB703', axes[0, 1], '알룰로스 (Allulose - HS 1702)'),
    ('Erythritol', '#219EBC', axes[1, 0], '에리스리톨 (Erythritol - HS 2905)'),
    ('Stevia', '#023047', axes[1, 1], '스테비아 (Stevia - HS 2938)')
]

for label, color, ax, title in plot_configs:
    df = dfs.get(label)
    if df is not None:
        # 메인 차트
        ax.plot(df['Year'], df['Weight_T'], marker='o', color=color, linewidth=3, markersize=8)
        ax.fill_between(df['Year'], df['Weight_T'], color=color, alpha=0.1)

        ax.set_title(title, fontsize=18, fontweight='black', pad=15)
        ax.set_ylabel('수입량 (Ton)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)

        # Y축 콤마 표시
        ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

        # 2024년 데이터 라벨 (중량 + 성장률)
        last = df.iloc[-1]
        ax.annotate(f"{int(last['Weight_T']):,}T\n({last['Growth_Rate']}%↑)", 
                    xy=(last['Year'], last['Weight_T']), 
                    xytext=(0, 12), textcoords='offset points',
                    ha='center', va='bottom', fontweight='bold', color=color, fontsize=12)

        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xticks(range(2010, 2025, 2))
    else:
        ax.text(0.5, 0.5, f'데이터 없음: {label}', ha='center', va='center', fontsize=15)

plt.suptitle('국내 주요 감미료 품목별 수입량 및 성장률 추이 (2010-2024)', fontsize=26, fontweight='black', y=1.02)
plt.tight_layout()
plt.show()


# In[ ]:





# In[43]:


import pandas as pd
import matplotlib.pyplot as plt
import platform

# 1. 한글 폰트 설정 (그래프 글자 깨짐 방지)
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')  # 윈도우
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')    # 맥
else:
    plt.rc('font', family='NanumGothic')    # 리눅스/코랩 등
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 2. 데이터 불러오기 (encoding='cp949' 추가)
try:
    df_2023 = pd.read_csv('data7/2023.csv', encoding='cp949')
    df_2024 = pd.read_csv('data7/2024.csv', encoding='cp949')
except UnicodeDecodeError:
    # cp949로 안 되면 euc-kr로 시도
    df_2023 = pd.read_csv('data7/2023.csv', encoding='euc-kr')
    df_2024 = pd.read_csv('data7/2024.csv', encoding='euc-kr')

# 3. 2023.csv 데이터 전처리
df_2023_clean = df_2023.drop(0).copy()

def clean_currency(x):
    if isinstance(x, str):
        cleaned = x.replace(',', '').replace(' ', '')
        if cleaned == '-' or cleaned == '':
            return 0
        return int(cleaned)
    return x if pd.notnull(x) else 0

col_map = {
    'Unnamed: 6': '2019',
    'Unnamed: 9': '2020',
    'Unnamed: 12': '2021',
    'Unnamed: 15': '2022',
    'Unnamed: 18': '2023'
}
cols = ['연령구분'] + list(col_map.keys())
df_2023_sel = df_2023_clean[cols].rename(columns=col_map)

for year in col_map.values():
    df_2023_sel[year] = df_2023_sel[year].apply(clean_currency)

df_2023_grouped = df_2023_sel.groupby('연령구분')[list(col_map.values())].sum()

# 4. 2024.csv 데이터 전처리
df_2024_grouped = df_2024.groupby('연령구분')['환자수'].sum().to_frame(name='2024')

# 5. 데이터 병합
df_final = df_2023_grouped.join(df_2024_grouped, how='outer')

# 6. 불필요한 연령대 제외 (0~9세, 100세 이상)
ages_to_exclude = ['0~9세', '100세이상', '100세 이상']
df_plot = df_final.drop(index=[a for a in ages_to_exclude if a in df_final.index])
df_plot = df_plot.sort_index()

# 7. 시각화
plt.figure(figsize=(12, 8))
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']

for i, age_group in enumerate(df_plot.index):
    marker = markers[i % len(markers)]
    plt.plot(df_plot.columns, df_plot.loc[age_group], marker=marker, label=age_group)

plt.title('연령별 당뇨병 환자 수 추이 (2019-2024)') # 한글 제목
plt.xlabel('연도')
plt.ylabel('환자 수 (명)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

plt.show() # 그래프 보여주기


# In[48]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import platform

# 1. 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# 2. 데이터 불러오기 및 전처리
def load_and_process():
    # 파일 읽기 (인코딩 처리)
    try:
        df_2023 = pd.read_csv('data7/2023.csv', encoding='utf-8')
        df_2024 = pd.read_csv('data7/2024.csv', encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df_2023 = pd.read_csv('data7/2023.csv', encoding='cp949')
            df_2024 = pd.read_csv('data7/2024.csv', encoding='cp949')
        except UnicodeDecodeError:
            df_2023 = pd.read_csv('data7/2023.csv', encoding='euc-kr')
            df_2024 = pd.read_csv('data7/2024.csv', encoding='euc-kr')

    # 2023년 데이터 정리
    df_2023_clean = df_2023.drop(0).copy()

    def clean_currency(x):
        if isinstance(x, str):
            cleaned = x.replace(',', '').replace(' ', '')
            if cleaned == '-' or cleaned == '': return 0
            return int(cleaned)
        return x if pd.notnull(x) else 0

    col_map = {'Unnamed: 6': '2019', 'Unnamed: 9': '2020', 'Unnamed: 12': '2021', 'Unnamed: 15': '2022', 'Unnamed: 18': '2023'}
    cols = ['연령구분'] + list(col_map.keys())
    df_2023_sel = df_2023_clean[cols].rename(columns=col_map)
    for year in col_map.values():
        df_2023_sel[year] = df_2023_sel[year].apply(clean_currency)

    df_2023_grouped = df_2023_sel.groupby('연령구분')[list(col_map.values())].sum()

    # 2024년 데이터 정리
    df_2024_grouped = df_2024.groupby('연령구분')['환자수'].sum().to_frame(name='2024')

    # 병합
    return df_2023_grouped.join(df_2024_grouped, how='outer')

df_plot = load_and_process()

# 3. 그래프 그리기 (GridSpec 사용)
fig = plt.figure(figsize=(14, 12))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1]) # 2행 2열

# 서브플롯 할당
ax1 = fig.add_subplot(gs[0, :]) # 첫 번째 행 전체 (20~30대)
ax2 = fig.add_subplot(gs[1, 0]) # 두 번째 행 왼쪽 (40대)
ax3 = fig.add_subplot(gs[1, 1]) # 두 번째 행 오른쪽 (50~60대)

# 공통 레이블 함수
def add_labels(ax, x, y, offset_y=10):
    for j, val in enumerate(y):
        ax.annotate(f"{int(val):,}", (x[j], val), textcoords="offset points", xytext=(0, offset_y), ha='center', fontsize=9)

# [상단] 20~30대
ages_top = ['20~29세', '30~39세']
markers = ['o', 's']
for i, age in enumerate(ages_top):
    if age in df_plot.index:
        y_vals = df_plot.loc[age]
        ax1.plot(y_vals.index, y_vals, marker=markers[i], label=age, linewidth=2)
        add_labels(ax1, y_vals.index, y_vals, offset_y=15 if i==0 else -20)

ax1.set_title('20대 & 30대 당뇨 환자 추이', fontsize=14, fontweight='bold')
ax1.set_ylabel('환자 수 (명)')
ax1.legend()
ax1.grid(True, linestyle='--')

# [하단 왼쪽] 40대
if '40~49세' in df_plot.index:
    y_vals = df_plot.loc['40~49세']
    ax2.plot(y_vals.index, y_vals, marker='D', color='green', label='40~49세', linewidth=2)
    add_labels(ax2, y_vals.index, y_vals)

ax2.set_title('40대 당뇨 환자 추이', fontsize=14, fontweight='bold')
ax2.set_ylabel('환자 수 (명)')
ax2.legend()
ax2.grid(True, linestyle='--')

# [하단 오른쪽] 50~60대
ages_older = ['50~59세', '60~69세']
markers_older = ['^', 'v']
colors_older = ['red', 'purple']
for i, age in enumerate(ages_older):
    if age in df_plot.index:
        y_vals = df_plot.loc[age]
        ax3.plot(y_vals.index, y_vals, marker=markers_older[i], color=colors_older[i], label=age, linewidth=2)
        add_labels(ax3, y_vals.index, y_vals, offset_y=15 if i==0 else -20)

ax3.set_title('50대 & 60대 당뇨 환자 추이', fontsize=14, fontweight='bold')
ax3.set_ylabel('환자 수 (명)')
ax3.legend()
ax3.grid(True, linestyle='--')

plt.tight_layout()
plt.show()


# In[50]:


import pandas as pd
import matplotlib.pyplot as plt
import platform

# 1. 한글 폰트 설정 (깨짐 방지)
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# 2. 데이터 불러오기 및 전처리 (여기부터 다시 꼼꼼하게 작성했습니다!)
def load_and_process():
    # 파일 읽기 (인코딩 문제 해결)
    try:
        df_2023 = pd.read_csv('data7/2023.csv', encoding='utf-8')
        df_2024 = pd.read_csv('data7/2024.csv', encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df_2023 = pd.read_csv('data7/2023.csv', encoding='cp949')
            df_2024 = pd.read_csv('data7/2024.csv', encoding='cp949')
        except UnicodeDecodeError:
            df_2023 = pd.read_csv('data7/2023.csv', encoding='euc-kr')
            df_2024 = pd.read_csv('data7/2024.csv', encoding='euc-kr')

    # --- 2023년 데이터 정리 ---
    # 첫 번째 행(헤더 잔여물) 제거
    df_2023_clean = df_2023.drop(0).copy()

    # 쉼표(,) 제거하고 숫자로 변환하는 함수
    def clean_currency(x):
        if isinstance(x, str):
            cleaned = x.replace(',', '').replace(' ', '')
            if cleaned == '-' or cleaned == '': return 0
            return int(cleaned)
        return x if pd.notnull(x) else 0

    # 컬럼 이름 변경 (Unnamed -> 연도)
    col_map = {
        'Unnamed: 6': '2019', 
        'Unnamed: 9': '2020', 
        'Unnamed: 12': '2021', 
        'Unnamed: 15': '2022', 
        'Unnamed: 18': '2023'
    }
    # 필요한 컬럼만 선택
    cols = ['연령구분'] + list(col_map.keys())
    df_2023_sel = df_2023_clean[cols].rename(columns=col_map)

    # 숫자 변환 적용
    for year in col_map.values():
        df_2023_sel[year] = df_2023_sel[year].apply(clean_currency)

    # 연령별 합계 구하기
    df_2023_grouped = df_2023_sel.groupby('연령구분')[list(col_map.values())].sum()

    # --- 2024년 데이터 정리 ---
    df_2024_grouped = df_2024.groupby('연령구분')['환자수'].sum().to_frame(name='2024')

    # --- 데이터 병합 (2019~2024) ---
    df_final = df_2023_grouped.join(df_2024_grouped, how='outer')

    return df_final

# 데이터 로드 실행
df_all = load_and_process()

# 3. 분석용 데이터 가공
# 분석에 불필요한 연령대(0~9세, 100세 이상) 제거
ages_to_exclude = ['0~9세', '100세이상', '100세 이상']
df_analysis = df_all.drop(index=[a for a in ages_to_exclude if a in df_all.index]).copy()

# 증가분(명) 계산
df_analysis['Increase_Num'] = df_analysis['2024'] - df_analysis['2019']

# 증가율(%) 계산
df_analysis['Growth_Rate'] = (df_analysis['Increase_Num'] / df_analysis['2019']) * 100

# 4. 그래프 그리기 (왼쪽: 속도 / 오른쪽: 규모)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# [왼쪽 그래프] 증가율 (%)
sorted_by_rate = df_analysis.sort_values(by='Growth_Rate', ascending=False)
# 1등만 빨간색 강조
colors_rate = ['red' if x == sorted_by_rate['Growth_Rate'].max() else 'skyblue' for x in sorted_by_rate['Growth_Rate']]
bars1 = axes[0].bar(sorted_by_rate.index, sorted_by_rate['Growth_Rate'], color=colors_rate)
axes[0].set_title('증가 속도 (2019 대비 증가율 %)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('증가율 (%)')
# 막대 위에 수치 표시
for bar in bars1:
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

# [오른쪽 그래프] 증가 인원 (명)
sorted_by_num = df_analysis.sort_values(by='Increase_Num', ascending=False)
# 1등만 빨간색 강조
colors_num = ['red' if x == sorted_by_num['Increase_Num'].max() else 'lightgreen' for x in sorted_by_num['Increase_Num']]
bars2 = axes[1].bar(sorted_by_num.index, sorted_by_num['Increase_Num'], color=colors_num)
axes[1].set_title('증가 규모 (2019 대비 늘어난 환자 수)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('명')
# 막대 위에 수치 표시
for bar in bars2:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()

plt.show()

