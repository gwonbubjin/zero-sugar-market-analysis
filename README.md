# 🥤 저당·제로 식품 시장 구조 변화 및 대체감미료 수요 분석
> **커널 어카데미 빅데이터 분석가(BDA) 26기 6조 프로젝트**

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=Pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-ffffff?style=flat-square&logo=Matplotlib&logoColor=black"/>
  <img src="https://img.shields.io/badge/Seaborn-444444?style=flat-square&logo=Seaborn&logoColor=white"/>
</p>

---

### 👥 Team Members
* [cite_start]**팀장:** 이희건 [cite: 3]
* [cite_start]**팀원:** 권법진, 정미주, 최신임, 이수연, 조서현 [cite: 3]

---

## 1. 프로젝트 개요 🔍
[cite_start]최근 2030 세대를 중심으로 **'헬시플레저(Healthy Pleasure)'** 열풍이 불며 저당 식품에 대한 관심이 급증하고 있습니다. [cite: 21, 80] [cite_start]본 프로젝트는 2030 세대의 당뇨 데이터와 식품 산업 트렌드를 연결하여 저당 시장의 성장 가능성을 진단하고 비즈니스 전략을 도출했습니다. [cite: 2, 57, 90]

## 2. 핵심 가설 및 분석 결과 📊

### **가설 1. 2030 건강 위기론과 저당 시장**
> **"2030의 건강 불안이 시장을 키웠는가?"**
* [cite_start]**결과:** 2019~2024년 기준, **20대 당뇨 증가율 91.7%**로 전 연령대 중 압도적 1위 확인 [cite: 63, 64]
* [cite_start]**인사이트:** 당뇨의 저연령화로 인한 실질적 혈당 관리 수요가 시장 성장의 핵심 동력임 [cite: 78, 79]

### **가설 2. 제품 다양화와 원료 수급의 상관관계**
> **"출시 제품이 늘면 첨가물 수입도 바로 늘까?"**
* [cite_start]**결과:** 2021년 이후 제로 제품군은 지속 성장 중이나, 원료 수입량과는 **6개월~1년의 시차** 존재 [cite: 113, 136, 148]
* [cite_start]**한계:** 제품별 배합비 및 응축량 차이로 인한 1:1 대응 분석의 한계 확인 [cite: 277, 278]

### **가설 3. 감미료별 맞춤형 비즈니스 전략**
| 감미료 | 적용 전략 | 분석 근거 |
| :--- | :--- | :--- |
| **알룰로스** | **프리미엄 전략** | [cite_start]고단가임에도 높은 수요와 판매액 기록 [cite: 333, 335] |
| **아스파탐** | **원가 효율 전략** | [cite_start]일반 제품과 동일가 유지, 다량 판매를 통한 이익 구조 [cite: 336, 337] |
| **에리스리톨** | **마케팅 전략** | [cite_start]주류 시장 특성상 가격 경쟁보다 브랜드 이미지 강조 [cite: 342, 343] |

---

## 3. 시행착오 및 문제 해결 💡
* [cite_start]**데이터 사각지대 발견:** '알룰로스'가 식품첨가물이 아닌 **'당류'**로 분류되어 초기 데이터셋에서 누락된 것을 포착 [cite: 359]
* [cite_start]**하이브리드 분석:** 부족한 통계 데이터를 기사 및 산업 보고서를 통한 수동 조사로 보완하여 신뢰도 확보 [cite: 361, 363]
* [cite_start]**지표 고도화:** 단순 수입량 지표의 한계를 극복하기 위해 **'국내 판매량-생산액'** 상관관계로 분석 방향 수정 [cite: 354, 355]

---

## 4. 기술 스택 및 데이터 소스 🛠
* [cite_start]**Environment:** `Python 3.x`, `Jupyter Notebook` [cite: 1]
* **Library:** `Pandas`, `Matplotlib`, `Seaborn`
* [cite_start]**Data Sources:** * 네이버 데이터랩 검색 트렌드 [cite: 27, 34]
  * [cite_start]건강보험심사평가원 만성질환 현황 [cite: 84]
  * [cite_start]식품의약품안전처 식품첨가물 생산 및 수입 통계 [cite: 116, 286]

---

## 📂 프로젝트 구조
```text
[cite_start]├── datasets/           # 원본 및 전처리 데이터셋 [cite: 116]
[cite_start]├── source/             # 시각화 및 분석 파이썬 스크립트 (.py) [cite: 1]
[cite_start]├── reports/            # 최종 분석 보고서 및 발표 PDF [cite: 1]
└── README.md           # 프로젝트 메인 가이드
