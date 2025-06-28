# 🛒 ChurnVision: 이커머스 고객 이탈 예측 프로젝트

> 머신러닝 기반 이탈 예측 + 고객 클러스터링 + Gemini API 마케팅 전략 자동화까지  
> 👉 고객 행동 데이터를 분석하여 실질적인 비즈니스 인사이트로 연결한 프로젝트

---

## 🗂️ 프로젝트 개요

이커머스 플랫폼 고객의 행동 데이터를 기반으로 **이탈 가능성이 높은 고객을 조기 예측**하고,  
KMeans 기반으로 세분화한 고객 클러스터에 대해 **Gemini API를 통해 자동 마케팅 전략을 제안**하는  
**엔드 투 엔드(End-to-End)** AI 기반 분석 시스템을 개발했습니다.

- 고객 이탈 예측 → 클러스터링 → 전략 제안 → 웹 대시보드 시각화
- 데이터 기반 **타겟 마케팅 전략** 및 **고객 생애 가치 향상** 도모

---

## 🧠 사용 기술 스택

| 구분 | 기술 |
|------|------|
| **분석/모델링** | Python, Pandas, Scikit-Learn, XGBoost, LightGBM, GradientBoosting |
| **클러스터링** | RFM 분석, KMeans |
| **시각화** | Seaborn, Matplotlib, Plotly |
| **웹 구현** | Streamlit |
| **AI 전략 자동화** | Google Gemini API |
| **환경 구성** | Git, Pycharm, requirements.txt |

---

## 🧾 주요 기능

- ✅ **이탈 예측 모델 학습 (Gradient Boosting + Threshold 최적화)**
- ✅ **고객 행동 분석 (EDA + 시각화)**
- ✅ **클러스터링 기반 페르소나 마케팅 전략 수립**
- ✅ **Gemini API 연동 → AI 기반 전략 자동 생성**
- ✅ **Streamlit 대시보드 구현 (모델 예측 + 전략 추천)**

---

## 🎯 성능 요약

**최종 모델: `GradientBoostingClassifier (Threshold=0.134)`**

| 지표       | 클래스 0 | 클래스 1 | 평균(F1) | 정확도 |
|------------|----------|----------|----------|---------|
| Precision  | 1.0000   | 0.9645   | 0.9822   |         |
| Recall     | 0.9925   | 1.0000   | 0.9963   |         |
| F1-score   | 0.9962   | 0.9819   | ✅ **0.9891** | ✅ **0.9938** |

---

## 💪 나의 기여

> **End-to-End 흐름을 단독으로 수행한 부분 중심으로 정리**

- **GradientBoosting + Threshold 조정**으로 예측 성능 극대화 (Recall 1.00 달성)
- **Streamlit 전체 화면 구성 및 배포용 인터페이스 설계**
- **Gemini API 연동 → 클러스터 특성 기반 자동 전략 생성 기능 개발**
- **EDA 및 고객 행동 인사이트 분석 (t-test, chi-square 포함)**
- **전체 README 마크다운 정리 및 결과 문서화**

---

## ⚠️ 부족했던 점 & 개선한 내용

| 부족했던 부분 | 개선 방법 |
|---------------|-----------|
| 모델 Threshold가 기본값(0.5)으로 고정되어 이탈자 Recall이 낮았음 | **ROC 커브 기반 최적 Threshold (0.134) 수동 탐색**으로 성능 개선 |
| 클러스터링 후 마케팅 전략이 정적이고 반복적이었음 | **Gemini API 기반 자동 전략 생성 기능 구현 → 각 클러스터 특성 요약 → 실시간 문장 생성** |
| 팀 프로젝트 당시 일부 시각화가 Streamlit에 적절히 반영되지 않음 | **Streamlit 내 Plotly 그래프로 재구현 + 레이아웃 최적화** |
| 모델 비교만 수행하고 스태킹/보팅 모델 성능은 반영 안 됨 | 별도 실험을 통해 **Stacking 성능 미반영 이유 정리 및 GradientBoosting 선택 근거** 기록 |

---

## 🧪 내가 따로 개선한 실험들 (팀 프로젝트 이후 개인 수행)

- 💡 **XGBoost vs LightGBM vs GradientBoosting** 비교 실험  
  → 전체 정확도는 XGBoost가 가장 높았으나, 이탈자 Recall이 가장 높은 **GradientBoosting**을 최종 선택

- 📊 **Threshold tuning 시나리오화**  
  → 여러 Threshold에 따른 Precision/Recall 변화 테이블 시각화 → `0.134` 기준 Recall 1.00 도달 확인

- 🧠 **Gemini API 프롬프트 엔지니어링 개선**  
  → 클러스터 특성 요약값(평균 R, F, M, churn rate 등)을 프롬프트에 구조적으로 삽입 → 전략 문장 품질 향상

---

## 📦 폴더 구조

```bash
ChurnVision/
├── data/                    # CSV 및 전처리된 데이터
├── models/                  # 모델 파일 (.pkl)
├── streamlit_app.py         # 웹 대시보드 실행 파일
├── utils/                   # 헬퍼 함수, 모델 로딩, API 연동 등
├── notebooks/               # EDA 및 실험 노트북
├── images/                  # 시각화 이미지
├── requirements.txt         # 실행 환경 패키지
└── README.md
