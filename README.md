# ESS 배터리 수명 예측

초기 100사이클의 열화 신호로 배터리 총수명을 예측하고, 새로운 운전 조건에서 **점검 우선순위와 교체 계획을 신뢰할 수 있는 범위**를 평가한다. 배치별 EDA → 피처 설계 → 정책 단위 모델 선택 → 외부 평가 → 오류 해석을 하나의 흐름으로 구성했다.

**최종 모델:** ΔQ 로그 분산 한 개를 사용하는 **Ridge 회귀(alpha=1, 원수명 타깃)**. 좋은 내부 점수를 새 조건의 신뢰성으로 확대하지 않고, 배치별 오차 방향을 운영 판단 기준으로 연결한다.

## 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 데이터셋 | MIT–Stanford Battery Dataset, Severson et al. (2019) |
| 학습 | Batch 1, 2017-05-12 · 46셀 |
| 평가 | Batch 2, 2018-02-20 · 유효 타깃 39셀 |
| 추가 검증 | Batch 3, 2018-04-12 · 유효 타깃 44셀 |
| 태스크·타깃 | Regression · 원본에 기록된 `cycle_life`(총사이클 수) |
| 입력 시점 | 실제 cycle 2–100, Q100(V)−Q10(V) |

## 핵심 발견

1. **전체 상관과 배치 내 신호는 다르다.** QD2–수명 r은 전체 −0.525에서 배치 중심화 후 −0.013으로 약해졌지만 ΔQ는 −0.773을 유지했다. 절대 상태보다 변화량을 기본 신호로 선정하는 근거다.
2. **복잡도와 추가 입력의 가치는 검증으로 판단한다.** 단일 Ridge 7.12%, 다변량 Ridge 7.37%, RF 9.77%로 개발 평균 오차 개선은 관찰되지 않았다. M2의 낮은 폴드 변동성은 별도 재검토 근거다.
3. **평균 점수와 운영 위험을 함께 본다.** Batch 2/3의 MAE는 159.28/159.41사이클로 비슷하지만 오차 방향은 다르다. 입력이 학습 범위 안인 Batch 2 28셀에서도 MAPE34.90%여서 범위 검사만으로 정확도를 보장하기 어렵다.

[여섯 시사점과 의사결정 기준](docs/insights.md) · [분석 방법·평가 전제](docs/methodology.md) · [EDA·설계 보고서](reports/eda_and_model_strategy.pdf) · [개발·검증 보고서](reports/model_development_evaluation.pdf)

## 파일 구조

```text
ess-battery-project/
├── data/                        # 원본 안내·초기 피처·품질 기록
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/                         # 피처·전처리·학습·오류·시사점
├── results/
│   ├── model_performance.csv    # 주 성능·Gap
│   ├── cell_predictions.csv     # 셀별 예측·오차
│   ├── insight_evidence.csv     # 시사점의 근거·해석 조건
│   └── charts/                 # Batch별 독립 그림
├── reports/                     # 분석 설계·개발 평가 PDF
├── docs/                        # 상세 시사점·방법·검증 기록
├── tests/test_pipeline.py
├── requirements.txt
└── README.md
```

주요 파일을 표시한 구조다. 전체 결과 목록은 [`results/README.md`](results/README.md)에 있다.

## 환경 설정 및 실행

Python 3.11에서 검증했다. 프로젝트 루트에서 실행한다.

```bash
git clone https://github.com/gabin-kim11/ess-battery-project.git
cd ess-battery-project
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m ipykernel install --prefix .venv --name python3 --display-name "Python (ESS battery)"
```

모델 학습·평가는 저장된 피처 CSV만으로 재현할 수 있다.

```bash
python -m src.train --features data/processed/cells_and_features.csv --output results
python -m src.finish --results results
python -m unittest discover -s tests -v
```

노트북은 **01_EDA → 02_feature_engineering → 03_modeling** 순서로 읽는다. 01·02 실행에는 [data/README.md](data/README.md)의 원본 MAT 세 파일을 `data/raw/`에 둔다. 원본이 다른 위치면 `ESS_DATA_DIR`로 지정한다. 03은 저장된 피처 CSV로 실행한다. 실행 출력은 세 노트북에 저장했다. 피처 재현 기준은 `docs/validation/analysis_validation.json`의 스키마·수치 지문으로 보존하고, 품질 규칙은 `data/processed` 한 곳에서 관리한다.

추론·시사점 표·보고서 재생성은 [상세 실행 안내](docs/methodology.md#결과-확인과-보고서-재생성)를 따른다.

## EDA

모든 그래프는 Batch 1·2·3을 분리하고, 동일 지표의 축 범위를 맞춰 비교한다.

| 분석 질문 | 핵심 발견 | 모델 설계 연결 |
|---|---|---|
| Cycle Life 분포 | 중앙값 858.5/472/1005.5사이클. <500 단수명 0/28/0셀 | 내부 점수와 외부 적용 범위를 구분 |
| 방전 용량 열화 | knee 후보 중앙 640/370/850사이클. 초기 용량 증가 중에도 단수명 셀 존재 | 초기 기울기를 비교하고 후반 knee는 진단으로 분리 |
| ΔQ(V) 곡선 | 배치 중심화 후 수명 상관 −0.773 유지 | 변화량 로그 분산을 기준 피처로 선정 |
| 충전 조건·전류 | RMS–수명 r=−0.848/−0.193/−0.684. 평균 전류와 충전 시간은 중복 | 전류·정책 묶음의 증분 가치 검증 |
| 상관·표본 품질 | QD2 통합 상관이 배치 내에서 약해짐. B2 IQR 제거 시 장수명 3→0셀 | 통합 상관·일괄 이상치 제거보다 배치별 신호·표본 보존 |

## Modeling

### 피처 엔지니어링 전략

ΔQ는 실제 Vdlin 전압 축(2–3.5V)에 정렬한 Q100−Q10의 표본분산(ddof=1)을 log10 변환한다. 원본 I는 C-rate, t는 분이다. 종료 용량·knee·전체 관측 길이는 후반 진단으로 구분한다.

ΔQ 최솟값은 로그 분산과 중복이 높아 추가 개발 검증 후보로 남겼다. 이번 비교는 대표 ΔQ 신호에 측정 역할이 다른 입력을 더하는 방식이다.

물리 5피처는 `logvar_deltaQ`, `slope_QD`, `delta_IR`, `mean_Tavg`, `mean_chargetime`이다. 정책 묶음은 `C1/SOC_switch/C2`를 추가하고, 전류 묶음은 충전 시간을 `current_rms_C/current_cv`로 대체한다.

### 후보 모델과 선택 근거

| 후보 | 역할·최선 설정 | CV MAPE(%) | 폴드 SD(%p) |
|---|---|---:|---:|
| M0 중앙값 | 기준 모델 | 19.63 | 10.23 |
| M1 단일 Ridge | ΔQ 1개 · raw · alpha=1 | **7.12** | 3.00 |
| M2 다변량 Ridge | 물리 5 · raw · alpha=0.01 | 7.37 | 1.45 |
| M3 Random Forest | 전류 묶음 · log10 · depth2 · leaf3 | 9.77 | 4.61 |

Batch 1 정책 단위 Hold-out으로 개발 35셀·18정책과 검증 11셀·5정책을 분리한다. 개발 집단의 5-fold GroupKFold에서 65설정을 비교한다. 매 폴드의 결측 대체·표준화는 학습 집단에서 적합한다.

모델 선택은 수명 규모별 상대 오차를 비교하는 MAPE를 주 지표로 확정했다. 설계에서 검토한 MAE는 사이클 단위 오차를 해석하는 보조 지표로 유지했다. 최소 평균 MAPE를 찾고 best 평균+SD/√5 범위에서 Ridge를 우선한다. 같은 계열에서는 최소 평균을 선택한다. **M1 단일 Ridge를 최종 고정**했다. M2의 평균 개선은 관찰되지 않았지만 낮은 변동성은 추가 표본에서 재검토할 근거다. one-SE는 복잡도 제어 규칙이며 통계적 동등성 검정이 아니다.

고정 설정으로 Hold-out 평가 후 전체 Batch 1 46셀에서 재학습해 Batch 2·3을 평가한다. 외부 점수는 후보 선택에 되돌려 사용하지 않는다.

## 성능 결과

MAPE는 상대 오차이며 정확도 백분율이 아니다. Gap은 표기된 앞 값−뒤 값이다.

| 구분 | MAPE(%) / Gap(%p) | 비고 |
|---|---:|---|
| Train (Batch 1 CV) | 7.12 | 5폴드 MAPE 단순 평균 |
| Valid (Batch 1 Hold-out) | 12.72 | 새 정책 11셀 |
| Test (Batch 2) | 32.35 | 39셀 |
| Gap (Train−Valid) | −5.61 | 7.12−12.72 |
| Gap (Valid−Test) | −19.62 | 12.72−32.35 |
| Gap (Target−Test) | **−23.25** | 9.1−32.35 |
| Test (Batch 3) | 12.74 | 44셀 |
| Gap (Batch 2−Batch 3) | +19.61 | 32.35−12.74 |
| Gap (Target−Test, Batch 3) | −3.64 | 9.1−12.74 |

MAPE는 낮을수록 좋으므로 음수 Gap은 뒤 집단의 오차 증가다. Batch 2는 문헌 비교값 9.1%보다 23.25%p 높다. 저자의 Batch 2 파일과 실험 병합·품질 처리가 달라 같은 실험의 엄밀한 재현 점수로 해석하지 않는다. CV 셀 가중 OOF 7.19%와 폴드 단순 평균 7.12%를 구분한다.

## 오류 분석

- **단수명 과대 예측:** Batch 2 35/39셀이 과대 예측이다. 학습 최단 534사이클보다 짧은 30셀은 모두 과대 예측, MAPE37.81%다. 큰 오차 셀 b2c6·b2c15·b2c18은 입력 범위 안이어도 실제 393–449를 678–753사이클로 추정했다.
- **장수명 과소 예측:** Batch 3의 학습 최대 1227사이클 초과 9셀은 모두 과소 예측, MAE450.26사이클이다. b3c38·b3c7·b3c45의 실제 수명은 1801–1935사이클이다.
- **적용 범위:** Batch 2의 입력 범위 안 28/39셀도 MAPE34.90%다. 입력 최소·최대만으로 신뢰하기 어렵다. 실제 수명 구간은 사후 진단이며 운영의 신규 셀 입력이 아니다.
- **원인·개선:** 정책·구조·수집 조건의 차이는 원인 가설이다. EOL 라벨과 단·장수명 개발 표본을 확보하고 정책·구조별 교정을 새 검증 집단에서 판단한다.

## ESS 운영 시사점과 해석 조건

수명 순위 Spearman은 Batch 2=0.709, Batch 3=0.797이다. 점검 우선순위는 예산별 위험 셀 적중률로, 교체 일정은 구간별 bias·과대 예측 손실·예측구간으로 각각 검증한다. 순위 상관을 현장 비용 절감이나 자동 교체 효과로 확대하지 않는다.

Batch 1의 종료 QD가 높은 13셀은 수명 라벨의 완료 여부를 검토해야 한다. 물리 셀ID 해독과 기존 전체 배치 탐색 이력도 평가 전제다. 실제 ESS의 SOC·DoD·온도·캘린더 열화·팩 편차에서 입력 가용성과 오류 비용을 검증한다. [상세 방법·한계](docs/methodology.md)와 [시사점의 근거](docs/insights.md)를 함께 읽는다.

## 참고문헌

- Severson et al. (2019), [Data-driven prediction of battery cycle life before capacity degradation](https://www.nature.com/articles/s41560-019-0356-8), Nature Energy 4, 383–391.
- [저자 데이터 처리 코드](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/LoadData.m).
- [원본 데이터 제공처와 재현 안내](data/README.md).

## 프로젝트 구성

개인 프로젝트. 배치별 EDA·피처 설계·모델 개발·외부 평가·오류 분석과 운영 시사점을 연결했다.
