# 결과 파일 안내

모델 선택과 외부 평가를 재현한 수치 스냅샷이다. 그림은 Batch 1·2·3을 분리하며 같은 지표에 같은 축 범위를 적용한다.

| 읽는 목적 | 파일 |
|---|---|
| 주 성능·Gap | `model_performance.csv`, `evaluation_metrics.csv` |
| 후보 비교·폴드 예측 | `candidate_comparison.csv`, `cv_folds.csv`, `candidate_oof_predictions.csv` |
| 최종 모델·학습 전처리 | `selected_model.json`, `model_artifact.json`, `fitted_preprocessing.json`, `model_coefficients.csv` |
| 검증 분할·고정 프로토콜 | `split_manifest.csv`, `protocol.json`, `run_validation.json` |
| 셀별 예측과 큰 오차 | `cell_predictions.csv`, `worst_cells.csv`, `policy_errors.csv` |
| 수명 범위·분포 진단 | `lifetime_range_errors.csv`, `error_strata.csv`, `cohort_profile.csv` |
| 기준 모델·품질 민감도 | `external_baseline.csv`, `batch3_quality_sensitivity.csv` |
| 시사점의 근거·해석 조건 | `insight_evidence.csv` |
| 탐색적 중요도·그림 계약 | `holdout_permutation_importance.csv`, `chart_contract.json` |
| 배치별 예측·오차·신호 대응 | `charts/` |

검증 기록은 [`docs/validation/`](../docs/validation/)에 있다. 재실행 시 검증 JSON과 EDA 표·그림이 `results/` 아래 추가 생성될 수 있다. `results/eda/`는 Git에서 제외한다. 현재 점수에 맞춘 재튜닝과 새로운 독립 검증을 구분한다.

품질 민감도의 입력 규칙은 [`data/processed/batch3_author_quality_rules.csv`](../data/processed/batch3_author_quality_rules.csv)를 사용한다. 결과 폴더에는 이 입력의 사본을 만들지 않고 민감도 집계만 저장한다.

문헌 기반 확장 후보 M4·M5를 포함한 275설정·1375회 CV 결과는 `candidate_comparison.csv`와 `cv_folds.csv`에 저장한다. 기존 후보의 spec_id S000–S064를 유지했다. 확장 설정·이전 모델 및 점수·평가 열람 이력은 `protocol.json`에 기록했다.
