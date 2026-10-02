# 데이터 재현

`processed/cells_and_features.csv`는 원본 MAT에서 추출한 139셀의 초기 피처 표다. 원본 MAT는 별도 내려받아 프로젝트의 `data/raw/`에 보관한다. 다음 세 파일이 필요하다:

- 2017-05-12_batchdata_updated_struct_errorcorrect.mat
- 2018-02-20_batchdata_updated_struct_errorcorrect.mat
- 2018-04-12_batchdata_updated_struct_errorcorrect.mat

```bash
python -m src.features --data-dir data/raw --output data/processed/cells_and_features.csv
```

[데이터 제공처](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle). 파일명·크기는 `processed/source_manifest.csv`에서 확인한다. `2018-04-03 varcharge`는 다른 실험 목적의 extra 데이터다.

한 행은 파일·원본 인덱스로 구분한 셀이다. 초기 관측 피처, 타깃, 후반 진단 메타데이터를 함께 보존하지만 모델 입력은 `src/preprocess.py`의 `FEATURE_SETS`로 제한한다. barcode/channel이 MATLAB 객체 참조여서 물리 셀 동일성은 별도 이력 확인이 필요하다.

원본이 다른 폴더에 있으면 노트북 실행 전 `ESS_DATA_DIR`를 해당 경로로 지정한다. 모델링은 저장된 피처 CSV만으로 재현할 수 있고, EDA·피처 재추출은 원본 MAT를 사용한다.


## 저장 파일

| 파일 | 역할 |
|---|---|
| `processed/cells_and_features.csv` | 139셀·31열의 재현용 초기 피처·진단 표 |
| `processed/source_manifest.csv` | 원본 파일명·크기·원본 셀 수 |
| `processed/current_quality.csv` / `current_rejected.csv` | 초기 실제 전류 파형의 단위·구간 품질 검토 |
| `processed/batch3_author_quality_rules.csv` | 저자의 Batch 3 품질 규칙 대응표 |

`raw/`는 내려받은 세 MAT 파일을 두는 로컬 폴더다. 실제 모델 입력의 허용 목록은 `src/preprocess.py`에 정의한다. 공개 피처 스냅샷과 원본 자료의 제공처를 구분하며, 본 저장소가 원본 데이터의 권리나 재배포 조건을 변경하지 않는다.

초기 피처의 재현 기준은 [`docs/validation/analysis_validation.json`](../docs/validation/analysis_validation.json)의 `feature_snapshot`에 저장한다. 별도 기준 CSV 대신 139행·31열 스키마와 수치 지문을 확인한다. 셀 순서를 정렬하고 숫자를 유효 숫자 10자리로 정규화하며, 피처 재추출 노트북은 검증을 통과한 값만 저장한다. 품질 규칙은 `processed/batch3_author_quality_rules.csv` 한 곳에서 읽는다.
