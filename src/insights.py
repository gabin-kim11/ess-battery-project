"""고정 평가 결과를 관찰-해석-판단으로 연결한다. 후보 선택이나 재튜닝에 사용하지 않는다.

학습 수명 범위와 실제 오차는 사후 진단값이다. 운영 입력 피처가 아니다.
"""
from pathlib import Path
import argparse
import pandas as pd


def build_insight_evidence(features, predictions, comparison, quality):
    valid = features[features.cycle_life.notna()].copy()
    train = valid[valid.batch == 1]
    best = comparison.sort_values('CV_MAPE_pct').groupby('candidate').head(1).set_index('candidate')
    rows = []

    def add(key, title, observation, interpretation, decision, limitation, source):
        rows.append({'id': key, '핵심 발견': title, '관찰 근거': observation,
                     '해석': interpretation, '의사결정 시사점': decision,
                     '해석 조건': limitation, '근거 파일': source})

    corr = {}
    for feature in ['QD_cycle2', 'logvar_deltaQ']:
        paired = valid[['batch', 'cycle_life', feature]].dropna()
        centered = paired[['cycle_life', feature]] - paired.groupby('batch')[['cycle_life', feature]].transform('mean')
        corr[feature] = (paired[feature].corr(paired.cycle_life),
                         centered[feature].corr(centered.cycle_life))
    add('I1', '절대 용량보다 변화량에 남는 수명 신호',
        f"QD2-수명 r: 전체 {corr['QD_cycle2'][0]:.3f}, 배치 중심화 {corr['QD_cycle2'][1]:.3f}; "
        f"ΔQ 로그 분산-수명 r: 전체 {corr['logvar_deltaQ'][0]:.3f}, 중심화 {corr['logvar_deltaQ'][1]:.3f}",
        '전체 상관에는 배치 평균 차이가 섞인다. ΔQ의 관계는 배치 평균을 구분한 뒤에도 유지된다.',
        '전체 상관 순위만으로 피처를 고르지 않고, 변화량 신호를 기준으로 정책 그룹 검증에서 추가 입력의 기여를 비교한다.',
        '중심화는 배치 수준 평균을 구분하는 진단이며 운전 조건의 인과 효과를 제거하는 실험이 아니다.',
        'data/processed/cells_and_features.csv')

    m1, m2, m3 = (best.loc[k, 'CV_MAPE_pct'] for k in ['M1', 'M2', 'M3'])
    contribution = ('추가 센서 묶음은 평균 오차를 개선하지 못했다.' if m2 >= m1
                    else '추가 센서 묶음에서 평균 오차 개선이 관찰됐다.')
    add('I2', '센서 추가의 기여와 모델 복잡도의 구분',
        f'M1 단일 Ridge {m1:.2f}%, M2 다변량 Ridge {m2:.2f}%, M3 RF {m3:.2f}%; '
        f"M2 폴드 SD {best.loc['M2','CV_MAPE_sd']:.2f}%p, M1 {best.loc['M1','CV_MAPE_sd']:.2f}%p",
        contribution + ' 평균 오차와 폴드 변동성은 별도 판단 근거다.',
        '추가 센서 수집·비선형 모델 도입은 같은 분할에서의 증분 가치로 판단한다. 낮은 M2 변동성은 추가 표본에서 재검토한다.',
        '개발 35셀의 조건부 비교다. 작은 성능 차이가 통계적으로 유의하거나 다른 배치에서도 유지된다고 단정하지 않는다.',
        'results/candidate_comparison.csv')

    b2, b3 = (predictions[predictions.batch == b] for b in [2, 3])
    mae2, mae3 = b2.error_cycles.abs().mean(), b3.error_cycles.abs().mean()
    add('I3', '평균 상대 오차가 가리는 운영 위험의 방향',
        f'Batch 2/3 MAE {mae2:.2f}/{mae3:.2f}사이클, MAPE {b2.APE_pct.mean():.2f}/{b3.APE_pct.mean():.2f}%; '
        f'Batch 2 과대 예측 {(b2.error_cycles > 0).sum()}/{len(b2)}셀',
        'MAPE의 분모가 되는 실제 수명 규모가 다르다. 낮은 상대 오차만으로 다른 배치의 운영 적합성이 더 좋다고 판단하기 어렵다.',
        '점검 지연 위험은 단수명 과대 예측, 조기 교체 위험은 장수명 과소 예측으로 구분해 수명 구간별 MAE·bias와 오류 비용을 평가한다.',
        '점검 지연·조기 교체는 오차 방향에 따른 가능성이다. 실제 비용과 현장 교체 결과는 측정하지 않았다.',
        'results/cell_predictions.csv; results/evaluation_metrics.csv')

    in_range = b2.logvar_deltaQ.between(train.logvar_deltaQ.min(), train.logvar_deltaQ.max())
    below = b2[b2.cycle_life < train.cycle_life.min()]
    above = b3[b3.cycle_life > train.cycle_life.max()]
    add('I4', '입력 범위 검사만으로 확인하기 어려운 적용 영역',
        f'Batch 2 ΔQ 학습 범위 안 {in_range.sum()}/{len(b2)}셀, 해당 MAPE {b2.loc[in_range,"APE_pct"].mean():.2f}%; '
        f'학습 최단수명 미만 {len(below)}셀 MAPE {below.APE_pct.mean():.2f}%; '
        f'Batch 3 학습 최대수명 초과 {len(above)}셀 MAE {above.error_cycles.abs().mean():.2f}사이클',
        '입력이 익숙한 범위여도 수명과의 대응 관계가 달라질 수 있다. 학습 수명 범위의 양쪽 끝에서 큰 오차가 남는다.',
        '정책·셀 구조·수집 조건을 포함한 새 검증 집단과 단수명·장수명 개발 표본을 확보한다. 입력 범위 안이라는 이유만으로 자동 결정을 허용하지 않는다.',
        '실제 수명 구간은 사후 진단이다. 운영에서는 새 셀의 정답 수명을 미리 알 수 없고, 정책·구조·불확실성 등 관측 가능한 정보가 필요하다.',
        'results/lifetime_range_errors.csv; results/cell_predictions.csv')

    life2 = valid.loc[valid.batch == 2, 'cycle_life']
    q1, q3 = life2.quantile([.25, .75])
    keep = life2.between(q1 - 1.5 * (q3-q1), q3 + 1.5 * (q3-q1))
    original, filtered = quality.iloc[0], quality.iloc[1]
    add('I5', '품질 검토와 예측 대상 축소의 구분',
        f'Batch 2 IQR 제외 표시 {(~keep).sum()}셀; 장수명>1000 {int((life2>1000).sum())}→{int(((life2>1000)&keep).sum())}셀; '
        f'Batch 3 품질 민감도 {int(original.n)}→{int(filtered.n)}셀, MAPE {original.MAPE_pct:.2f}→{filtered.MAPE_pct:.2f}%',
        '일괄 이상치 제거는 중요한 수명 영역을 없앨 수 있다. 저자 품질 규칙을 대응시킨 검토에서도 외부 오차가 크게 줄지 않았다.',
        '품질 문제는 측정 이력으로 판단하고, 정상적인 극단 수명은 검증 대상으로 보존한다. 정제보다 라벨 완료 여부와 표본 범위를 먼저 점검한다.',
        'IQR은 타깃 분포의 진단이며 실제 셀 삭제나 초기 시점의 품질 판정 규칙으로 사용하지 않았다.',
        'data/processed/cells_and_features.csv; results/batch3_quality_sensitivity.csv')

    rank2, rank3 = (g.cycle_life.corr(g.prediction, method='spearman') for g in [b2, b3])
    add('I6', '절대 교체 시점과 상대 점검 순위의 분리',
        f'Batch 2/3 수명 순위 Spearman {rank2:.3f}/{rank3:.3f}',
        '절대 수명에 편향이 남아도 순위 정보는 일부 유지된다. 가능한 활용 목적에 따라 검증 기준이 달라진다.',
        '점검 우선순위의 적용 가능성을 상위 위험 셀 탐지율·예산별 적중률로 별도 검증하고, 교체 자동 결정은 수명 구간별 교정과 현장 검증 이후 판단한다.',
        '순위 상관은 위험 셀 탐지 성능이나 비용 절감의 검증 결과가 아니다. 현장 적용은 별도 실험이 필요하다.',
        'results/cell_predictions.csv')
    return pd.DataFrame(rows)


def save_insight_evidence(root, out):
    root, out = Path(root), Path(out)
    evidence = build_insight_evidence(
        pd.read_csv(root/'data/processed/cells_and_features.csv'),
        pd.read_csv(out/'cell_predictions.csv'),
        pd.read_csv(out/'candidate_comparison.csv'),
        pd.read_csv(out/'batch3_quality_sensitivity.csv'))
    evidence.to_csv(out/'insight_evidence.csv', index=False)
    return evidence


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root', default='.')
    parser.add_argument('--results', default='results')
    args=parser.parse_args()
    evidence=save_insight_evidence(args.root,args.results)
    print(evidence[['id','핵심 발견','관찰 근거']].to_string(index=False))


if __name__=='__main__':
    main()
