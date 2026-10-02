"""배치별 EDA와 일치하는 MAT 피처 추출. 모델 입력은 cycle 2..100과 정책이다.
전체 궤적 메타데이터는 진단에만 사용한다. 원본에 쓰지 않는다."""
from pathlib import Path
import argparse,re,hashlib,json
import h5py
import numpy as np
import pandas as pd
BATCH_FILES={1:"2017-05-12_batchdata_updated_struct_errorcorrect.mat",2:"2018-02-20_batchdata_updated_struct_errorcorrect.mat",3:"2018-04-12_batchdata_updated_struct_errorcorrect.mat"}
NOMINAL_AH=1.1
EOL_NEAR_AH=.885
QD_UPPER_AH=1.5
EARLY_START,EARLY_END=2,100

def feature_fingerprint(frame, significant_digits=10):
    """셀 순서를 정렬하고 유효 숫자를 정규화한 재현 검증 지문.

    CSV 읽기·저장의 미세한 부동소수점 차이를 구분하지 않는다.
    10자리 정규화는 오차 허용 구간 검정과는 다른 규칙이다.
    """
    if 'cell_id' not in frame or not frame.cell_id.is_unique:
        raise ValueError('피처 지문에는 고유한 cell_id가 필요합니다.')
    ordered = frame.sort_values('cell_id')
    numeric = [pd.api.types.is_numeric_dtype(ordered[c]) for c in ordered.columns]
    rows = []
    for row in ordered.itertuples(index=False, name=None):
        rows.append([
            None if pd.isna(value) else
            format(float(value), f'.{significant_digits}g') if is_numeric else str(value)
            for value, is_numeric in zip(row, numeric)
        ])
    payload = json.dumps({'columns': list(ordered.columns), 'rows': rows},
                         ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()

def validate_feature_snapshot(frame, snapshot):
    """별도 기준 CSV 없이 저장된 스키마·행 수·수치 지문을 검증한다."""
    if len(frame) != snapshot['rows'] or list(frame.columns) != snapshot['columns']:
        raise ValueError('기준 피처의 행 수 또는 열 구성이 달라졌습니다.')
    fingerprint = feature_fingerprint(frame, snapshot['numeric_significant_digits'])
    if fingerprint != snapshot['fingerprint_sha256']:
        raise ValueError('초기 피처 값이 기준 지문과 다릅니다. 추출 조건과 원본을 확인하세요.')
    return fingerprint

def vector(dataset):
    return np.asarray(dataset[()], dtype=float).reshape(-1)

def decode_text(dataset):
    return "".join(chr(int(v)) for v in dataset[()].reshape(-1))

def read_batch(batch_no, path):
    """원본 summary와 필요한 초기 곡선만 읽습니다. 원본에는 쓰지 않습니다."""
    cell_rows, summary_frames, curves = [], [], {}
    with h5py.File(path, "r") as handle:
        batch = handle["batch"]
        for index in range(batch["summary"].shape[0]):
            cell_id = f"b{batch_no}c{index}"
            summary_group = handle[batch["summary"][index, 0]]
            curve_group = handle[batch["cycles"][index, 0]]
            summary = {k: vector(summary_group[k]) for k in
                       ["cycle", "QDischarge", "QCharge", "IR", "Tavg", "Tmax", "Tmin", "chargetime"]}
            if len({len(v) for v in summary.values()}) != 1:
                raise ValueError(f"요약 배열 길이 불일치: {cell_id}")
            cycle = summary["cycle"]
            if not np.all(np.diff(cycle) > 0):
                raise ValueError(f"cycle 라벨 순서/중복 확인 필요: {cell_id}")
            life = float(vector(handle[batch["cycle_life"][index, 0]])[0])
            policy = decode_text(handle[batch["policy_readable"][index, 0]])
            qd = summary["QDischarge"]
            usable = (cycle >= 2) & np.isfinite(qd) & (qd > 0) & (qd < QD_UPPER_AH)
            tail = qd[usable][-5:]
            cell_rows.append({
                "cell_id": cell_id, "batch": batch_no, "cell_index": index,
                "cycle_life": life, "charging_policy": policy,
                "summary_cycles": len(cycle),
                "QD_bad_rows": int((~np.isfinite(qd) | (qd <= 0) | (qd >= QD_UPPER_AH)).sum()),
                "tail_median_QD": float(np.median(tail)) if len(tail) else np.nan,
                "eol_near": bool(len(tail) and np.median(tail) <= EOL_NEAR_AH),
            })
            frame = pd.DataFrame(summary).rename(columns={"QDischarge": "QD", "QCharge": "QC"})
            frame["cell_id"], frame["batch"], frame["cycle_life"] = cell_id, batch_no, life
            summary_frames.append(frame)
            voltage = vector(handle[batch["Vdlin"][index, 0]])
            q_curves = {}
            for actual_cycle in [10, 100]:
                matches = np.flatnonzero(cycle == actual_cycle)
                if len(matches) != 1:
                    raise ValueError(f"cycle {actual_cycle} 라벨 확인 필요: {cell_id}")
                q_curves[actual_cycle] = vector(handle[curve_group["Qdlin"][int(matches[0]), 0]])
            if not all(len(q) == len(voltage) for q in q_curves.values()):
                raise ValueError(f"전압과 Qdlin 길이 불일치: {cell_id}")
            order = np.argsort(voltage)
            if not np.all(np.diff(voltage[order]) > 0):
                raise ValueError(f"전압 축 중복 확인 필요: {cell_id}")
            curves[cell_id] = {"V": voltage[order], "Q10": q_curves[10][order], "Q100": q_curves[100][order]}
    return pd.DataFrame(cell_rows), pd.concat(summary_frames, ignore_index=True), curves
# Reused verbatim as a notebook code cell.
CURRENT_START, CURRENT_END = EARLY_START, EARLY_END
HIGH_CURRENT_C = 4.5
CURRENT_MAX_DURATION_MIN=60
CURRENT_INTEGRAL_TOLERANCE=.05
FAST_CHARGE_AH = .8 * NOMINAL_AH
CURRENT_INPUT_FIELDS = ['current_mean_C', 'current_rms_C', 'current_cv', 'high_current_charge_fraction']

def charge_pattern(t, current_c, qc):
    """0~공칭80% 충전의 시간 가중 통계. 원본 I는 C-rate, t는 분입니다."""
    t, current_c, qc = map(lambda a: np.asarray(a, dtype=float), (t, current_c, qc))
    if not (len(t)==len(current_c)==len(qc)) or len(t)<3:
        return None
    neg = np.flatnonzero(current_c < -.05)
    limit = int(neg[0]) if len(neg) else len(t)
    crossing = np.flatnonzero((qc[:limit]>=FAST_CHARGE_AH)&np.isfinite(qc[:limit]))
    if not len(crossing):
        return None
    end = int(crossing[0])
    t, current_c, qc = t[:end+1].copy(), current_c[:end+1].copy(), qc[:end+1].copy()
    if not np.all(np.isfinite(t)&np.isfinite(current_c)&np.isfinite(qc)) or np.any(np.diff(t)<-1e-8):
        return None
    # 마지막 구간의 0.88Ah 교차 시점을 선형 보간합니다.
    if end>0 and qc[-1]>qc[-2] and qc[-2]<FAST_CHARGE_AH:
        fraction=(FAST_CHARGE_AH-qc[-2])/(qc[-1]-qc[-2])
        t[-1]=t[-2]+fraction*(t[-1]-t[-2])
        current_c[-1]=current_c[-2]+fraction*(current_c[-1]-current_c[-2])
        qc[-1]=FAST_CHARGE_AH
    dt=np.diff(t)
    keep=dt>1e-8
    dt=dt[keep]
    c0=np.maximum(current_c[:-1][keep],0);c1=np.maximum(current_c[1:][keep],0)
    cm=(c0+c1)/2
    duration=dt.sum()
    if duration<=0 or not len(dt):return None
    integral=np.sum(cm*dt)
    mean=integral/duration
    # 구간 내 선형 전류의 제곱 적분은 (c0²+c0*c1+c1²)/3입니다.
    square_integral=np.sum((c0*c0+c0*c1+c1*c1)/3*dt)
    rms=np.sqrt(square_integral/duration)
    cv=np.sqrt(max(rms*rms-mean*mean,0))/mean if mean>0 else np.nan
    high=np.sum(cm*dt*(cm>HIGH_CURRENT_C))/integral if integral>0 else np.nan
    # dQc/dt와 원본 I를 독립 비교하여 C-rate 정규화 및 시간 단위를 점검합니다.
    dq=np.diff(qc)[keep]
    stable=(cm>1)&(dt>1e-4)&(dq>0)&(dq<.05)&(np.abs(c1-c0)<.02)
    unit_ratio=np.median(dq[stable]*60/(cm[stable]*dt[stable])) if stable.any() else np.nan
    return {'current_mean_C':mean,'current_rms_C':rms,'current_cv':cv,
            'high_current_charge_fraction':high,'fast_charge_duration_min':duration,
            'current_unit_ratio_A_per_C':unit_ratio,
            'charge_integral_ratio':NOMINAL_AH*integral/60/(qc[-1]-qc[0]),
            'zero_time_intervals':int((~keep).sum()),
            't':t,'current_c':current_c}

def extract_features(data_dir):
    DATA_DIR=Path(data_dir)
    cell_parts, summary_parts, curve_records, provenance = [], [], {}, []
    for batch_no, name in BATCH_FILES.items():
        path = DATA_DIR / name
        batch_cells, batch_summary, batch_curves = read_batch(batch_no, path)
        cell_parts.append(batch_cells)
        summary_parts.append(batch_summary)
        curve_records.update(batch_curves)
        provenance.append({"batch": batch_no, "file": name, "bytes": path.stat().st_size,
                           "raw_cells": len(batch_cells)})
        print(f"Batch {batch_no}: {len(batch_cells)}셀, {len(batch_summary):,}사이클 행 로딩 완료")
    cells = pd.concat(cell_parts, ignore_index=True)
    df = pd.concat(summary_parts, ignore_index=True)
    assert cells.cell_id.is_unique
    assert not df.duplicated(["cell_id", "cycle"]).any()
    valid_targets = cells.dropna(subset=["cycle_life"]).copy()
    eol_targets = valid_targets[valid_targets.eol_near].copy()


    delta_rows = []
    for cell_id, curve in curve_records.items():
        delta = curve["Q100"] - curve["Q10"]
        curve["delta"] = delta
        valid_curve = np.isfinite(delta).all()
        delta_rows.append({"cell_id": cell_id, "curve_valid": valid_curve,
            "logvar_deltaQ": np.log10(max(np.var(delta, ddof=1), 1e-15)) if valid_curve else np.nan,
            "min_deltaQ": np.min(delta) if valid_curve else np.nan,
            "mean_deltaQ": np.mean(delta) if valid_curve else np.nan})
    delta_features = pd.DataFrame(delta_rows)
    cells_delta = cells.merge(delta_features, on="cell_id", validate="one_to_one")

    policy_rows = []
    for row in cells.itertuples():
        match = re.search(r"([\d.]+)C\(([\d.]+)%\)-([\d.]+)C", row.charging_policy)
        values = list(map(float, match.groups())) if match else [np.nan] * 3
        policy_rows.append({"cell_id": row.cell_id, "C1": values[0], "SOC_switch": values[1], "C2": values[2]})
    policy_features = pd.DataFrame(policy_rows)

    slope_rows = []
    for cell_id, group in df.groupby("cell_id", sort=False):
        early = group[group.cycle.between(EARLY_START, EARLY_END)]
        keep = np.isfinite(early.QD) & (early.QD > 0) & (early.QD < QD_UPPER_AH)
        slope = np.polyfit(early.loc[keep, "cycle"], early.loc[keep, "QD"], 1)[0] if keep.sum() > 2 else np.nan
        slope_rows.append({"cell_id": cell_id, "slope_QD": slope})
    qd_slopes = pd.DataFrame(slope_rows)
    cells_charge = cells_delta.merge(policy_features, on="cell_id", validate="one_to_one").merge(qd_slopes, on="cell_id", validate="one_to_one")

    current_rows, current_waveforms = [], {}
    current_failed=[]
    for batch_no,name in BATCH_FILES.items():
        with h5py.File(DATA_DIR/name,'r') as handle:
            batch=handle['batch']
            for index in range(batch['summary'].shape[0]):
                cell_id=f'b{batch_no}c{index}'
                summary=handle[batch['summary'][index,0]]
                cycles=vector(summary['cycle'])
                curve_group=handle[batch['cycles'][index,0]]
                for actual_index in np.flatnonzero((cycles>=CURRENT_START)&(cycles<=CURRENT_END)):
                    cycle=int(cycles[actual_index])
                    arrays={k:vector(handle[curve_group[k][actual_index,0]]) for k in ['I','t','Qc']}
                    pattern=charge_pattern(arrays['t'],arrays['I'],arrays['Qc'])
                    if pattern is None:
                        current_failed.append({'cell_id':cell_id,'batch':batch_no,'cycle':cycle,'reason':'invalid_time_or_missing_80pct_endpoint'})
                        continue
                    if pattern['fast_charge_duration_min']>CURRENT_MAX_DURATION_MIN or abs(pattern['charge_integral_ratio']-1)>CURRENT_INTEGRAL_TOLERANCE:
                        current_failed.append({'cell_id':cell_id,'batch':batch_no,'cycle':cycle,'reason':'duration_or_charge_integral_inconsistent',
                                               'duration_min':pattern['fast_charge_duration_min'],'integral_ratio':pattern['charge_integral_ratio']})
                        continue
                    if cycle==10:current_waveforms[cell_id]={'t':pattern['t'],'current_c':pattern['current_c']}
                    current_rows.append({'cell_id':cell_id,'batch':batch_no,'cycle':cycle,
                                         **{k:v for k,v in pattern.items() if k not in ['t','current_c']}})
    current_cycles=pd.DataFrame(current_rows)
    current_rejected=pd.DataFrame(current_failed)
    current_features=current_cycles.groupby('cell_id').agg(
        **{f:(f,'mean') for f in CURRENT_INPUT_FIELDS},
        fast_charge_duration_min=('fast_charge_duration_min','mean'),
        current_cycle_n=('cycle','size'),
        current_unit_ratio_A_per_C=('current_unit_ratio_A_per_C','median'),
        charge_integral_ratio=('charge_integral_ratio','median')).reset_index()
    cells_charge=cells_charge.merge(current_features,on='cell_id',how='left',validate='one_to_one')
    current_quality=current_cycles.groupby('batch').agg(
        cell_n=('cell_id','nunique'),cycle_n=('cycle','size'),
        min_cycle=('cycle','min'),max_cycle=('cycle','max'),
        median_unit_ratio=('current_unit_ratio_A_per_C','median'),
        median_charge_ratio=('charge_integral_ratio','median'),
        min_charge_ratio=('charge_integral_ratio','min'),max_charge_ratio=('charge_integral_ratio','max'),
        zero_time_intervals=('zero_time_intervals','sum')).reset_index()
    # 실습 설정: 충전 시간 품질 기준. 산업 승인 기준이 아닙니다.
    CHARGE_TIME_MAX_MIN = 60

    def positive_mean(values, upper=None):
        values = np.asarray(values, dtype=float)
        keep = np.isfinite(values) & (values > 0)
        if upper is not None: keep &= values <= upper
        return float(values[keep].mean()) if keep.any() else np.nan

    early_rows = []
    for cell_id, group in df.groupby("cell_id", sort=False):
        early = group[group.cycle.between(EARLY_START, EARLY_END)]
        q2 = group.loc[group.cycle == 2, "QD"]
        first_ir = group.loc[group.cycle.between(2, 10), "IR"]
        last_ir = group.loc[group.cycle.between(91, 100), "IR"]
        early_rows.append({"cell_id": cell_id,
            "QD_cycle2": float(q2.iloc[0]) if len(q2) else np.nan,
            "mean_IR": positive_mean(early.IR),
            "delta_IR": positive_mean(last_ir) - positive_mean(first_ir),
            "mean_Tavg": positive_mean(early.Tavg), "mean_Tmax": positive_mean(early.Tmax),
            "mean_chargetime": positive_mean(early.chargetime, upper=CHARGE_TIME_MAX_MIN)})
    early_features = pd.DataFrame(early_rows)
    features = cells_charge.merge(early_features, on="cell_id", validate="one_to_one")

    return features, pd.DataFrame(provenance), current_quality, current_rejected


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',required=True)
    parser.add_argument('--output',default='data/processed/cells_and_features.csv')
    args=parser.parse_args();features,provenance,quality,rejected=extract_features(args.data_dir)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    features.to_csv(out,index=False)
    provenance.to_csv(out.with_name('source_manifest.csv'),index=False)
    quality.to_csv(out.with_name('current_quality.csv'),index=False)
    rejected.to_csv(out.with_name('current_rejected.csv'),index=False)
    from .quality import author_rules
    author_rules(args.data_dir).to_csv(out.with_name('batch3_author_quality_rules.csv'),index=False)
    print(f'저장: {out}, {len(features)}셀')
if __name__=='__main__': main()
