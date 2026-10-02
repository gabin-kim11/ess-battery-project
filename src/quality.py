"""저자 LoadData.m의 Batch 3 품질 규칙을 원본 순서에 대응시키는 진단."""
from pathlib import Path
import h5py
import pandas as pd
SOURCE='https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/LoadData.m'
def author_rules(data_dir):
 path=Path(data_dir)/'2018-04-12_batchdata_updated_struct_errorcorrect.mat'
 with h5py.File(path) as h:
  b=h['batch'];n=b['summary'].shape[0]
  if n!=46:raise ValueError('저자 인덱스 규칙은 원본 46셀 순서에만 적용합니다.')
  caps=[float(h[b['summary'][i,0]]['QDischarge'][()].reshape(-1)[-1]) for i in range(n)]
 remaining=[i for i in range(n) if i!=37 and not caps[i]>.885]
 noisy=[remaining[i] for i in [2,39,40]] # MATLAB 1-based [3,40,41]
 flags={37:'channel-46 collection issue',**{i:'author noisy-cell rule after terminal filter' for i in noisy}}
 return pd.DataFrame([{'cell_id':f'b3c{i}','raw_index':i,'last_QD':caps[i],'author_flag':i in flags,'author_rule':flags.get(i,''),'terminal_above_0885':caps[i]>.885,'source':SOURCE} for i in range(n)])
def quality_sensitivity(pred,rules,out):
 out=Path(out);rules.to_csv(out/'batch3_author_quality_rules.csv',index=False)
 excluded=set(rules.loc[rules.author_flag|rules.terminal_above_0885,'cell_id'])
 g=pred[pred.batch==3];clean=g[~g.cell_id.isin(excluded)]
 result=pd.DataFrame([{'cohort':label,'n':len(z),'MAPE_pct':z.APE_pct.mean(),'MAE_cycles':z.error_cycles.abs().mean()} for label,z in [('all finite targets',g),('author-rule quality sensitivity, same locked model',clean)]])
 result.to_csv(out/'batch3_quality_sensitivity.csv',index=False)
 return result
