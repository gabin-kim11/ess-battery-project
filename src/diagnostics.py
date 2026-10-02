"""模型 고정 후 오차, 순위, 정책/수명 분포 진단. 재튜닝에 사용하지 않는다."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def diagnose(features,pred,out):
 out=Path(out);rows=[]
 train=features[(features.batch==1)&features.cycle_life.notna()]
 for role,g in pred.groupby('role',sort=False):
  rank=spearmanr(g.cycle_life,g.prediction)
  for label,mask in [('all',np.ones(len(g),bool)),('below_train_lifetime',g.cycle_life<train.cycle_life.min()),('above_train_lifetime',g.cycle_life>train.cycle_life.max()),('within_train_lifetime',g.cycle_life.between(train.cycle_life.min(),train.cycle_life.max()))]:
   z=g.loc[mask]
   if len(z):rows.append({'role':role,'stratum':label,'n':len(z),'MAPE_pct':z.APE_pct.mean(),'MAE_cycles':z.error_cycles.abs().mean(),'bias_cycles':z.error_cycles.mean(),'Spearman_all':float(rank.statistic)})
 pd.DataFrame(rows).to_csv(out/'lifetime_range_errors.csv',index=False)
 groups=[]
 for b,g in features.groupby('batch'):
  valid=g[g.cycle_life.notna()]
  groups.append({'batch':b,'raw_n':len(g),'scored_n':len(valid),'missing_target_n':g.cycle_life.isna().sum(),'life_min':valid.cycle_life.min(),'life_max':valid.cycle_life.max(),'life_median':valid.cycle_life.median(),
   'below_500_n':int((valid.cycle_life<500).sum()),'over_1000_n':int((valid.cycle_life>1000).sum()),
   'policy_n':valid.charging_policy.nunique(),'new_policy_n':int((~valid.charging_policy.isin(train.charging_policy)).sum()),'EOL_near_n':int(valid.eol_near.sum()),'IR_missing_n':int(valid.delta_IR.isna().sum())})
 pd.DataFrame(groups).to_csv(out/'cohort_profile.csv',index=False)
