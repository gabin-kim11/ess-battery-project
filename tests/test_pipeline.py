"""실제 누수와 시간 범위를 검증하는 단위 테스트."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.preprocess import FEATURE_SETS,split_development,make_model,metrics
from src.features import charge_pattern
class PipelineChecks(unittest.TestCase):
 def setUp(self):self.frame=pd.read_csv(Path(__file__).resolve().parents[1]/'data/processed/cells_and_features.csv')
 def test_no_policy_overlap(self):
  dev,hold,folds=split_development(self.frame[self.frame.batch==1])
  self.assertFalse(set(dev.charging_policy)&set(hold.charging_policy))
  self.assertEqual(len(dev)+len(hold),46)
  for ti,vi in folds:self.assertFalse(set(dev.iloc[ti].charging_policy)&set(dev.iloc[vi].charging_policy))
 def test_fit_statistics_use_training_only(self):
  spec={'family':'ridge','target':'raw','alpha':1}
  x=pd.DataFrame({'x':[0.,2.,np.nan]});model=make_model(spec).fit(x,[10.,20.,30.])
  model.predict(pd.DataFrame({'x':[1000000.,np.nan]}))
  self.assertEqual(model.named_steps['imputer'].statistics_[0],1.)
  self.assertEqual(model.named_steps['scaler'].mean_[0],1.)
 def test_future_metadata_excluded(self):
  forbidden={'cycle_life','eol_near','tail_median_QD','summary_cycles','cell_id','batch','knee_cycle'}
  self.assertFalse(set(sum(FEATURE_SETS.values(),[]))&forbidden)
 def test_current_unit_and_interpolation(self):
  # 4C에 1.1Ah를 곱하면 4.4A. 0.88Ah 충전은 정확히 12분이다.
  t=np.linspace(0.,13.,131);i=np.full(len(t),4.);qc=t*4.4/60
  result=charge_pattern(t,i,qc)
  self.assertAlmostEqual(result['fast_charge_duration_min'],12.)
  self.assertAlmostEqual(result['current_rms_C'],4.)
  self.assertAlmostEqual(result['current_unit_ratio_A_per_C'],1.1)
 def test_metric_units(self):
  score=metrics([100.,200.],[110.,180.])
  self.assertAlmostEqual(score['MAPE_pct'],10.);self.assertAlmostEqual(score['MAE_cycles'],15.)
if __name__=='__main__':unittest.main()
