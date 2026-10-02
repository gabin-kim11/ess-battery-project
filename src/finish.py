"""고정 결과의 그림·진단과 학습된 JSON 모델을 재현한다."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from .artifact import export_model,predict_artifact
from .visualize import build_charts
from .diagnostics import diagnose
from .quality import quality_sensitivity
from .insights import save_insight_evidence

def finish(root,out):
 root=Path(root);out=Path(out)
 features=pd.read_csv(root/'data/processed/cells_and_features.csv')
 cv=pd.read_csv(out/'candidate_comparison.csv');pred=pd.read_csv(out/'cell_predictions.csv')
 locked=json.loads((out/'selected_model.json').read_text())
 build_charts(features,cv,locked,pred,out/'charts');diagnose(features,pred,out)
 rules=pd.read_csv(root/'data/processed/batch3_author_quality_rules.csv')
 quality_sensitivity(pred,rules,out)
 save_insight_evidence(root,out)
 artifact=export_model(features,locked,out)
 test=pred[pred.batch.isin([2,3])]
 np.testing.assert_allclose(predict_artifact(test,artifact),test.prediction,rtol=1e-12,atol=1e-9)
 print('JSON 모델 예측과 고정 테스트 예측 일치')
 return artifact

def main():
 p=argparse.ArgumentParser();p.add_argument('--results',default='results');p.add_argument('--root',default='.');a=p.parse_args();finish(a.root,a.results)
if __name__=='__main__':main()
