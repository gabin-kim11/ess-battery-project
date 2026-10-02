"""Ridge 최종 학습값을 실행 가능한 JSON으로 저장한다."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from .preprocess import validate_features,make_model,predict_positive

def export_model(features,locked,out):
    spec=locked['spec'];cols=locked['features'];frame=validate_features(features)
    train=frame[frame.batch==1]
    model=make_model(spec).fit(train[cols],train.cycle_life)
    pipe=model.regressor_ if hasattr(model,'regressor_') else model
    if spec['family']!='ridge':raise ValueError('현재 JSON 내보내기는 Ridge 모델에 적용합니다.')
    artifact={'format':'ridge-json-v1','spec':spec,'features':cols,'target':spec['target'],
      'imputer':pipe.named_steps['imputer'].statistics_.tolist(),
      'mean':pipe.named_steps['scaler'].mean_.tolist(),'scale':pipe.named_steps['scaler'].scale_.tolist(),
      'coef':pipe.named_steps['model'].coef_.tolist(),'intercept':float(pipe.named_steps['model'].intercept_),
      'training_feature_min':train[cols].min().tolist(),'training_feature_max':train[cols].max().tolist(),
      'training_cells':len(train),'prediction_cycle':100,'output':'total recorded cycle life, cycles'}
    path=Path(out)/'model_artifact.json';path.write_text(json.dumps(artifact,ensure_ascii=False,indent=2))
    return artifact

def predict_artifact(frame,artifact):
    X=frame[artifact['features']].to_numpy(float);X=np.where(np.isnan(X),artifact['imputer'],X)
    transformed=(X-np.array(artifact['mean']))/np.array(artifact['scale'])
    y=transformed@np.array(artifact['coef'])+artifact['intercept']
    if artifact['target']=='log10':y=10.**y
    return np.maximum(y,1.)

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--model',default='results/model_artifact.json');p.add_argument('--output',default='results/new_predictions.csv');a=p.parse_args()
    data=pd.read_csv(a.input);artifact=json.loads(Path(a.model).read_text())
    output=data[['cell_id']].copy() if 'cell_id' in data else pd.DataFrame(index=data.index)
    output['predicted_total_cycles']=predict_artifact(data,artifact)
    output['remaining_cycles_at_100']=output.predicted_total_cycles-100
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);output.to_csv(path,index=False);print(path)
if __name__=='__main__':main()
