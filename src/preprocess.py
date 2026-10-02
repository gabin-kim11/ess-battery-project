"""충전 정책 단위 분할과 학습 폴드 내 전처리."""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit,GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.compose import TransformedTargetRegressor
BASE=['logvar_deltaQ','slope_QD','delta_IR','mean_Tavg','mean_chargetime']
FEATURE_SETS={'single':['logvar_deltaQ'],'physical':BASE,'policy':BASE+['C1','SOC_switch','C2'],
 'current':[f for f in BASE if f!='mean_chargetime']+['current_rms_C','current_cv']}
FEATURE_SETS.update({'delta_pair':['logvar_deltaQ','min_deltaQ'],
 'physical_delta':FEATURE_SETS['physical']+['min_deltaQ'],
 'current_delta':FEATURE_SETS['current']+['min_deltaQ']})
SEED=42

def validate_features(frame):
    assert frame.cell_id.is_unique
    required=set(sum(FEATURE_SETS.values(),[])+['batch','cell_id','cycle_life','charging_policy'])
    if not required.issubset(frame):raise ValueError(f'필수 변수 누락: {required-set(frame)}')
    valid=frame[np.isfinite(frame.cycle_life)&(frame.cycle_life>100)].copy()
    assert valid.batch.isin([1,2,3]).all()
    assert not np.isinf(valid[list(required-{'cell_id','charging_policy'})].astype(float)).any().any()
    return valid

def split_development(batch1):
    splitter=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED)
    ti,vi=next(splitter.split(batch1,groups=batch1.charging_policy))
    dev=batch1.iloc[ti].reset_index(drop=True);hold=batch1.iloc[vi].reset_index(drop=True)
    assert set(dev.cell_id).isdisjoint(hold.cell_id)
    assert set(dev.charging_policy).isdisjoint(hold.charging_policy)
    folds=list(GroupKFold(n_splits=5).split(dev,groups=dev.charging_policy))
    for ti,vi in folds:assert set(dev.iloc[ti].charging_policy).isdisjoint(dev.iloc[vi].charging_policy)
    return dev,hold,folds

def make_model(spec):
    steps=[('imputer',SimpleImputer(strategy='median',keep_empty_features=True))]
    if spec['family']=='median':reg=DummyRegressor(strategy='median')
    elif spec['family']=='ridge':
        steps.append(('scaler',StandardScaler()));reg=Ridge(alpha=spec['alpha'])
    elif spec['family']=='elasticnet':
        steps.append(('scaler',StandardScaler()))
        reg=ElasticNet(alpha=spec['alpha'],l1_ratio=spec['l1_ratio'],max_iter=50000,tol=1e-6,selection='cyclic')
    elif spec['family']=='forest':reg=RandomForestRegressor(n_estimators=200,max_depth=spec['max_depth'],min_samples_leaf=spec['min_samples_leaf'],max_features=1.,random_state=SEED,n_jobs=1)
    else:raise ValueError(f"지원하지 않는 모델 계열: {spec['family']}")
    steps.append(('model',reg));model=Pipeline(steps)
    if spec['target']=='log10':model=TransformedTargetRegressor(regressor=model,func=np.log10,inverse_func=lambda a:10.**a)
    return model

def predict_positive(model,X):return np.maximum(model.predict(X),1.)

def metrics(y,pred):
    y=np.asarray(y,float);err=np.asarray(pred,float)-y
    return {'n':len(y),'MAPE_pct':float(np.mean(np.abs(err)/y)*100),'MAE_cycles':float(np.mean(np.abs(err))),
      'RMSE_cycles':float(np.sqrt(np.mean(err**2))),'bias_cycles':float(err.mean()),
      'overprediction_pct':float(np.mean(err>0)*100),'over20pct_pct':float(np.mean(err/y>.2)*100)}
