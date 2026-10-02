"""Batch 1에서 후보를 고정한 뒤 Hold-out, Batch 2, Batch 3을 평가한다."""
from pathlib import Path
from itertools import product
import argparse,json,platform,importlib.metadata
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from .preprocess import FEATURE_SETS,SEED,validate_features,split_development,make_model,predict_positive,metrics
PROTOCOL={'seed':SEED,'task':'regression','target':'recorded cycle_life (cycles)',
 'feature_cycle_start':2,'feature_cycle_end':100,
 'holdout':'Batch 1 GroupShuffleSplit by charging_policy, test_size=0.2',
 'cv':'5-fold GroupKFold within development cells only',
 'primary_selection_metric':'unweighted mean of fold MAPE (%)',
 'metric_rationale':'MAPE chosen for relative error across lifetime scales; design-stage MAE retained for absolute-cycle interpretation',
 'feature_screening_basis':'logvar_deltaQ represents correlated Delta-Q summaries; min_deltaQ remains a separate development hypothesis',
 'secondary_metrics':['MAE','fold MAPE standard deviation','overprediction'],
 'selection_rule':'minimum CV MAPE; eligible = mean <= best mean + best fold SD/sqrt(5); prefer Ridge to Random Forest within eligible families, then minimum mean within family',
 'targets':['raw','log10'],'ridge_alpha':[.01,.1,1.,10.,100.],
 'rf_max_depth':[2,4],'rf_min_samples_leaf':[3,6],'rf_n_estimators':200,
 'feature_sets':FEATURE_SETS,'external_evaluation':'selected config only; refit all Batch 1; Batch 2 mandatory, Batch 3 separate',
 'gap_formula':'literal signed A-B; negative gaps mean an increased MAPE in B',
 'paper_target_MAPE_pct':9.1,
 'prior_access':'Exploratory analysis examined all batches; earlier exploratory scores exist. Current protocol frozen before new external predictions; this is not a pristine unseen test.',
 'label_quality':'all finite cycle_life > 100; terminal capacity metadata used in diagnostics only, no future-based training filter'}

def dump_json(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def candidate_specs():
    specs=[{'candidate':'M0','family':'median','features':'single','target':'raw'}]
    for target,alpha in product(PROTOCOL['targets'],PROTOCOL['ridge_alpha']):
        specs.append({'candidate':'M1','family':'ridge','features':'single','target':target,'alpha':alpha})
    for feature,target,alpha in product(['physical','policy','current'],PROTOCOL['targets'],PROTOCOL['ridge_alpha']):
        specs.append({'candidate':'M2','family':'ridge','features':feature,'target':target,'alpha':alpha})
    for feature,target,depth,leaf in product(['physical','policy','current'],PROTOCOL['targets'],PROTOCOL['rf_max_depth'],PROTOCOL['rf_min_samples_leaf']):
        specs.append({'candidate':'M3','family':'forest','features':feature,'target':target,'max_depth':depth,'min_samples_leaf':leaf})
    return [{'spec_id':f'S{i:03}',**s} for i,s in enumerate(specs)]

def compare_candidates(dev,folds,out):
    records,fold_records,oof_records=[],[],[]
    for spec in candidate_specs():
        cols=FEATURE_SETS[spec['features']];scores=[];mae=[]
        for fold,(ti,vi) in enumerate(folds,1):
            model=make_model(spec);model.fit(dev.iloc[ti][cols],dev.iloc[ti].cycle_life)
            pred=predict_positive(model,dev.iloc[vi][cols]);score=metrics(dev.iloc[vi].cycle_life,pred)
            scores.append(score['MAPE_pct']);mae.append(score['MAE_cycles'])
            fold_records.append({'spec_id':spec['spec_id'],'candidate':spec['candidate'],'fold':fold,**score})
            for row,p in zip(dev.iloc[vi].itertuples(),pred):oof_records.append({'spec_id':spec['spec_id'],'fold':fold,'cell_id':row.cell_id,'batch':1,'charging_policy':row.charging_policy,'cycle_life':row.cycle_life,'prediction':p})
        records.append({**spec,'CV_MAPE_pct':np.mean(scores),'CV_MAPE_sd':np.std(scores,ddof=1),'CV_MAE_cycles':np.mean(mae),'feature_n':len(cols)})
    cv=pd.DataFrame(records).sort_values(['CV_MAPE_pct','CV_MAE_cycles','spec_id']).reset_index(drop=True)
    best=cv.iloc[0];threshold=best.CV_MAPE_pct+best.CV_MAPE_sd/np.sqrt(5)
    eligible=cv[cv.CV_MAPE_pct<=threshold].copy();eligible['priority']=eligible.family.map({'ridge':0,'forest':1,'median':2})
    selected=eligible.sort_values(['priority','CV_MAPE_pct','CV_MAE_cycles','spec_id']).iloc[0]
    spec=next(s for s in candidate_specs() if s['spec_id']==selected.spec_id)
    cv['selected']=cv.spec_id==spec['spec_id'];cv['within_one_SE']=cv.CV_MAPE_pct<=threshold
    cv.to_csv(out/'candidate_comparison.csv',index=False)
    pd.DataFrame(fold_records).to_csv(out/'cv_folds.csv',index=False)
    oof=pd.DataFrame(oof_records);oof.to_csv(out/'candidate_oof_predictions.csv',index=False)
    locked={'spec':spec,'features':FEATURE_SETS[spec['features']],'best_CV_MAPE_pct':float(best.CV_MAPE_pct),
      'one_SE_threshold_pct':float(threshold),'selected_CV_MAPE_pct':float(selected.CV_MAPE_pct),'selected_CV_sd':float(selected.CV_MAPE_sd),
      'selection_basis':'Batch 1 development CV only','lock_stage':'before Hold-out and Batch 2/3 predictions'}
    dump_json(out/'selected_model.json',locked)
    print('고정 후보:',spec,'CV MAPE',round(selected.CV_MAPE_pct,4),flush=True)
    return cv,oof,locked

def prediction_frame(data,pred,role):
    frame=data.copy();frame['prediction']=pred;frame['role']=role
    frame['error_cycles']=frame.prediction-frame.cycle_life
    frame['APE_pct']=frame.error_cycles.abs()/frame.cycle_life*100
    frame['signed_PE_pct']=frame.error_cycles/frame.cycle_life*100
    return frame

def clustered_ci(data,n_boot=2000):
    rng=np.random.default_rng(SEED);groups=[g.APE_pct.to_numpy() for _,g in data.groupby('charging_policy')]
    values=[np.mean(np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])) for _ in range(n_boot)]
    return np.quantile(values,[.025,.975]).tolist()

def evaluate_locked(frame,dev,hold,cv,oof,locked,out):
    spec=locked['spec'];cols=locked['features']
    dev_model=make_model(spec);dev_model.fit(dev[cols],dev.cycle_life)
    valid=prediction_frame(hold,predict_positive(dev_model,hold[cols]),'Valid (Batch 1 Hold-out)')
    cv_pred=oof[oof.spec_id==spec['spec_id']].drop(columns='spec_id').merge(dev,on=['cell_id','batch','charging_policy','cycle_life'],validate='one_to_one')
    cv_pred=prediction_frame(cv_pred,cv_pred.prediction,'Train (Batch 1 CV)')
    batch1=frame[frame.batch==1].copy();final_model=make_model(spec);final_model.fit(batch1[cols],batch1.cycle_life)
    external=[]
    for b in [2,3]:
        cohort=frame[frame.batch==b].copy()
        external.append(prediction_frame(cohort,predict_positive(final_model,cohort[cols]),f'Test (Batch {b})'))
    predictions=pd.concat([cv_pred,valid,*external],ignore_index=True)
    lo=batch1[cols].min();hi=batch1[cols].max()
    predictions['out_of_training_range_n']=((predictions[cols]<lo)|(predictions[cols]>hi)).sum(axis=1)
    predictions['missing_input_n']=predictions[cols].isna().sum(axis=1)
    predictions.to_csv(out/'cell_predictions.csv',index=False)
    perf=[]
    for role,group in predictions.groupby('role',sort=False):
        score=metrics(group.cycle_life,group.prediction)
        if role.startswith('Train'):
            score['MAPE_pct']=locked['selected_CV_MAPE_pct'];note='개발 5폴드 MAPE 단순 평균; 후보 선택 후 CV 추정치'
        else:note='개발 집단 모델' if role.startswith('Valid') else '전체 Batch 1 재학습 모델'
        ci=clustered_ci(group)
        perf.append({'role':role,**score,'OOF_or_cell_MAPE_pct':group.APE_pct.mean(),'cluster_CI_low_pct':ci[0],'cluster_CI_high_pct':ci[1],'note':note})
    performance=pd.DataFrame(perf);performance.to_csv(out/'evaluation_metrics.csv',index=False)
    score=dict(zip(performance.role,performance.MAPE_pct))
    a=score['Train (Batch 1 CV)'];v=score['Valid (Batch 1 Hold-out)'];b2=score['Test (Batch 2)'];b3=score['Test (Batch 3)']
    reporting=pd.DataFrame([
      ('Train (Batch 1 CV)',a,'%', '폴드 MAPE 단순 평균'),('Valid (Batch 1 Hold-out)',v,'%', '정책 단위 Hold-out'),('Test (Batch 2)',b2,'%', '고정 모델 최종 평가'),
      ('Gap (Train-Valid)',a-v,'%p','Train - Valid; 음수는 Valid 오차 증가'),('Gap (Valid-Test)',v-b2,'%p','Valid - Batch 2; 음수는 외부 오차 증가'),
      ('Gap (Target-Test)',9.1-b2,'%p','9.1 - Batch 2; 음수는 논문 기준 미달'),('Test (Batch 3)',b3,'%', '별도 추가 검증'),
      ('Gap (Batch2-Batch3)',b2-b3,'%p','Batch 2 - Batch 3'),('Gap (Target-Test, Batch 3)',9.1-b3,'%p','9.1 - Batch 3')],columns=['구분','MAPE (%) 또는 Gap (%p)','unit','비고'])
    reporting.to_csv(out/'model_performance.csv',index=False)
    predictions.sort_values('APE_pct',ascending=False).groupby('role',sort=False).head(5).to_csv(out/'worst_cells.csv',index=False)
    rows=[]
    for (role,policy),g in predictions.groupby(['role','charging_policy']):rows.append({'role':role,'charging_policy':policy,**metrics(g.cycle_life,g.prediction),'mean_range_excess_n':g.out_of_training_range_n.mean()})
    pd.DataFrame(rows).to_csv(out/'policy_errors.csv',index=False)
    strata=[]
    for role,g in predictions.groupby('role',sort=False):
        for name,mask in [('all',np.ones(len(g),bool)),('missing_input',g.missing_input_n>0),('complete_input',g.missing_input_n==0),('in_range',g.out_of_training_range_n==0),('out_of_range',g.out_of_training_range_n>0),('terminal_QD_near_EOL',g.eol_near),('terminal_QD_above_EOL',~g.eol_near)]:
            z=g.loc[mask]
            if len(z):strata.append({'role':role,'stratum':name,**metrics(z.cycle_life,z.prediction)})
    pd.DataFrame(strata).to_csv(out/'error_strata.csv',index=False)
    baseline=make_model(candidate_specs()[0]);baseline.fit(batch1[['logvar_deltaQ']],batch1.cycle_life)
    base_rows=[]
    for b in [2,3]:
        g=frame[frame.batch==b];base_rows.append({'batch':b,**metrics(g.cycle_life,predict_positive(baseline,g[['logvar_deltaQ']]))})
    pd.DataFrame(base_rows).to_csv(out/'external_baseline.csv',index=False)
    importance=permutation_importance(dev_model,hold[cols],hold.cycle_life,scoring='neg_mean_absolute_percentage_error',n_repeats=50,random_state=SEED,n_jobs=1)
    pd.DataFrame({'feature':cols,'MAPE_increase_pp':importance.importances_mean*100,'SD_pp':importance.importances_std*100}).sort_values('MAPE_increase_pp',ascending=False).to_csv(out/'holdout_permutation_importance.csv',index=False)
    pipe=final_model.regressor_ if hasattr(final_model,'regressor_') else final_model
    if spec['family']=='ridge':
        pd.DataFrame({'feature':cols,'standardized_coefficient':pipe.named_steps['model'].coef_}).to_csv(out/'model_coefficients.csv',index=False)
    elif spec['family']=='forest':
        pd.DataFrame({'feature':cols,'impurity_importance':pipe.named_steps['model'].feature_importances_}).to_csv(out/'model_coefficients.csv',index=False)
    dump_json(out/'fitted_preprocessing.json',{'features':cols,'imputer_statistics':pipe.named_steps['imputer'].statistics_.tolist(),
      'scaler_mean':pipe.named_steps['scaler'].mean_.tolist() if 'scaler' in pipe.named_steps else None,
      'scaler_scale':pipe.named_steps['scaler'].scale_.tolist() if 'scaler' in pipe.named_steps else None})
    print(reporting.to_string(index=False),flush=True)
    return predictions,performance,reporting

def run(frame,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);dump_json(out/'protocol.json',PROTOCOL)
    frame=validate_features(frame);b1=frame[frame.batch==1].reset_index(drop=True)
    dev,hold,folds=split_development(b1)
    split=frame[['cell_id','batch','charging_policy']].copy()
    split['role']=np.where(split.batch==1,'development',np.where(split.batch==2,'external_test','additional_test'))
    split.loc[split.cell_id.isin(hold.cell_id),'role']='holdout';split['CV_fold']=np.nan
    for fold,(_,vi) in enumerate(folds,1):split.loc[split.cell_id.isin(dev.iloc[vi].cell_id),'CV_fold']=fold
    split.to_csv(out/'split_manifest.csv',index=False)
    print('개발/검증/Batch 2/Batch 3:',len(dev),len(hold),int((frame.batch==2).sum()),int((frame.batch==3).sum()),flush=True)
    cv,oof,locked=compare_candidates(dev,folds,out)
    pred,perf,report=evaluate_locked(frame,dev,hold,cv,oof,locked,out)
    dump_json(out/'environment.json',{'python':platform.python_version(),**{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','h5py','matplotlib','nbformat','nbclient']}})
    dump_json(out/'run_validation.json',{'unique_cells':True,'holdout_policy_overlap':0,'CV_policy_overlap':0,'feature_cycle_max':100,'future_metadata_in_input':False,
      'cohort_counts':frame.batch.value_counts().sort_index().to_dict(),'candidate_configs':len(cv),'CV_fits':len(cv)*5,'selected_before_external_prediction':True,
      'evaluation_scope':'conditional external validation after prior all-batch EDA, not pristine untouched external validation'})
    return cv,locked,pred,perf,report

def main():
    p=argparse.ArgumentParser();p.add_argument('--features',default='data/processed/cells_and_features.csv');p.add_argument('--output',default='results');a=p.parse_args();run(pd.read_csv(a.features),a.output)
if __name__=='__main__':main()
