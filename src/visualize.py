"""같은 축 범위를 적용한 Batch별 독립 그림."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .preprocess import make_model,predict_positive
BLUE='#3569a6';INK='#25303a';GREY='#b5bbc3'
def save(fig,path):
 fig.tight_layout();fig.savefig(path,dpi=180,bbox_inches='tight');plt.close(fig)
def build_charts(features,cv,locked,pred,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK,'axes.edgecolor':'#7c858e'})
 contract={'renderer':'Matplotlib PNG in notebooks and PDF','grain':'one cell per point; all finite recorded targets','palette':'single blue root plus neutrals; circle/square and solid/dashed distinguish roles','batch_rule':'each quantitative chart contains one Batch only','scatter_limits_cycles':[0,2100],'signed_error_limits_pct':[-100,100],'charts':[]}
 best=cv.sort_values('CV_MAPE_pct').groupby('candidate',sort=True).head(1).sort_values('candidate')
 fig,ax=plt.subplots(figsize=(9,3.3));x=np.arange(len(best))
 ax.bar(x,best.CV_MAPE_pct,color=BLUE,edgecolor=INK,width=.58)
 ax.errorbar(x,best.CV_MAPE_pct,yerr=best.CV_MAPE_sd,fmt='none',ecolor=INK,capsize=5)
 labels={'M0':'M0\nMedian','M1':'M1\nSingle Ridge','M2':'M2\nMulti Ridge','M3':'M3\nRandom Forest','M4':'M4\nDelta-Q Ridge','M5':'M5\nElasticNet'}
 ax.set_xticks(x,[labels[k] for k in best.candidate]);ax.set_ylim(0,35)
 ax.set_ylabel('MAPE (%)');ax.set_title('Batch 1 | Candidate cross-validation MAPE\n35 development cells, 5 policy groups folds; error bars = fold SD',loc='left',fontsize=11)
 for i,r in enumerate(best.itertuples()):ax.text(i,r.CV_MAPE_pct+r.CV_MAPE_sd+.7,f'{r.CV_MAPE_pct:.2f}%',ha='center')
 save(fig,out/'batch1_candidates.png');contract['charts'].append({'file':'batch1_candidates.png','family':'bar','rows':len(best),'takeaway':'single Ridge has minimum grouped-CV MAPE after EDA-grounded Delta-Q/ElasticNet extension; uncertainty limits fine distinctions'})
 for b in [1,2,3]:
  g=pred[pred.batch==b].copy();fig,ax=plt.subplots(figsize=(8.5,4.3))
  if b==1:
   for role,marker,fill in [('Train (Batch 1 CV)','o','none'),('Valid (Batch 1 Hold-out)','s',BLUE)]:
    z=g[g.role==role];ax.scatter(z.cycle_life,z.prediction,marker=marker,facecolors=fill,edgecolors=BLUE,label=role,s=40)
   ax.legend(fontsize=9);caption='35 CV out-of-fold + 11 hold-out cells; distinct fitted models'
  else:ax.scatter(g.cycle_life,g.prediction,color=BLUE,edgecolor=INK,linewidth=.5,s=40);caption=f'{len(g)} cells; selected configuration refitted on all 46 Batch 1 cells'
  ax.plot([0,2100],[0,2100],color=INK,linestyle='--',linewidth=1.1)
  for j,r in enumerate(g.nlargest(3,'APE_pct').itertuples()):
   ax.annotate(r.cell_id,(r.cycle_life,r.prediction),xytext=[(6,12),(8,-18),(-40,12)][j],textcoords='offset points',fontsize=9,arrowprops={'arrowstyle':'-','color':GREY,'lw':.6})
  ax.set(xlim=(0,2100),ylim=(0,2100),xlabel='Recorded cycle life (cycles)',ylabel='Predicted total life (cycles)')
  ax.set_title(f'Batch {b} | Recorded and predicted cycle life\n{caption}',loc='left',fontsize=11)
  save(fig,out/f'batch{b}_prediction.png');contract['charts'].append({'file':f'batch{b}_prediction.png','family':'scatter','rows':len(g),'takeaway':'calibration relative to equality line; label largest relative errors'})
  fig,ax=plt.subplots(figsize=(8.5,3.6));ax.scatter(g.cycle_life,g.signed_PE_pct,facecolor=BLUE,edgecolor=INK,linewidth=.5,s=38)
  ax.axhline(0,color=INK,lw=1);ax.axhline(20,color=GREY,lw=1,ls='--');ax.axhline(-20,color=GREY,lw=1,ls='--')
  ax.set(xlim=(0,2100),ylim=(-100,100),xlabel='Recorded cycle life (cycles)',ylabel='Signed error (%)')
  ax.set_title(f'Batch {b} | Relative prediction error\n(prediction - recorded life) / recorded life; n={len(g)}',loc='left',fontsize=11)
  save(fig,out/f'batch{b}_errors.png');contract['charts'].append({'file':f'batch{b}_errors.png','family':'scatter','rows':len(g),'takeaway':'positive error delays replacement planning; negative error advances it'})
 b1=features[(features.batch==1)&features.cycle_life.notna()];model=make_model(locked['spec']);model.fit(b1[locked['features']],b1.cycle_life)
 if locked['features']==['logvar_deltaQ']:
  grid=np.linspace(features.logvar_deltaQ.min()-.1,features.logvar_deltaQ.max()+.1,100)
  line=predict_positive(model,pd.DataFrame({'logvar_deltaQ':grid}))
  for b in [1,2,3]:
   g=features[(features.batch==b)&features.cycle_life.notna()];fig,ax=plt.subplots(figsize=(8.5,3.5))
   ax.scatter(g.logvar_deltaQ,g.cycle_life,color=BLUE,s=36,edgecolor=INK,linewidth=.5)
   ax.plot(grid,line,color=INK,linestyle='--',lw=1.1,label='Fixed Batch 1 model');ax.legend(fontsize=9)
   ax.set(xlim=(grid.min(),grid.max()),ylim=(0,2100),xlabel='Log10 variance of Q100(V) - Q10(V)',ylabel='Recorded cycle life (cycles)')
   ax.set_title(f'Batch {b} | Early signal and lifetime calibration\nSame Batch 1 mapping shown in each figure; n={len(g)}',loc='left',fontsize=11)
   save(fig,out/f'batch{b}_signal_mapping.png');contract['charts'].append({'file':f'batch{b}_signal_mapping.png','family':'scatter','rows':len(g),'takeaway':'input range membership does not guarantee transfer of signal-to-lifetime calibration'})
 (out.parent/'chart_contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2))
