"""실행 결과 표·단일 Batch 그림을 사용하는 모델 개발·평가 보고서."""
from pathlib import Path
import json,html
import pandas as pd
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results';A=ROOT/'reports/assets'
for name,file in [('NG','NanumGothic-Regular.ttf'),('NGB','NanumGothic-Bold.ttf'),('Symbols','DejaVuSans.ttf'),('SymbolsBold','DejaVuSans-Bold.ttf')]:pdfmetrics.registerFont(TTFont(name,str(A/file)))
styles={
 'title':ParagraphStyle('title',fontName='NGB',fontSize=21,leading=30,textColor=colors.HexColor('#25303a'),spaceAfter=18),
 'head':ParagraphStyle('head',fontName='NGB',fontSize=15,leading=22,textColor=colors.HexColor('#25303a'),spaceAfter=14),
 'sub':ParagraphStyle('sub',fontName='NGB',fontSize=11,leading=17,spaceBefore=12,spaceAfter=7),
 'body':ParagraphStyle('body',fontName='NG',fontSize=10.2,leading=16,spaceAfter=10,wordWrap='CJK'),
 'note':ParagraphStyle('note',fontName='NG',fontSize=8.4,leading=13,spaceAfter=8,textColor=colors.HexColor('#515b65'),wordWrap='CJK'),
 'table':ParagraphStyle('table',fontName='NG',fontSize=8.8,leading=13,wordWrap='CJK')}
story=[]
def symbols(text,kind='body'):
 font='SymbolsBold' if kind in {'title','head','sub'} else 'Symbols'
 return text.replace('Δ',f'<font name="{font}">Δ</font>')
def p(text,kind='body'):story.append(Paragraph(symbols(text,kind),styles[kind]))
def page(title):
 if story:story.append(PageBreak())
 p(title,'head')
def table(rows,widths):
 t=Table([[Paragraph(symbols(html.escape(str(x)),'table'),styles['table']) for x in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef4')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.8,colors.HexColor('#7b8794')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d5dbe1')),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
 story.append(t);story.append(Spacer(1,12))
def chart(name,width=495):
 path=R/'charts'/name
 from PIL import Image as PILImage
 w,h=PILImage.open(path).size;story.append(Image(str(path),width=width,height=width*h/w));story.append(Spacer(1,10))
perf=pd.read_csv(R/'evaluation_metrics.csv');cv=pd.read_csv(R/'candidate_comparison.csv');pred=pd.read_csv(R/'cell_predictions.csv')
locked=json.loads((R/'selected_model.json').read_text());reporting=pd.read_csv(R/'model_performance.csv')
p('초기 열화 신호 기반<br/>배터리 수명 예측 모델 개발 및 검증','title')
p('단일 ΔQ Ridge의 선택과 배치별 적용 범위','sub')
p('초기 100 사이클의 방전 곡선 차이를 사용하는 Ridge(alpha=1)를 최종 모델로 선택했다. Batch 1 교차검증에서는 추가 센서·정책 피처와 비선형 모델보다 낮은 평균 MAPE를 보였지만, Batch 2 에서는 수명을 과대 예측하는 오차가 집중되었다. 핵심 개발 문제는 모델 복잡도를 높이는 것보다 새 운전 조건에서 수명으로 변환하는 관계를 검증하는 데 있다.')
rows=[['평가 집단','셀 수','MAPE(%)','MAE(사이클)','Bias(사이클)']]
for r in perf.itertuples():rows.append([r.role,r.n,f'{r.MAPE_pct:.2f}',f'{r.MAE_cycles:.2f}',f'{r.bias_cycles:+.2f}'])
table(rows,[205,45,78,86,94])
p('핵심 발견과 의사결정 기준','sub')
p('첫째, 절대 용량의 높은 통합 상관과 배치 내 신호를 구분했다. QD2의 수명 상관은 배치 중심화 후 거의 사라졌지만 ΔQ는 유지됐다. 둘째, 추가 센서와 비선형 모델의 개발 평균 오차 개선이 관찰되지 않아 단일 Ridge를 선택했다. 셋째, 외부 배치의 오차는 활용 목적과 적용 조건을 나누어 검증해야 한다는 근거를 남겼다.')
p('Batch 2 와 Batch 3 의 MAE는 약 159 사이클로 비슷하다. 그러나 단수명이 많은 Batch 2 는 MAPE32.35%, 장수명이 많은 Batch 3 은 12.74% 다. 상대 오차의 차이를 그대로 기업 적용의 우열로 판단하면 과대 예측에 따른 점검 지연과 과소 예측에 따른 조기 교체를 놓칠 수 있다.')
p('Batch 2 의 89.74% 가 과대 예측, Batch 3 의 학습 최대수명 초과 9 셀은 모두 과소 예측이었다. 따라서 절대 수명 추정, 점검 우선순위, 새 조건의 적용 범위를 각각 검증하는 것이 필요하다.')
p('학습 입력: Q100(V)-Q10(V)에서 구한 log10 표본분산 1 개. 타깃: 원본 MAT의 기록된 cycle_life. CV는 개발 35 셀의 5 폴드 평균, Hold-out은 11 셀, 외부 평가는 전체 Batch 1 46 셀로 재학습한 모델이다.','note')
page('1. EDA 근거와 개발 구현의 연결')
table([['확인한 데이터 특성','구현 선택','검증 질문'],['ΔQ 신호 유지·요약값 간 강한 상관','M1 기준·M4 최솟값·M5 ElasticNet','중복 요약값에도 증분 예측력이 있는가?'],['온도·IR·초기 열화·시간은 다른 측정 역할','M2 물리 5 피처 묶음','센서 추가가 평균 오차를 줄이는가?'],['충전 시간과 평균 전류의 높은 중복','current 묶음은 시간을 RMS·CV로 대체','중복을 줄인 전류 정보의 증분 가치가 있는가?'],['충전 조건별 반복 셀과 작은 표본','정책 단위 Hold-out·GroupKFold','새 정책에 대한 검증이 낙관적이지 않은가?'],['새 배치의 수명 분포 차이','Batch 2·3 별도 평가, 동일 축 그림','평균 점수와 오차 방향이 함께 달라지는가?']],[155,177,176])
p('배치 평균 차이와 예측 신호의 구분','sub')
p('QD2-수명 Pearson 상관은 유효 타깃 129 셀에서 -0.525지만 배치 평균 중심화 후 -0.013으로 약해졌다. ΔQ 로그 분산은 전체 -0.851, 중심화 후 -0.773으로 관계가 유지됐다. 높은 통합 상관만으로 초기 절대 용량을 선택하면 배치 차이를 예측 신호로 해석할 수 있다. 변화량을 기준 신호로 두고 추가 입력의 기여를 정책 단위 검증에서 판단하는 이유다. 중심화는 배치 평균을 구분하는 진단이며 인과 효과를 제거한 실험은 아니다.')
p('피처 설계와 관측 시점','sub')
p('모든 학습 입력은 실제 사이클 라벨 2~100 에서 계산한다. ΔQ는 동일 셀의 실제 Vdlin 전압 축에 정렬된 Q100-Q10 이다. 원본 전압 범위는 2~3.5V이며 전류 I는 C-rate, 시간 t는 분으로 해석한다. 차분은 같은 셀의 기준 곡선을 사용해 배치별 절대 곡선 시작값의 영향을 줄인다.')
p('knee와 종료 용량은 전체 궤적을 해석하는 진단값이다. 피처 목록에는 초기 신호와 충전 정책을 사용하며, 현재 모델은 logvar_deltaQ 한 개로 구성된다. 원본에서 재추출한 피처 139행 31열은 탐색적 분석에 사용한 피처 표와 일치했다.')
p('ΔQ 요약값의 증분 가치와 문헌 기반 검증','sub')
p('ΔQ 로그 분산과 최솟값의 강한 상관은 중복 가능성을 뜻하지만, 최솟값의 증분 예측력이 없다는 증거는 아니다. 초기 Q100−Q10에서 이미 추출한 `min_deltaQ`를 로그 분산에 추가한 delta_pair와 물리·전류 묶음 확장을 비교했다. 상관된 피처의 계수를 정규화하고 선택할 수 있는 ElasticNet도 동일 분할에서 검증했다. 관측 시점·타깃·표본·품질 규칙은 유지했다.')
p('평가 지표의 선정 근거','sub')
p('설계에서 검토한 MAE는 사이클 단위의 오차와 교체 일정의 영향을 해석하는 보조 지표로 유지했다. 개발 단계에서는 수명 규모가 다른 셀의 상대 오차를 비교하기 위해 평균 MAPE를 모델 선택의 주 지표로 확정했다. 같은 100사이클 오차도 실제 수명 400사이클에서는 25%, 1000사이클에서는 10%다. 후보 선택은 Batch 1 개발 CV의 폴드 평균 MAPE에 따르고, MAE·폴드 변동성·과대 예측 비율로 운영 위험을 함께 해석한다.')
page('2. 학습 집단과 정책 단위 검증 구조')
table([['역할','표본·분할','사용 목적'],['개발 집단','Batch 1 35 셀·18 정책','후보·피처 묶음·하이퍼파라미터 선택'],['Train (Batch 1 CV)','개발 집단 내 5-fold GroupKFold','폴드 MAPE의 단순 평균'],['Valid (Batch 1 Hold-out)','Batch 1 11 셀·5 정책','고정 후보의 새 정책 오차 확인'],['최종 재학습','Batch 1 전체 46 셀','설정을 유지하고 학습 표본 확대'],['Test (Batch 2)','39 셀 / 원본 47 셀','고정 모델의 외부 평가'],['Test (Batch 3)','44 셀 / 원본 46 셀','수명 분포가 다른 배치의 추가 검증']],[151,175,182])
p('분할과 전처리의 순서','sub')
p('seed=42, 정책 단위 20% Hold-out을 먼저 분리한 뒤 개발 집단에서 5 폴드 교차검증을 구성했다. 개발·Hold-out 간 충전 정책 중복은 0 이고, 각 CV 폴드의 학습·검증 정책 중복도 0 이다. 결측 중앙값과 표준화 평균·분산은 각각의 학습 폴드에서 추정한다. 테스트 데이터로 전처리를 재적합하지 않는다.')
p('해석의 전제','sub')
p('정답이 결측인 10 셀은 점수 계산의 분모에서 분리했다. 전체 종결 용량으로 학습 대상을 선별하면 초기 시점 수명 예측에서 미래 정보를 활용한 선택 편향이 생길 수 있어, 주 평가는 기록된 유효 타깃 129 셀을 유지했다.')
p('기존 Hold-out·Batch 2·3 점수와 모든 배치 EDA를 확인한 이력이 있다. 후보 확장 범위·그리드·선정 규칙을 실행 전에 고정하고 Batch 1 개발 CV만으로 선택했다. 재사용 집단의 결과는 후속 비교이며, 독립적인 개선 확인에는 새 검증 집단이 필요하다. 이 결과는 탐색 이력을 가진 데이터에서 고정 절차의 적용 가능성을 확인한 평가이며, 새 독립 배치의 검증이 성능 추정의 신뢰성을 보완한다. 파일·행 수준 cell_id는 고유하지만 MATLAB 객체 참조인 barcode/channel로 물리적 셀 독립성을 검증하기 어렵다.')
page('3. 후보 비교와 최종 모델의 선정 근거')
best=cv.sort_values('CV_MAPE_pct').groupby('candidate',sort=True).head(1).sort_values('candidate')
rows=[['후보','최선 입력·타깃','CV MAPE(%)','SD(%p)','CV MAE']]
for r in best.itertuples():rows.append([{'M0':'M0 중앙값','M1':'M1 단일 Ridge','M2':'M2 다변량 Ridge','M3':'M3 Random Forest','M4':'M4 ΔQ 확장 Ridge','M5':'M5 ElasticNet'}[r.candidate],f'{r.features} / {r.target}',f'{r.CV_MAPE_pct:.2f}',f'{r.CV_MAPE_sd:.2f}',f'{r.CV_MAE_cycles:.2f}'])
table(rows,[138,148,90,60,72]);chart('batch1_candidates.png',width=425)
p('선정 결과: 단일 ΔQ Ridge, raw 타깃, alpha=1','sub')
p('275 개 설정,1375 회 CV 학습에서 M1 평균 MAPE7.12% 가 가장 낮았다. best 평균+SD/√5 인 8.46% 이내에서 Ridge·ElasticNet을 같은 선형 계열 우선순위로 두고 평균 MAPE가 가장 낮은 설정을 선택했다. 이 규칙은 모델 복잡도 제어 기준이며 통계적 동등성을 입증하는 검정은 아니다.')
p('M2 물리 5피처 7.37%(SD1.45%p), M3 RF 9.77%에 이어 ΔQ 확장 Ridge 7.24%, ElasticNet 7.18%를 확인했다. 최솟값과 정규화 방식의 추가가 단일 기준의 평균 오차 개선으로 이어지지 않았다. 낮은 M2 변동성은 별도 재검토 근거이며 작은 점수 차이를 유의한 우열로 해석하지 않는다.')
p('M4 최선: delta_pair/log10/alpha=1. M5 최선: delta_pair/raw/alpha=0.01/l1_ratio=0.1. 기존 65설정+확장 Ridge30+ElasticNet180. ElasticNet 6묶음·타깃별 5 alpha·3 l1_ratio; 정확한 그리드는 protocol.json에 기록했다. Severson et al.(2019)의 정규화·최솟값 활용을 참고했고 기존 초기 피처와 분할을 유지했다.','note')
page('4. 수명 예측 성능과 배치 간 일반화')
rows=[['구분','MAPE(%) / Gap(%p)','수식·해석']]
for r in reporting.itertuples(index=False):rows.append([r[0],f'{r[1]:+.2f}' if r[2]=='%p' else f'{r[1]:.2f}',r[3]])
table(rows,[191,118,199])
p('Gap의 부호와 비교 기준','sub')
p('Gap은 각 항목의 앞 평가값에서 뒤 평가값을 뺀 차이로 정의했다. MAPE는 낮을수록 좋으므로 음수 Gap은 뒤 집단의 오차 증가를 뜻한다. Batch 2의 MAPE는 논문 비교값 9.1%보다 23.25%p, Batch 3은 3.64%p 높다. 내부 검증과 외부 평가의 차이는 모델의 적용 범위를 판단하는 근거로 사용한다.')
p('원논문과의 차이','sub')
p('본 분석의 Batch 2는 2018-02-20 파일이다. 논문 저자 LoadData.m은 2017-06-30 파일을 사용하고 Batch 1의 연속 실험을 병합하며, Batch 3에 별도 품질 필터를 적용한다. 따라서 9.1%는 문헌의 비교값으로 해석한다. 실험 집단과 데이터 처리의 차이를 고려할 때 이 Gap은 동일 실험의 재현 오차보다 서로 다른 분석 조건에서의 성능 차이를 나타낸다.')
p('Train은 후보 선택 후 CV 추정치여서 선택 낙관성이 남는다. CV 단순평균 7.12% 와 셀 가중 OOF MAPE7.19% 는 분모가 다르다. Hold-out과 외부 평가의 학습 표본 수는 각각 35 셀과 46 셀로 다르므로 Gap에 순수한 배치 효과만 들어 있다고 볼 수 없다.','note')
page('5. Batch 1 의 정책 변화와 내부 검증 오차')
chart('batch1_prediction.png');chart('batch1_errors.png')
p('CV보다 큰 Hold-out 오차의 의미','sub')
p('정책 분리 Hold-out11 셀에서 MAPE12.72%, MAE118.61 사이클을 보였다.35 셀의 CV MAPE7.12% 보다 5.61%p 높은 오차는 개발 집단의 좋은 점수를 새 정책 전체로 확대하기 어렵다는 근거다. 충전 정책 구성과 학습 표본 수의 차이가 함께 작용하므로 이를 과적합 하나로 단정하지 않는다.')
p('b1c0·b1c1 은 실제 1179~1190 사이클을 약 1448 사이클로 과대 예측했다. 이 두 셀은 Hold-out에 들어간 3.6C(80%)-3.6C 정책이다. 낮은 평균 오차와 별개로 정책별 교정과 완전한 EOL 라벨 검토가 필요하다.','note')
page('6. Batch 2 의 수명 과대 예측과 점검 지연 위험')
chart('batch2_prediction.png');chart('batch2_errors.png')
p('단수명 영역에서 집중되는 오차','sub')
p('39 셀 중 35 셀(89.74%)이 과대 예측이고,28 셀(71.79%)은 실제 수명보다 20% 이상 높게 예측됐다. 학습 최단수명 534 사이클보다 짧은 30 셀은 MAPE37.81%, MAE169.49 사이클이며 모두 과대 예측이다. 학습 수명 범위 9 셀의 MAPE는 14.14% 다.')
p('단순 중앙값 기준의 Batch 2 MAPE72.66% 보다 개선되었지만 최종 모델의 32.35% 는 절대 교체 시점 산정에 큰 차이를 남긴다. 점검 지연 가능성을 관리하려면 단수명 과대 예측을 별도 손실로 평가해야 한다. 실제 수명 구간은 사후 진단이며 운영에서 새 셀의 정답을 미리 알 수 없다. 정책·구조·온도와 교정 불확실성을 관측 기준으로 검증한다.','note')
page('7. Batch 3 의 장수명 영역과 조기 교체 가능성')
chart('batch3_prediction.png');chart('batch3_errors.png')
p('상대 오차 감소와 절대 오차의 분리','sub')
p('Batch 3 MAPE12.74% 는 Batch 2 보다 낮지만 MAE159.41 사이클은 비슷하고 RMSE260.81 사이클은 더 크다. 장수명 셀에서 큰 오차가 남아 평균 상대 오차만으로 일반화 수준을 판단하기 어렵다.')
p('학습 최대 1227 사이클을 초과하는 9 셀은 모두 과소 예측이며, MAPE26.59%·MAE450.26 사이클이다. 학습 수명 범위 35 셀의 MAPE9.17%·MAE84.61 사이클과 대조된다. 장수명 개발 표본 확보와 구간별 bias·예측구간 검증이 교체 계획의 판단 기준이다. 조기 교체는 오차 방향이 만드는 가능성이며 실제 교체 비용을 측정한 결과는 아니다.','note')
page('8. 큰 오차 셀의 공통 조건과 원인 가설')
rows=[['Batch·셀','실제→예측(사이클)','APE(%)','충전 정책']]
for b in [2,3]:
 for r in pred[pred.batch==b].nlargest(3,'APE_pct').itertuples():rows.append([f'B{b} {r.cell_id}',f'{r.cycle_life:.0f} → {r.prediction:.0f}',f'{r.APE_pct:.2f}',r.charging_policy])
table(rows,[81,122,64,241])
p('Batch 2: 동일 초기 신호의 다른 수명 대응','sub')
p('가장 큰 상대 오차는 b2c6·b2c15 의 3.6C(9%)-5C 정책과 b2c18 의 5.2C(50%)-4.25C 정책에서 나타났다. 이 셀들의 실제 수명은 393~449 사이클인데 678~753 사이클로 추정했다. 이 세 셀의 ΔQ 입력은 학습 범위 안에 있다. 전체 Batch 2 에서는 28/39 셀이 범위 안이며 해당 집단도 MAPE34.90%, 평균 편향+152.09 사이클이다. 입력 범위 검사만으로 큰 오차를 걸러내기 어렵다.')
p('Batch 3: 새 구조 정책과 긴 수명','sub')
p('b3c38·b3c7·b3c45 의 실제 수명은 1801~1935 사이클이고, newstructure가 표기된 정책에서 큰 과소 예측이 관찰된다. 셀 구조·운전 정책·수집 시기의 차이가 수명 대응 관계에 영향을 줄 수 있다는 가설이지만, 배치별 자료만으로 각 요인의 인과 효과를 분리할 수 없다.')
p('개발 판단','sub')
p('한쪽 배치의 편향을 보정하는 상수나 타깃 변환을 지금 테스트 점수에 맞춰 선택하면 현재 점수는 최종 검증의 역할을 잃는다. 개선 모델은 새 구조·새 정책·수명 범위를 포함한 개발 데이터로 구성하고 독립된 검증 집단에서 판단해야 한다.')
page('9. 초기 신호의 범위와 수명 대응 관계')
chart('batch1_signal_mapping.png');chart('batch2_signal_mapping.png');chart('batch3_signal_mapping.png')
p('세 그림에는 같은 Batch 1 학습식을 표시했다. logvar_deltaQ가 학습 범위 안에 있어도 Batch 2 단수명 셀은 선 아래, Batch 3 일부 장수명 셀은 선 위에 놓인다. Batch 2 의 범위 안 28 셀에서도 MAPE34.90%가 남는다. 입력 범위 검사는 정확도 보장이 아니다. 운전 정책과 구조별 대응 관계의 검증이 추가로 필요하다.','note')
page('10. 품질 민감도와 성능 추정의 불확실성')
quality=pd.read_csv(R/'batch3_quality_sensitivity.csv')
table([['Batch 3 평가 코호트','셀 수','MAPE(%)','MAE(사이클)']]+[[r.cohort,r.n,f'{r.MAPE_pct:.2f}',f'{r.MAE_cycles:.2f}'] for r in quality.itertuples()],[267,49,96,96])
p('저자 품질 규칙의 진단 적용','sub')
p('원본 46 셀 순서에서 수집 문제 b3c37 을 표기하고, 종료 용량 필터 뒤의 저자 인덱스를 b3c2·b3c42·b3c43 에 대응시켰다. 동일한 고정 모델의 민감도 40 셀 MAPE12.78% 는 주 평가 44 셀 12.74% 와 유사하다. 이 데이터에서는 품질 필터만으로 외부 오차가 크게 줄어든다는 근거가 약하다.')
rows=[['평가 집단','정책 bootstrap95% MAPE 구간','해석']]
for r in perf.itertuples():
 if not r.role.startswith('Train'):rows.append([r.role,f'{r.cluster_CI_low_pct:.2f}~{r.cluster_CI_high_pct:.2f}%','정책을 복원 추출한 집계 오차 구간'])
table(rows,[169,174,165])
p('품질 정제와 적용 대상 보존','sub')
p('EDA에서 Batch 2 IQR 표시 9 셀을 일괄 제외하면 장수명 3 셀이 모두 사라진다. 통계적으로 드문 수명과 측정 오류를 구분하지 않으면 중요한 적용 영역을 없앤 채 성능을 평가하게 된다. 품질은 측정 이력으로 검토하고 정상적인 극단 수명은 검증 대상으로 보존한다.')
p('표본·라벨·물리 식별자의 제약','sub')
p('Batch 1 의 13 셀은 종료 용량이 0.885Ah보다 높다. 기록된 cycle_life를 모든 셀의 80% EOL로 동일하게 해석하기 어렵고, 총수명 라벨의 완료 여부를 확인할 필요가 있다. 원본 MATLAB 객체 참조인 barcode/channel은 문자열로 직접 해독되지 않아 파일 간 동일한 물리 셀인지 검증하는 데 한계가 있다.')
p('bootstrap 구간은 해당 배치의 정책·셀 구성에 따른 점수 변동을 나타낸다. 모델 선택, 새 배치의 분포 변화, 개별 셀 예측 불확실성을 모두 반영하는 구간이 아니다. 중요도는 11 셀 Hold-out의 탐색적 진단이며, 단일 피처의 안정성이나 인과적 효과를 입증하지 않는다.','note')
page('11. 기업 운영 적용과 개선 우선순위')
p('절대 수명과 상대 점검 순위의 분리','sub')
p('Batch 2·3 의 수명 순위 Spearman은 0.709·0.797 이다. 상대적 점검 우선순위에 활용할 가능성은 있지만 상위 위험 셀 탐지율과 실제 운영 비용 효과를 보장하지 않는다. 절대 수명 추정은 Batch 2 의 과대 예측과 Batch 3 의 장수명 과소 예측을 별도로 교정·검증해야 한다.')
table([['운영 목적','이번 결과의 함의','현장 검증 기준'],['점검 우선순위','수명 순위 정보가 남음','상위 위험 셀 탐지율, 점검 예산별 적중률'],['교체 일정 수립','짧은 수명 과대 예측·긴 수명 과소 예측','수명 구간별 bias·과대 예측 손실·예측구간'],['새 조건 적용','입력 범위 안에서도 대응 관계 변화','정책·구조·온도별 신규 배치 평가'],['운영 시스템 연동','100 사이클 관측이 필요','데이터 수집률, 피처 결측률, 초기 선별 소요시간']],[107,184,217])
p('개선 순서','sub')
p('첫째, 완전한 EOL 라벨과 신뢰 가능한 물리 셀ID를 확보하고, 학습 수명 범위 밖의 단수명·장수명 셀을 개발 데이터에 보강한다. 둘째, 정책·셀 구조별 교정 효과를 새 검증 집단에서 확인한다. 셋째, 실제 ESS의 SOC·DoD·온도·캘린더 열화·팩 불균형 조건에서 초기 100 사이클의 입력 가용성과 평가 목적을 검증한다.')
p('이번 개발의 결론','sub')
p('ΔQ 신호를 중심으로 단순한 회귀 모델을 선택한 결정은 개발 데이터에서 지지되었다. 추가 센서·ΔQ 최솟값·ElasticNet의 평균 개선은 개발 CV에서 관찰되지 않았고, 고정된 단일 모델의 외부 결과는 수명 범위와 배치별 교정 문제를 드러냈다. 두 관찰을 구분해 입력의 증분 가치와 적용 조건을 각각 검증해야 한다. 기업 관점의 가치는 예측값 하나를 제시하는 데서 끝나지 않고, 어떤 조건에서 점검 순위와 교체 계획을 신뢰할 수 있는지 평가 기준으로 연결하는 데 있다.')
output=ROOT/'reports/model_development_evaluation.pdf'
SimpleDocTemplate(str(output),pagesize=(595,842),leftMargin=43,rightMargin=44,topMargin=42,bottomMargin=36,title='ESS 배터리 수명 예측 모델 개발 및 평가',author='').build(story)
print(output)
