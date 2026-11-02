"""Replot selected published figures from CSV; no model, Torch or GPU required."""
import argparse,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
COL={'teacher':'#263d53','hc':'#c75730','reap':'#007c91'}
def read(name):return list(csv.DictReader((R/name).open()))
def save(fig,out,name):
 fig.tight_layout();fig.savefig(out/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(out/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='reproduced');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 rows=read('results/visualization/vector_values.csv');fig,axes=plt.subplots(2,1,figsize=(15,7))
 for ax,li in zip(axes,[3,15]):
  for m in ['teacher','hc','reap']:
   q=sorted([v for v in rows if int(v['layer'])==li and v['model']==m],key=lambda v:int(v['feature']));ax.plot([int(v['feature']) for v in q],[float(v['value']) for v in q],color=COL[m],label=m,lw=1.25)
  ax.set_title(f'Layer{li}, sequence0, token32: identical teacher input');ax.set_xlabel('Hidden feature index');ax.set_ylabel('Raw branch output');ax.axhline(0,color='grey',lw=.6);ax.grid(alpha=.15)
 axes[0].legend(ncol=3,frameon=False);save(fig,out,'output_vector')
 rows=read('results/diagnosis/output_interpolation.csv');fig,axes=plt.subplots(1,2,figsize=(11,4.1),sharey=True)
 for ax,split in zip(axes,['original','independent']):
  for kind,col in [('lora','#237b93'),('teacher','#c56537')]:
   q=sorted([v for v in rows if v['method']=='hc' and v['layer']=='15' and v['split']==split and v['correction']==kind],key=lambda v:float(v['alpha']));x=[float(v['alpha']) for v in q]
   ax.plot(x,[float(v['actual_gain']) for v in q],'o-',color=col,label=kind);ax.fill_between(x,[float(v['gain_ci95_low']) for v in q],[float(v['gain_ci95_high']) for v in q],color=col,alpha=.12)
  ax.set_title(split);ax.set_xticks([0,.25,.5,.75,1,1.25]);ax.set_xlabel('Branch correction alpha');ax.axhline(0,color='grey',lw=.7);ax.axvline(1,color='grey',ls='--',lw=.7);ax.grid(alpha=.15)
 axes[0].set_ylabel('NLL recovery gain');axes[0].legend(frameon=False);save(fig,out,'hc_l15_interpolation')
 rows=read('results/visualization/all_layer_metrics.csv');fig,axes=plt.subplots(1,3,figsize=(15,4.4))
 for ax,key in zip(axes,['relative_l2','cosine_similarity','output_norm_ratio']):
  for m in ['hc','reap']:
   q=sorted([v for v in rows if v['method']==m],key=lambda v:int(v['layer']));ax.plot([int(v['layer']) for v in q],[float(v[key]) for v in q],'-o',color=COL[m],label=m,ms=4)
  ax.set_title(key);ax.set_xlabel('Layer');ax.grid(alpha=.15)
 axes[0].legend(frameon=False);save(fig,out,'output_layers')
 print(f'Regenerated3 figures (PNG/PDF) in {out}. No model execution.')
if __name__=='__main__':main()
