import pandas as pd, numpy as np, matplotlib

matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans'})
e1=pd.read_csv('results/e1.csv'); e4=pd.read_csv('results/e4.csv')
e4['label']=np.where(e4.tag=='E4b_pyro2','3 faults, PYRO_2 registered',np.where(e4.tag=='E4b_noreg','3 faults, no spare sensor',e4.label))
d=pd.concat([e1,e4]); 

sc=['in-regime, 0 faults','in-regime, 1 faults','in-regime, 2 faults','out-regime, 0 faults','out-regime, 1 faults','out-regime, 2 faults','new grade D, 0 faults','new grade D, 1 fault','3 faults, no spare sensor','3 faults, PYRO_2 registered']
names={'in-regime, 0 faults':'in-regime\n0 faults','in-regime, 1 faults':'in-regime\n1 fault','in-regime, 2 faults':'in-regime\n2 faults','out-regime, 0 faults':'out-of-\nregime\n0 faults','out-regime, 1 faults':'out-of-\nregime\n1 fault','out-regime, 2 faults':'out-of-\nregime\n2 faults','new grade D, 0 faults':'new grade D\n0 faults','new grade D, 1 fault':'new grade D\n1 fault','3 faults, no spare sensor':'3 faults\nno spare','3 faults, PYRO_2 registered':'3 faults\nspare registered'}
st=[('B1','B1 fixed fallback'),('B2','B2 expert rules'),('S_noregime','S w/o regime check'),('S','S semantic')]
cols=['#9aa5b1','#d99a3a','#c9544d','#2f6f9f']

fig,ax=plt.subplots(2,1,figsize=(8.4,6.0),sharex=True)
w=0.2; x=np.arange(len(sc))
for k,(m,lab) in enumerate(zip(['useful','harmful'],['(a) useful predictions\n(|error| ≤ 10 HV), fraction','(b) harmful predictions\n(|error| > 10 HV), fraction'])):
    for i,(s,sl) in enumerate(st):
        v=[d[(d.label==l)&(d.strategy==s)][m].mean() for l in sc]
        ax[k].bar(x+(i-1.5)*w,v,w,label=sl,color=cols[i])
    ax[k].set_ylabel(lab,fontsize=8); ax[k].set_ylim(0,1.05); ax[k].grid(axis='y',alpha=.3)
ax[0].legend(ncol=4,fontsize=7,loc='lower center',bbox_to_anchor=(.5,1.0),frameon=False)
ax[1].set_xticks(x); ax[1].set_xticklabels([names[l] for l in sc],fontsize=6)
plt.tight_layout(); plt.savefig('/home/claude/work/fig/fig_demo_results.png',dpi=300); plt.close()

# fig 2
det=pd.read_csv('results/e2_detect.csv').set_index('error'); sc3=pd.read_csv('results/e3_scaling.csv')
dn=pd.read_csv('results/e2_downstream.csv')
fig,ax=plt.subplots(1,3,figsize=(7.4,2.9))
lab={'schema':'unknown quantity','unit':'unit/scale error','swap_out':'wrong output class','widen':'validity widened ×2','drop_grade':'grade limit omitted','hall_input':'spurious input'}
im=ax[0].imshow(det.values,vmin=0,vmax=1,cmap='Blues',aspect='auto'); ax[0].set_xticks(range(4)); ax[0].set_xticklabels(['G1','G2','G3','G4'])
ax[0].set_yticks(range(len(det))); ax[0].set_yticklabels([lab[i] for i in det.index],fontsize=7)
for i in range(det.shape[0]):
    for j in range(4): ax[0].text(j,i,'%.2f'%det.values[i,j],ha='center',va='center',fontsize=7,color='w' if det.values[i,j]>.6 else 'k')
ax[0].set_title('(a) detection rate by gate',fontsize=8)
for p,c in zip((0.1,0.2,0.3),('#9aa5b1','#d99a3a','#c9544d')):
    h=[dn[(dn.p==p)&(dn.gate==g)].harmful.mean()*100 for g in ['G0','G1','G2','G3','G4']]
    ax[1].plot(range(5),h,'o-',color=c,label='p=%.1f'%p,ms=3)
ax[1].set_xticks(range(5)); ax[1].set_xticklabels(['none','G1','G2','G3','G4']); ax[1].set_ylabel('harmful predictions [%]'); ax[1].set_ylim(1,7.5); ax[1].legend(fontsize=6.5,title='error rate per contract',title_fontsize=6,loc='upper center',ncol=3,columnspacing=.8,handlelength=1.2); ax[1].set_title('(b) downstream harm',fontsize=8); ax[1].grid(alpha=.3)
ax[2].loglog(sc3.triples,sc3.select_ms_med,'o-',color='#2f6f9f',label='median'); ax[2].loglog(sc3.triples,sc3.select_ms_p95,'s--',color='#c9544d',label='95th pct')
ax[2].set_xlabel('triples in the KG'); ax[2].set_ylabel('selection time [ms]'); ax[2].legend(fontsize=7); ax[2].set_title('(c) SPARQL selection latency',fontsize=8); ax[2].grid(alpha=.3,which='both')
plt.tight_layout(); plt.savefig('/home/claude/work/fig/fig_gates_latency.png',dpi=300); plt.close()
