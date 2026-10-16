import time, numpy as np, pickle
import sim, models
from sim import gen_cycle, t_end

t0=time.time()
lib=models.Library()
print('library built %.0fs'%(time.time()-t0))
for m,c in lib.contracts.items(): print(m,'latency %.2f ms'%(c['latency']*1e3))
rng=np.random.default_rng(99)
def eval_chain(steps,binding,cyc):
    errs=[]
    for c in cyc:
        arrs={k:c[k] for k in ('TC_A','TC_B','PYRO','PYRO_2')}
        v=lib.run_chain(steps,binding,c,arrs)
        errs.append(np.nan if v is None else v-c['HV'])
    e=np.array(errs); return np.sqrt(np.nanmean(e**2)), np.nanmean(e), np.isnan(e).mean()
chains={
 'T1 lumped>hybrid':(('M_core_lumped','M_HV_hybrid'),{'FTP':'TC_A'}),
 'T2 lumped>phys':(('M_core_lumped','M_HV_phys'),{'FTP':'TC_A'}),
 'T3 FD>hybrid':(('M_core_FD','M_HV_hybrid'),{'FTP':'TC_A'}),
 'T4 FD>phys':(('M_core_FD','M_HV_phys'),{'FTP':'TC_A'}),
 'T5 surr>hybrid':(('M_core_surr','M_HV_hybrid'),{'STP':'PYRO'}),
 'T6 surr>phys':(('M_core_surr','M_HV_phys'),{'STP':'PYRO'}),
 'T7 HVml':(('M_HV_ml',),{'STP':'PYRO'}),
 'T8 HVFD':(('M_HV_FD',),{'FTP':'TC_A'}),
}
sets={      'grade D':[gen_cycle(rng,grade='D') for _ in range(100)]}
for sn,cyc in sets.items():
    print('==',sn)
    for n,(st,b) in chains.items():
        r,bi,nf=eval_chain(st,b,cyc); print('  %-18s RMSE %6.2f  bias %6.2f  none %.2f'%(n,r,bi,nf))
