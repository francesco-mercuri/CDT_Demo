import pickle, time, numpy as np
import sim, kg, models

lib,wf=pickle.load(open('lib.pkl','rb'))
g=kg.build_kg(lib,{k:v for k,v in wf.items()})
print('triples',len(g))
orch=kg.KGOrchestrator(g).refresh()
print('workflows loaded',len(orch.wf))
rng=np.random.default_rng(5)
table=kg.contract_table(lib); wfd=orch.wf
agree=0; N=0
for regime,grade,nf in [('in',None,0),('in',None,1),('in',None,2),('out',None,0),('out',None,2),('in','D',0),('in','D',1)]:
    for _ in range(6):
        c=sim.gen_cycle(rng,grade=grade,regime=regime)
        arrs,status,desc=sim.apply_scenario(c,rng,nf)
        r=kg.add_run(g,'R1','Furnace1',c,status)
        sel=orch.select('R1',budget=1.0)
        py=kg.select_py(table,wfd,status,{'PCR':c['pcr'],'TH':c['d']},c['grade'])
        kg.remove_run(g,'R1')
        for s in status: 
            kk=kg.DT[s]; g.set((kk,kg.DT.status,kg.Literal('ok')))
        same=(sel['choice'] is None and py is None) or (sel['choice'] is not None and py is not None and sel['choice'][0]==py[0] and sel['choice'][1]==py[1])
        N+=1; agree+=same
        if _==0: print(regime,c['grade'],nf,status,'->',sel['choice'],'ncand',sel['ncand'],'t=%.1fms'%(sel['t']*1e3),'| py same:',same)
print('SPARQL vs python mirror agreement',agree,'/',N)
