import pickle, time, numpy as np
import models, kg
t0=time.time()
lib=models.Library()
wf=kg.calibrate_workflows(lib)
pickle.dump((lib,wf),open('lib.pkl','wb'))
print('built in %.0fs'%(time.time()-t0))
for k,(s,e) in wf.items(): print('%-28s %5.2f'%(k,e))
for m,c in lib.contracts.items(): print(m,'%.2f ms'%(c['latency']*1e3))
