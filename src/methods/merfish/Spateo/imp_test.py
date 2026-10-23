import time, sys
import statsmodels.sandbox.stats.multicomp as mc
from statsmodels.stats.multitest import multipletests

mc.multipletests = multipletests
t = time.time()
import spateo, torch

print("spateo", spateo.__file__, "import_sec", time.time() - t, flush=True)
print("cuda", torch.cuda.is_available(), torch.cuda.device_count(), flush=True)
from spateo.tdr import deep_intepretation

print(deep_intepretation, flush=True)
