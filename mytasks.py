

import sys, os

print("IMP00\n")

whereAmI = os.path.dirname(__file__)

print("IMP0\n")

sys.path.append(whereAmI)

print("IMP1\n")

from task_polsolve import polsolve as polsolve

print("IMP2\n")

#from task_polsimulate import polsimulate as polsimulate
#from task_CCextract import CCextract as CCextract
#from poltools_helper import plotPolImage


