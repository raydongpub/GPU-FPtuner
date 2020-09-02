import sys
import fileinput
import subprocess
from contextlib import closing
from collections import OrderedDict

time = []
error = []
kernel = []

with open(sys.argv[1], "r") as stmt_list:
  for stmt in stmt_list:
    if "time:" in stmt:
      time.append(stmt.replace("time:", ""))
    if "error:" in stmt:
      error.append(stmt.replace("error:", ""))
    if "kernel:" in stmt:
      kernel.append(stmt.replace("kernel:", ""))

with open(sys.argv[2], "w") as list:
  for t in time:
    list.write(t)
with open(sys.argv[3], "w") as list:
  for e in error:
    list.write(e)
with open(sys.argv[4], "w") as list:
  for k in kernel:
    list.write(k)
