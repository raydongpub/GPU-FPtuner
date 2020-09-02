import sys
import fileinput
import subprocess
from contextlib import closing
from collections import OrderedDict

block = []

with open(sys.argv[1], "r") as stmt_list:
  for stmt in stmt_list:
    if "block:" in stmt:
      block.append(stmt.replace("block:", ""))

with open(sys.argv[2], "w") as list:
  for t in block:
    list.write(t)

