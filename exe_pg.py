import subprocess
import sys

subprocess.call(['./Translator ' + str(sys.argv[1])], shell=True)
