import sys
import re
inStr = sys.argv[1]

print(re.sub("[^\w\d.]","", inStr))  
