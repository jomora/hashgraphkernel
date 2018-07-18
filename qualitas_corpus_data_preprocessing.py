
# Execute this in the qualitas corpus  base directory after running the install
# script located in the bin folder

from subprocess import Popen, PIPE
from shlex import split

p1 = Popen(split("find . -name *.jar"), stdout=PIPE)
p2 = Popen(split("xargs -I % basename %"), stdin=p1.stdout,stdout=PIPE)
stdout, stderr = p2.communicate()

jars = stdout.split("\n")
import collections
jarCounters =  [(item,count) for item, count in collections.Counter(jars).items() ]

print [proj for proj,count in jarCounters if count > 18]

for i in range(0,30):
    print(str(i) + ": "+str(len([proj for proj,count in jarCounters if count > i])))
