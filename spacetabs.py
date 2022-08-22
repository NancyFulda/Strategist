import sys

f=open(sys.argv[1],'r')
data = f.read()
f.close()
data = data.replace('\t','        ')

f=open(sys.argv[1],'w')
f.write(data)
f.close()
