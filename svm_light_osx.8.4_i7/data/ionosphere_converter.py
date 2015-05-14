from sys import argv
from itertools import izip

script, fileName = argv
point = 1

outputFile = open('ionosphere_'+fileName,'w')

with open(fileName,'r') as inputFile:
	for line in inputFile:
		line = line.strip()
		parts = line.split(',')
		outputFile.write(str(parts[0])+' ')
		for i in xrange(len(parts)-1):
			outputFile.write(str(i+1)+':'+str(parts[i+1])+' ')
		outputFile.write('\n')
		point += 1
outputFile.close()