from sys import argv
from itertools import izip

filename = '2moons'
pos_class_pt = 123
neg_class_pt = 142

'''
filename = 'clock'
pos_class_pt = 127
neg_class_pt = 1
'''

output_file_train = open(filename+'_train.dat','w')
output_file_test = open(filename+'_test.dat','w')
point = 1
with open(filename+'_x','r') as x_file, open(filename+'_y','r') as y_file:
	for x,y in izip(x_file,y_file):
		x = x.strip()
		y = y.strip()
		if point == pos_class_pt:
			output_file_train.write('1 ')
		elif point == neg_class_pt:
			output_file_train.write('-1 ')
		else:
			output_file_train.write('0 ')
		output_file_test.write(str(y)+' ')
		parts = x.split(',')
		for i in xrange(len(parts)):
			output_file_train.write(str(i+1)+':'+str(parts[i])+' ')
			output_file_test.write(str(i+1)+':'+str(parts[i])+' ')
		output_file_train.write('\n')
		output_file_test.write('\n')
		point += 1
output_file_train.close()
output_file_test.close()