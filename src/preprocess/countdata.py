'''
# Count the number of each operator 
'''
import os.path
import pandas as pd
import re
import argparse
import itertools
# train data set prefix
FILEPREFIX = "MR_MATH_Train_"
# development data set prefix
FILEPREFIX2 = "MR_MATH_Develop_"
# test data set prefix
FILEPREFIX3 = "MR_MATH_Test_"
# all operator list
operator_list = []
for r in range(1,2+1):
	for i in itertools.combinations_with_replacement(["+", "-", "*", "/"], r):
		operator_list.append("".join(i))
def readfile(filename):
	file = pd.read_csv(filename, header = 0)
	return file
def count_data(prefix, SAVE_DIR):
	counter = 0
	operator_count = dict(zip(operator_list,[0] * len(operator_list)))
	for i in range(1, 6+1):
		j = 1
		while True:
			filename = os.path.join(SAVE_DIR,prefix +  
				"G{}_{:02d}.xml.csv.filtered_three.operands_threenum".format(i,j))
			if os.path.isfile(filename):
				file = readfile(filename)
				for operator in file['Operand']:
					if operator in operator_list:
						operator_count[operator] += 1
					
				# count row
				counter += file.shape[0]
			else:
				break
			j += 1
	return counter, operator_count

def main():
	parser = argparse.ArgumentParser(description="Count how many operator in data")
	parser.add_argument('--data_dir', type=str, default="/home/jimlin7777/MWP_Data/", 
		help="data directory")
	args = parser.parse_args()
	counter, operator_count = count_data(FILEPREFIX, args.data_dir)
	# Print out operator counter
	print("Training Data has {} data".format(counter))
	for key, value in operator_count.items():
		print("{} : {}".format(key, value))
	counter, operator_count = count_data(FILEPREFIX2, args.data_dir)
	print("Develop Data has {} data".format(counter))
	for key, value in operator_count.items():
		print("{} : {}".format(key, value))
	counter, operator_count = count_data(FILEPREFIX3, args.data_dir)
	print("Test Data has {} data".format(counter))
	for key, value in operator_count.items():
		print("{} : {}".format(key, value))
	


if __name__ == '__main__':
	main()