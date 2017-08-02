import os.path
import pandas as pd
import re
TRAIN_PATH = "Training Set/"
DEV_PATH = "Develop Set/"
TEST_PATH = "Test Set/"
FILEPREFIX = "MR_MATH_Train_"
FILEPREFIX2 = "MR_MATH_Develop_"
FILEPREFIX3 = "MR_MATH_Test_"
SAVE_DIR = "/home/jimlin7777/MWP_Data/"
def readfile(filename):
	file = pd.read_csv(filename, header = 0)
	return file
def count_data(prefix):
	os.chdir(SAVE_DIR)
	counter = 0
	for i in range(1, 6+1):
		j = 1
		while True:
			filename = prefix + "G{}_{:02d}.xml.csv.filtered.operands".format(i,j)
			if os.path.isfile(filename):
				file = readfile(filename)
				# count row
				counter += file.shape[0]
			else:
				break
			j += 1
	os.chdir(os.pardir)
	return counter

def main():
	print("Training Data has {} data".format(count_data(FILEPREFIX)))
	print("Develop Data has {} data".format(count_data(FILEPREFIX2)))
	print("Test Data has {} data".format(count_data(FILEPREFIX3)))

if __name__ == '__main__':
	main()