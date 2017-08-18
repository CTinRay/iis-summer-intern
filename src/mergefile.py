import os.path
import pandas as pd
import re
FILEPREFIX = "MR_MATH_Train_"
FILEPREFIX2 = "MR_MATH_Develop_"
FILEPREFIX3 = "MR_MATH_Test_"
SAVE_DIR = "~/MWP_Data/"
def readfile(filename):
	file = pd.read_csv(filename, header = 0)
	#print(type(file))
	return file
def set_parser(prefix):
	os.chdir(os.path.expanduser(SAVE_DIR) )
	df_list = []
	for i in range(1, 6+1):
		j = 1
		while True:
			filename = prefix + "G{}_{:02d}.xml.csv.filtered_three.operands_threenum".format(i,j)
			if os.path.isfile(filename):
				file = readfile(filename)
				df_list.append(file)
			else:
				break
			j += 1
	os.chdir(os.pardir)
	output = pd.concat(df_list)
	if prefix == FILEPREFIX:
		output.to_csv(SAVE_DIR+"TrainSet", index = False)
	elif prefix == FILEPREFIX2:
		output.to_csv(SAVE_DIR+"DevSet", index = False)
	elif prefix == FILEPREFIX3:
		output.to_csv(SAVE_DIR+"TestSet", index = False)
set_parser(FILEPREFIX)
set_parser(FILEPREFIX2)
set_parser(FILEPREFIX3)
