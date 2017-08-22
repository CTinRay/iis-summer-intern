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
def filter(file, filename):
	'''
	if Body and Question have exactly two numbers, then we preserve this datum
	if Answer has only one number, then we preserve this datum
	'''
	answers, bodys, questions = file["Answer"], file["Body"], file["Question"]
	new_answer = []
	new_bodys = []
	new_questions = []
	for answer, body, question in zip(answers, bodys, questions):
		pat = r'[0-9.\-]+'
		ans_number_list = re.findall(pat, answer)
		body_number_list = re.findall(pat, body)
		question_number_list = re.findall(pat, question)
		if len(ans_number_list) == 1 and (len(body_number_list) + len(question_number_list) == 2):
			new_answer.append(answer)
			new_bodys.append(body)
			new_questions.append(question)
	output = pd.DataFrame({"Body":new_bodys, "Answer":new_answer, "Question":new_questions})
	output.to_csv(SAVE_DIR + filename + ".filtered", index = False)

def set_parser(prefix):
	os.chdir(SAVE_DIR)
	for i in range(1, 6+1):
		j = 1
		while True:
			filename = prefix + "G{}_{:02d}.xml.csv".format(i,j)
			if os.path.isfile(filename):
				file = readfile(filename)
				filter(file, filename)
			else:
				break
			j += 1
	os.chdir(os.pardir)

def main():
	set_parser(FILEPREFIX)
	set_parser(FILEPREFIX2)
	set_parser(FILEPREFIX3)

if __name__ == '__main__':
	#file = readfile(SAVE_DIR+"MR_MATH_Train_G1_01.xml.csv")
	#filter(file)
	main()