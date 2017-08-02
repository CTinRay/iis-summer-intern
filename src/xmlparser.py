import xml.etree.ElementTree as ET
import os.path
import pandas as pd
FILE_PATH = "/home/jimlin7777/xmlfile/bug.iis.sinica.edu.tw/seminar/MachineReading/Corpus/Elementary_Mathematics_v1.2.1/"
TRAIN_PATH = "Training Set/"
DEV_PATH = "Develop Set/"
TEST_PATH = "Test Set/"
FILEPREFIX = "MR_MATH_Train_"
FILEPREFIX2 = "MR_MATH_Develop_"
FILEPREFIX3 = "MR_MATH_Test_"
SAVE_DIR = "/home/jimlin7777/MWP_Data/"
def parse(filename):
	'''
	Note: Parse a xml file to csv file with triplet(Body, Question, Answer)
	- Machine-Reading-Corpus-File
		- Content
			-Unit
				-Body
				-QA
					-Question
					-Answer
	'''
	# initial list for pandas DataFrame
	bodylist = []
	questionlist = []
	answerlist = []
	tree = ET.parse(filename)
	root = tree.getroot()
	Content = root[0]
	for unit in Content:
		for child in unit:
			if child.tag == "Body":
				bodyname = child.text
			elif child.tag == "QA":
				bodylist.append(bodyname.encode('utf-8'))
				for grandchild in child:
					if grandchild.tag == "Question":
						questionlist.append((grandchild.text).encode('utf-8'))
					elif grandchild.tag == "Answer":
						answerlist.append((grandchild.text).encode('utf-8'))
			
	output = pd.DataFrame({'Body':bodylist, 'Question':questionlist, 'Answer':answerlist})
	output.to_csv(SAVE_DIR+filename+".csv",index = False)
def set_parser(path):
	# Train Set
	NOW_DIR = FILE_PATH + path
	if path == TRAIN_PATH:
		PREFIX = FILEPREFIX
	elif path == DEV_PATH:
		PREFIX = FILEPREFIX2
	elif path == TEST_PATH:
		PREFIX = FILEPREFIX3
	os.chdir(NOW_DIR)
	for i in range(1, 6+1):
		NOW_PATH = "G{}/".format(i)
		os.chdir(NOW_PATH)
		j = 1
		while True:
			FILE_NAME = "{}G{}_{:02d}.xml".format(PREFIX,i,j)
			if os.path.isfile(FILE_NAME):
				parse(FILE_NAME)
				j += 1
			else:
				break
		os.chdir(os.pardir)
	os.chdir(os.pardir)
def main():
	set_parser(TRAIN_PATH)
	set_parser(DEV_PATH)
	set_parser(TEST_PATH)


if __name__ == '__main__':
	'''
	NOW_DIR = FILE_PATH + TRAIN_PATH
	os.chdir(NOW_DIR)
	os.chdir("G1/")
	parse(FILEPREFIX + "G1_01.xml")
	os.chdir(os.pardir)
	'''
	main()