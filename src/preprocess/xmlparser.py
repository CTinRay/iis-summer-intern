import xml.etree.ElementTree as ET
import os.path
import pandas as pd
import argparse
def parse(DIR, filename, SAVE_DIR):
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
	tree = ET.parse(os.path.join(DIR, filename))
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
	output.to_csv(os.path.join(SAVE_DIR, filename+".csv"),index = False)

def set_parser(NOW_DIR, PREFIX, SAVE_DIR):
	for i in range(1, 6+1):
		NOW_GRADE_DIR = "G{}".format(i)
		j = 1
		while True:
			FILE_NAME = "{}G{}_{:02d}.xml".format(PREFIX,i,j)
			if os.path.isfile(os.path.join(NOW_DIR, NOW_GRADE_DIR,FILE_NAME)):
				parse(os.path.join(NOW_DIR, NOW_GRADE_DIR), FILE_NAME, SAVE_DIR)
				j += 1
			else:
				break
def main():
	# default path
	TRAIN_PATH = "/home/jimlin7777/xmlfile/bug.iis.sinica.edu.tw/seminar/MachineReading/Corpus/Elementary_Mathematics_v1.2.1/Training Set"
	DEV_PATH = "/home/jimlin7777/xmlfile/bug.iis.sinica.edu.tw/seminar/MachineReading/Corpus/Elementary_Mathematics_v1.2.1/Develop Set"
	TEST_PATH = "/home/jimlin7777/xmlfile/bug.iis.sinica.edu.tw/seminar/MachineReading/Corpus/Elementary_Mathematics_v1.2.1/Test Set"
	FILEPREFIX = "MR_MATH_Train_"
	FILEPREFIX2 = "MR_MATH_Develop_"
	FILEPREFIX3 = "MR_MATH_Test_"
	SAVE_DIR = "/home/jimlin7777/MWP_Data"
	# arg parser
	parser = argparse.ArgumentParser(description="Parse xml file")
	parser.add_argument('--save_dir', type=str, default=SAVE_DIR,
		help="directory where you want to save parsed file")
	parser.add_argument('--xml_train_dir', type=str, default=TRAIN_PATH,
		help="directory where your xml train data set locate")
	parser.add_argument('--xml_dev_dir', type=str, default=DEV_PATH,
		help="directory where your xml dev data set locate")
	parser.add_argument('--xml_test_dir', type=str, default=TEST_PATH,
		help="directory where your xml test data set locate")
	args = parser.parse_args()

	set_parser(args.xml_train_dir, FILEPREFIX, args.save_dir)
	set_parser(args.xml_dev_dir, FILEPREFIX2, args.save_dir)
	set_parser(args.xml_test_dir, FILEPREFIX3, args.save_dir)


if __name__ == '__main__':
	main()