import os.path
import pandas as pd
import re
import argparse
def readfile(filename):
	file = pd.read_csv(filename, header = 0)
	return file

def filter(SAVE_DIR, file, filename):
	'''
	if Body and Question have exactly two numbers, then we preserve this datum
	if Answer has only one number, then we preserve this datum
	'''
	answers, bodys, questions = file["Answer"], file["Body"], file["Question"]
	new_answer = []
	new_bodys = []
	new_questions = []
	operands = []
	counter_unknown_op = 0
	for answer, body, question in zip(answers, bodys, questions):
		pat = r'[0-9.\-]+'
		ans_number_list = re.findall(pat, answer)
		body_number_list = re.findall(pat, body)
		question_number_list = re.findall(pat, question)
		is_op, op = bruteforce_find_operand(ans_number_list, body_number_list, question_number_list)
		if is_op:
			new_answer.append(answer)
			new_bodys.append(body)
			new_questions.append(question)
			operands.append(op)
		else:
			counter_unknown_op += 1

	output = pd.DataFrame({"Body":new_bodys, 
		"Answer":new_answer, "Question":new_questions, "Operand": operands})

	output.to_csv(os.path.join(SAVE_DIR ,filename + ".operands"), index = False)

def bruteforce_find_operand(ans_number_list, body_number_list, question_number_list):
	assert len(ans_number_list) == 1
	assert (len(body_number_list) + len(question_number_list)) == 2
	ans = ans_number_list[0]
	x, y = (body_number_list + question_number_list)[0], (body_number_list + question_number_list)[1]
	x = num(x)
	y = num(y)
	ans = num(ans)
	counter = 0
	operand_list = []
	if x != None and y != None and ans != None:
		if isclose(x + y, ans):
			counter += 1
			operand_list.append("+")
		if isclose(x * y, ans):
			counter += 1
			operand_list.append("*")
		if isinstance(x, int) and isinstance(y, int) and isinstance(ans, int):
			if x >= y:
				if y != 0:
					if x % y == ans:
						counter += 1
						operand_list.append("%")
					if x / y == ans:
						counter += 1
						operand_list.append("//")
			else:
				if x != 0:
					if y % x == ans:
						counter += 1
						operand_list.append("%")
					if y / x == ans:
						counter += 1
						operand_list.append("//")
		for i in range(0,2):
			if isclose(x - y,ans):
				counter += 1
				operand_list.append("-")
			if y != 0:
				if isclose(float(x) / float(y), float(ans)):
					counter += 1
					operand_list.append("/")
			x,y = y,x
	if counter == 1:
		return True, operand_list[0]
	else:
		return False, None

def num(s):
	try:
		return int(s)
	except ValueError:
		try:
			return float(s)
		except ValueError:
			return None


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def set_filter(SAVE_DIR, DATA_DIR, prefix):
	for i in range(1, 6+1):
		j = 1
		while True:
			filename = prefix + "G{}_{:02d}.xml.csv.filtered".format(i,j)
			if os.path.isfile(os.path.join(DATA_DIR, filename)):
				file = readfile(os.path.join(DATA_DIR,filename))
				filter(SAVE_DIR ,file, filename)
			else:
				break
			j += 1

def main():
	FILEPREFIX = "MR_MATH_Train_"
	FILEPREFIX2 = "MR_MATH_Develop_"
	FILEPREFIX3 = "MR_MATH_Test_"
	# Parse argument
	parser = argparse.ArgumentParser(description="Filter out 2 number in Body and Question and add label(+,-,*,/,//,%)")
	parser.add_argument('--data_dir', type=str, default="/home/jimlin7777/MWP_Data", 
		help="directory where your data locate")
	parser.add_argument('--save_dir', type=str, default="/home/jimlin7777/MWP_Data", 
		help="directory where your parsed data want to save")
	args = parser.parse_args()
	SAVE_DIR = args.save_dir
	DATA_DIR = args.data_dir
	# Filter train, dev, test data set
	set_filter(SAVE_DIR, DATA_DIR, FILEPREFIX)
	set_filter(SAVE_DIR, DATA_DIR, FILEPREFIX2)
	set_filter(SAVE_DIR, DATA_DIR, FILEPREFIX3)

if __name__ == '__main__':
	main()