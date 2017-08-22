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

	output.to_csv(os.path.join(SAVE_DIR ,filename + ".operands_threenum"), index = False)

def bruteforce_find_operand(ans_number_list, body_number_list, question_number_list):
	length = len(body_number_list) + len(question_number_list)
	assert len(ans_number_list) == 1
	assert (length) >= 2
	assert (length) <= 3
	operators = ["+", "-", "*", "/"]
	match_list = []
	number_list = map(num, body_number_list + question_number_list)
	ans_number_list = map(num, ans_number_list)
	for c in combinations_with_replacement(operators, length - 1):
		if operator_match(c, ans_number_list, number_list) == True:
			match_list.append(''.join(c))
	if len(match_list) == 1:
		return True, match_list[0]
	return False, None

def all_perms(elements):
    if len(elements) <=1:
        yield elements
    else:
    	# divide and conquer
        for perm in all_perms(elements[1:]):
            for i in range(len(elements)):
                # nb elements[0:1] works in both string and list contexts
                yield perm[:i] + elements[0:1] + perm[i:]

def combinations_with_replacement(iterable, r):
	pool = tuple(iterable)
	n = len(pool)
	indices = [0] * r
	yield tuple(pool[i] for i in indices)
	while True:
		isdone = True
		# if all indices is "n-1", then done.
		for i in reversed(range(r)):
			if indices[i] != n - 1:
				isdone = False
				break
		if isdone:
			return
		indices[i:] = [indices[i] + 1] * (r - i)
		yield tuple(pool[i] for i in indices)
def combination(iterable, r):
	pool = tuple(iterable)
	n = len(pool)
	if r > n:
		return
	indices = [i for i in range(r)]
	yield tuple(pool[i] for i in indices)
	while True:
		for i in reversed(range(r)):
			if indices[i] != n + i - r:
				break
		else:
			return
		indices[i] += 1
		for j in range(i+1, r):
			indices[j] = indices[j-1] + 1
		yield tuple(pool[i] for i in indices)

def operator_match(operators, ans_number_list, number_list):
	n = len(number_list)
	result_list = []
	recursive_merge_number_list(operators, number_list, result_list)
	for i in range(len(result_list)):
		if ans_number_list[0] is not None:
			if isclose(result_list[i], ans_number_list[0]):
				return True
	return False

def recursive_merge_number_list(operators, number_list, result_list):
	if len(number_list) == 1:
		result_list.append(number_list[0])
	else:
		for c in combination(number_list, 2):
			for p in all_perms(c):
				for i, op in enumerate(operators):
					temp_sum = None
					if op == "+":
						temp_sum = p[0] + p[1]
					elif op == "-":
						temp_sum = p[0] - p[1]
					elif op == "*":
						temp_sum = p[0] * p[1]
					elif op == "/" and p[1] != 0:
						temp_sum = p[0] / p[1]
					if temp_sum == None:
						continue
					temp_list = number_list[:]
					temp_list.remove(p[0])
					temp_list.remove(p[1])
					temp_list.append(temp_sum)
					recursive_merge_number_list(operators[:i] + operators[i+1:], temp_list, result_list)
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
			filename = prefix + "G{}_{:02d}.xml.csv.filtered_three".format(i,j)
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
	parser = argparse.ArgumentParser(description="Filter out 2 number in Body and Question and add label(+,-,*,/,%)")
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