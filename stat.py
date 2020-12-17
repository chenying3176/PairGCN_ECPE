from collections import defaultdict

def load_data(inp_file):
	data = {}
	with open(inp_file, 'r', encoding='utf-8') as f:
		lines = list(f.readlines())
		line_nums = len(lines)
		index = 0
		while index < line_nums:
			pair_index, pair_line_nums = [int(z) for z in lines[index].strip().split()]
			index += 1
			pair_label = lines[index].strip()
			text = []
			for i in range(pair_line_nums):
				line = lines[index+i+1].strip().split(",")
				text.append(line[-1])
			assert (len(text) == pair_line_nums)
			index += pair_line_nums + 1
			data[pair_index] = (text, pair_label)
	return data


def multi_label(data):
	emo_acc = 0 
	emo_num = 0
	cause_acc = 0 
	cause_num = 0

	for key in data:
		label = data[key][1]
		emo_cause_pair = {}
		print('labels ')
		for l in label.split(', '):
			l = l.strip().replace(' ', '').replace('(', '').replace(')', '').split(',')
			if int(l[0]) in emo_cause_pair:
				emo_cause_pair[int(l[0])].append(int(l[1]))
			else:
				emo_cause_pair[int(l[0])] = [int(l[1]),]
		for k in emo_cause_pair:
			if len(emo_cause_pair[k]) > 1:
				emo_acc += 1
				cause_acc +=  len(emo_cause_pair[k])
			emo_num += 1
			cause_num +=  len(emo_cause_pair[k])
		print(emo_cause_pair)
	 
	print('multi_label emo: %d, %d, %0.3f' % (emo_acc, emo_num, emo_acc/emo_num)) 
	print('multi_label cause:%d, %d, %0.3f' % (cause_acc, cause_num, cause_acc/cause_num)) 

if __name__ == '__main__':
	data = load_data('data/splited_data/all_data_pair.txt')

	type_ = defaultdict(int)
	for key in data:
		label = data[key][1]
		for l in label.split(', '):
			l = l.strip().replace(' ', '').replace('(', '').replace(')', '').split(',')
			dis = abs(int(l[0])-int(l[1]))
			type_[dis] += 1
	i = [0, 1, 2]
	num = sum([type_[key] for key in type_])
	for ii in i:
		print('num %d, %d, %0.3f' % (ii, type_[ii], type_[ii]/num))
		acc = 0
		for jj in i:
			if jj <= ii:
				acc += type_[jj]
		print('num <=%d, %d, %0.3f' % (ii, acc, acc/num))
	print(num)
	multi_label(data)
