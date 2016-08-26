#!/usr/bin/env python2.7

import re
import matplotlib.pyplot as plt


def loss_graph():
	regex = re.compile(r"(loss\s+\=\s)(\d+\.\d+)")
	step_lst = []
	step = 0
	loss_lst = []
	with open('output.txt') as f:
		for line in f:
			line = line.rstrip('\n')
			r = re.search(regex, line)
			if r is not None:
				loss_lst.append(r.group(2))
				step_lst.append(step)
				step+=1
	plt.plot(step_lst, loss_lst, 'go-', label = 'Training Loss')
	plt.legend(loc='center', shadow=True)
	plt.xlabel('Step#')
	plt.ylabel('Loss')
	plt.title('Loss')
	plt.show()

def test_train_accuracy_graph():
	regex1 = re.compile(r"(training accuracy\s+\=\s)(\d+\.\d+)")
	regex2 = re.compile(r"(validation accuracy\s+\=\s)(\d+\.\d+)")
	step_lst = []
	test_accuracy_lst = []
	train_accuracy_lst = []
	step = 0
	with open('output.txt') as f:
		for line in f:
			line = line.rstrip('\n')
			r = re.search(regex1, line)
			if r is not None:
				print r.group(2)
				train_accuracy_lst.append(r.group(2))

			r = re.search(regex2, line)
			if r is not None:
				print r.group(2)
				test_accuracy_lst.append(r.group(2))
				step_lst.append(step)
				step += 10
	plt.plot(step_lst, test_accuracy_lst, 'rx-', label = 'Validation Accuracy')
	plt.plot(step_lst, train_accuracy_lst, 'b^-', label = 'Training Accuracy')
	plt.legend(loc = 'lower left')
	plt.xlabel('Step#')
	plt.ylabel('Accuracy')
	plt.title('Accuracy')
	plt.show()

#loss_graph()
test_train_accuracy_graph()