class classification_performance_evaluator:
	def __init__(self, names_categories):
		self.names_categories = names_categories
		self.num_categories = len(names_categories)
		print self.num_categories
		self.classification_2dlst = [[0] * self.num_categories for i in range(self.num_categories)]
		
	def update(self, labels, predictions):
		#print labels
		#print predictions
		for i in xrange(len(labels)):
			self.classification_2dlst[labels[i]][predictions[i]] += 1


	def print_performance(self):
		for i in xrange(self.num_categories):
			total = sum(self.classification_2dlst[i])
			print total
			print self.names_categories[i]
			print '****************************************************************'
			for j in xrange(self.num_categories):
				print 'Label: '+self.names_categories[i]+'\t'+'Classified As: '+self.names_categories[j]+'\t'+str(float(self.classification_2dlst[i][j])/total)

	def _label_to_category(self, label):
		return self.names_categories(label)


