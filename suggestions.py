import os

'''
作者：费德里科·泰尔齐
此库包含操作单词和单词所需的类
征求意见这是一项正在进行的工作。。。。
'''

class Hinter:
	'''
	Hinter用于加载字典并获取一些建议
	关于下一个可能的字母或兼容的单词
	'''
	def __init__(self, words):
		self.words = words

	@staticmethod
	def load_english_dict():
		'''
		加载英语词典并返回带有
		加载到self.words列表中的单词
		'''
		ENGLISH_FILENAME = "dict" + os.sep + "english.txt"
		words = [i.replace("\n","") for i in open(ENGLISH_FILENAME)]
		return Hinter(words)

	def compatible_words(self, word, limit = 10):
		'''
		返回以“word”参数开头的单词。
		“limit”参数定义函数的字数
		返回最大值
		'''
		output = []
		word_count = 0
		#循环浏览所有单词，找出以“word”开头的单词
		for i in self.words:
			if i.startswith(word):
				output.append(i)
				word_count+=1
			if word_count>=limit: #如果达到限制，则退出
				break
		return output

	def next_letters(self, word):
		'''
		返回兼容字母的列表。
		当有以“word”开头的单词时，字母是兼容的,然后是字母。
		'''
		#获得100个兼容的单词
		words = self.compatible_words(word, 100)
		letters = []
		#循环浏览所有兼容的单词
		for i in words:
			if len(i)>len(word): #如果“单词”比兼容单词长，请跳过
				letter = i[len(word):len(word)+1] #收到下面的信
				if not letter in letters: #避免重复
					letters.append(letter)
		return letters

	def does_word_exists(self, word):
		'''
		检查加载的词典中是否存在特定单词
		'''
		if word in self.words:
			return True
		else:
			return False

	def most_probable_letter(self, clf, classes, linearized_sample, word):
		'''
		获取给定记录符号和当前单词的最可能字母
		'''
		if word=="":
			return None
		
		probabilities = clf.predict_log_proba(linearized_sample)
		ordered = sorted(classes)
		values = {}
		for i in range(len(probabilities[0])):
			values[round(probabilities[0,i], 5)] = classes[ordered[i]]
		ordered = sorted(values, reverse=True)
		possible_letters = self.next_letters(word)
		for i in ordered:
			if values[i] in possible_letters:
				return values[i]
		return None

