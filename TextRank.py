import sys
import math
import string
from textblob import TextBlob
from textblob_aptagger import PerceptronTagger
from nltk.stem.snowball import SnowballStemmer
import nltk.data
import urllib
from HTMLParser import HTMLParser

# create a subclass and override the handler methods
class MyHTMLParser(HTMLParser):
	data_list = []
	def handle_starttag(self, tag, attrs):
		pass
	def handle_endtag(self, tag):
		pass
	def handle_data(self, data):
		self.data_list.append(str(data))
	def clear_data(self):
		self.data_list = []
		
parser = MyHTMLParser()
		
ap_tagger = PerceptronTagger()
snow_stemmer = SnowballStemmer("english")
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
damping_factor = 0.85
co_occurence_window = 2

stop_list = ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is' ,'it', 'its', 'of', 'on', 'that', 'the', 'to', 'were', 'was', 'will' ,'with']

''' 
Author: Keith Dyer
Date: 12/18/2014

This is a basic TextRank program based on the paper by Rada Mihalcea and Paul Tarau at University of North Texas 2004.
'''

class node:
	text = ''
	rank = 1
	links = []
	def __init__(self, text):
		self.text = text
	
def get_url_text(input_file):
	# Function for fetching text from a website. Todo.
	return None
	
def get_input_text(input_file, parser):
	with open(input_file, 'rb') as input:
		text_local = ''.join([x for x in input])
		
	text_local = text_local.replace('\n', ' ')
	parser.feed(text_local)
	
	text_local = ''.join([x for x in parser.data_list])
	return text_local
	
def build_graph(text, parser):
	sentences = parser.data_list
	verticies = []
	for k in sentences:
		sentence = str(k).strip()
		sentence = sentence.replace('\n', '')
		sentence = sentence.replace('\t', '')
		if len(sentence) > 1:
			temp = node(str(sentence))
			verticies.append(temp)
	
	for k in verticies:
		k.links = verticies
	
	return verticies
	
def calculate_edges(graph):
	edge_matrix = [[None for _ in range(len(graph))] for _ in range(len(graph))]
	k_counter = 0
	for k in graph:
		j_counter = 0
		k_words = k.text.split()
		for j in k.links:
			if j == k:
				edge_matrix[k_counter][j_counter] = 0
				j_counter+=1
				continue
			j_words = j.text.split()
			similarity = 0
			tmp = []
			for x in k_words:
				if x in j_words and x not in stop_list:
					similarity+=1
					tmp.append(x)
			try:
				edge_matrix[k_counter][j_counter] = similarity / (math.log(len(k_words)) + math.log(len(j_words)))
			except ZeroDivisionError:
				edge_matrix[k_counter][j_counter] = 0
			j_counter+=1
		k_counter+=1
	return edge_matrix
		
def TextRank(graph, edge_matrix):
	global_count = 0
	for vertex in graph:
		new_rank = (1-damping_factor)
		global_sum = 0
		local_count = 0
		for x in edge_matrix[global_count]:
			if x==0:
				local_count+=1
				continue
			if global_count == local_count: 
				local_count+=1
				continue
			local_sum = sum(edge_matrix[local_count])
			global_sum+=(x/local_sum)*graph[local_count].rank
			local_count+=1
		vertex.rank = new_rank + damping_factor * global_sum
		global_count+=1

def Converge_Graph(graph, edge_matrix):
	difference = 1
	while(difference>0.000001):
		rank_vector = []
		for x in graph:
			rank_vector.append(x.rank)
		
		TextRank(graph, edge_matrix)
		max_diff = 0
		for x in xrange(len(graph)):
			if math.fabs(rank_vector[x] - graph[x].rank) > max_diff:
				max_diff = math.fabs(rank_vector[x] - graph[x].rank)
		difference = max_diff
		
def write_summary(text, graph, output_file, threshold=-1):
	if threshold == -1:
		threshold = max(len(graph)/10,4)
		threshold = min(threshold, len(graph))
	graph.sort(key = lambda x: x.rank)
	graph.reverse()
	summary = []
	
	for k in xrange(5): 
		position = text.find(graph[k].text)
		summary.append((graph[k].text, position))
	
	summary.sort(key = lambda x: x[1])
	if output_file == None:
		for k in summary: print k[0]
	else:
		with open(output_file, 'wb') as output:
			for k in summary:
				output.write(str(k[0])+'\n')

def Summarize(input_file, output_file):
	parser.clear_data()
	
	if 'http' in input_file:
		text = get_url_text(input_file)
	else:
		text = get_input_text(input_file, parser)
	text = unicode(text, 'utf-8')	

	graph = build_graph(text, parser)
	edge_matrix = calculate_edges(graph)
	Converge_Graph(graph, edge_matrix)
	write_summary(text, graph, output_file[:-2])
	
if __name__ == '__main__':
	input_file = sys.argv[1]
	
	text = get_input_text(input_file)
	graph = build_graph(text)
	edge_matrix = calculate_edges(graph)
	Converge_Graph(graph, edge_matrix)
	write_summary(text, graph, 5)

	