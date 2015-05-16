import sys
import math
import string
from textblob import TextBlob
from textblob import Word
from textblob.wordnet import Synset
from textblob_aptagger import PerceptronTagger
from nltk.stem.snowball import SnowballStemmer
import nltk.data
import urllib
import random
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
damping_factor = 0.85
co_occurence_window = 2

stop_list = ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is' ,'it', 'its', 'of', 'on', 'that', 'the', 'to', 'were', 'was', 'will' ,'with']

''' 
Author: Keith Dyer
Date: 12/18/2014

This is a modified version of TextRank for sentence extraction.

First use TextRank to extract key words and their weights.
Define sentence similarity function as sum of the weights of overlapping key words / total words.
Hypothesis: This will allow more relevant extraction, because the similarity function between sentences will be weighted based on how important the overlapping words are to the article.
'''

class node:
	text = ''
	rank = 1
	links = []
	def __init__(self, text):
		self.text = text
		self.links = []
		
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
	
def build_word_graph(text):
	exclude = set(string.punctuation)
	total_word_array = ''.join([x for x in text if x not in exclude])
	total_word_array = total_word_array.split()
	
	words = TextBlob(text, pos_tagger=ap_tagger)
	word_list = set()
	
	for k in words.tags:
		if 'NN' in k[1] or 'JJ' in k[1]:
			word_list.add(k[0].lower())
	
	verticies = []
	word_locations = {}
	
	for k in word_list:
		temp = node(k)
		verticies.append(temp)
		word_locations[k] = (temp,[])
		for x in xrange(len(total_word_array)):
			if k==(total_word_array[x].lower()):
				word_locations[k][1].append(x)
	
	for k in verticies:
		word = k.text
		range = []
		for x in word_locations[word][1]:
			for y in xrange(-2,3,1):
				range.append(x+y)
		for x in range:
			for key in word_locations:
				if x in word_locations[key][1] and k!=word_locations[key][0]:
					k.links.append(word_locations[key][0])
	return verticies
	
def build_graph(text, parser):
	sentences = [x for x in parser.data_list if x in text]
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
	
def calculate_edges(graph, word_graph):
	edge_matrix = [[None for _ in range(len(graph))] for _ in range(len(graph))]
	
	word_matrix = {}
	for k in word_graph:
		word_matrix[k.text] = k
		
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
			for x in k_words:
				if x in j_words and x not in stop_list:
					word_cross_over = x.lower()
					if word_cross_over in word_matrix:
						similarity += word_matrix[word_cross_over].rank
			
			try:
				edge_matrix[k_counter][j_counter] = similarity
			except ZeroDivisionError:
				edge_matrix[k_counter][j_counter] = 0
			j_counter+=1
		k_counter+=1
	return edge_matrix
	
def TextRank_words(graph):
	for vertex in graph:
		score = 0
		for link in vertex.links:
			try:
				score+= link.rank / (len(link.links))
			except ZeroDivisionError:
				print '\n\n', 'ERROR', vertex.text, link.text, ' '.join([x.text for x in link.links])
		vertex.rank = 1-damping_factor + damping_factor * score

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
		
def Converge_Word_Graph(graph):
	difference = 1
	while(difference>0.000001):
		rank_vector = []
		for x in graph:
			rank_vector.append(x.rank)
		
		TextRank_words(graph)
		max_diff = 0
		for x in xrange(len(graph)):
			if math.fabs(rank_vector[x] - graph[x].rank) > max_diff:
				max_diff = math.fabs(rank_vector[x] - graph[x].rank)
		difference = max_diff
	
	graph.sort(key = lambda x: x.rank)
	graph.reverse()
		
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

def greedy_search_word_removal(graph, word_graph):
	word_table = {}
	for k in word_graph:
		word_table[k.text] = k
	
	for sentence in graph:
		score = 0
		words = sentence.text.split()
		for word in set(words):
			if word.lower().strip() in word_table:
				score+=word_table[word.lower().strip()].rank
		sentence.rank = score
	
	final_five = []
	for k in xrange(5):
		# Sort graph to get top sentence
		graph.sort(key = lambda x: x.rank)
		graph.reverse()
		
		final_five.append(graph[0])
		
		# Remove terms from top sentence in word_graph
		words = graph[0].text.split()
		for word in set(words):
			if word.lower().strip() in word_table:
				del word_table[word.lower().strip()]
		
		# Re-score all sentences with new word_graph
		for sentence in graph:
			# move already selected elements to the bottom
			if sentence in final_five: 
				sentence.rank = 0
				continue
			
			score = 0
			words = sentence.text.split()
			for word in set(words):
				if word.lower().strip() in word_table:
					score+=word_table[word.lower().strip()].rank
			sentence.rank = score
	
	# return updated graph
	graph = final_five
	return graph
	
def greedy_search_sentence_removal(text):
	final_five = []
	
	for x in xrange(5):
		word_graph = build_word_graph(text)
		Converge_Word_Graph(word_graph)
		graph = build_graph(text, parser)
		edge_matrix = calculate_edges(graph, word_graph)
		Converge_Graph(graph, edge_matrix)
		
		graph.sort(key = lambda x: x.rank)
		graph.reverse()
		final_five.append(graph[0])
		text = text.replace(graph[0].text, '')
	
	return final_five
	
	
			
	
def Summarize(input_file, output_file):
	parser.clear_data()
	
	if 'http' in input_file:
		text = get_url_text(input_file)
	else:
		text = get_input_text(input_file, parser)
	text = unicode(text, 'utf-8')
	
	word_graph = build_word_graph(text)
	Converge_Word_Graph(word_graph)
	graph = build_graph(text, parser)
	
	#graph = greedy_search_word_removal(graph, word_graph)
		
	edge_matrix = calculate_edges(graph, word_graph)
	Converge_Graph(graph, edge_matrix)
	
	write_summary(text, graph, output_file[:-2])
		
if __name__ == '__main__':
	input_file = sys.argv[1]
	output_file = None
	if 'http' in input_file:
		text = get_url_text(input_file)
	else:
		text = get_input_text(input_file)
	text = unicode(text, 'utf-8')	
	
	word_graph = build_word_graph(text)
	Converge_Word_Graph(word_graph)
	
	graph = build_graph(text)
	edge_matrix = calculate_edges(graph, word_graph)
	Converge_Graph(graph, edge_matrix)
	write_summary(text, graph, output_file)

	