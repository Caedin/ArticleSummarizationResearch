import sys
from pyteaser import Summarize as BitOfNews
from HTMLParser import HTMLParser

# create a subclass and override the handler methods
class MyHTMLParser(HTMLParser):
	data_list = []
	header = False
	txt = False
	head = []
	
	def handle_starttag(self, tag, attrs):
		if tag == 'head':
			self.header = True
		if tag == 'text':
			self.txt = True
	def handle_endtag(self, tag):
		if tag == 'head':
			self.header = False
		if tag == 'text':
			self.txt = False
	def handle_data(self, data):
		if self.header == True:
			self.head.append(str(data))
		elif self.txt == True:
			self.data_list.append(str(data))
	def clear_data(self):
		self.data_list = []
		self.header = False
		self.txt = False
		self.head = []
		
	def get_head(self): return self.head
		
parser = MyHTMLParser()

def get_input_text(input_file, parser):
	with open(input_file, 'rb') as input:
		text_local = ''.join([x for x in input])
		
	text_local = text_local.replace('\n', ' ')
	parser.feed(text_local)
	
	text_local = ''.join([x for x in parser.data_list])
	head_local = ''.join([x for x in parser.head])

	return text_local, head_local
	
def write_summary(text, output_file, head):
	if len(head) == 0:
		head = text[0:75]
	summaries = BitOfNews(head, text)
	if output_file == None:
		for k in summaries: print k
	else:
		with open(output_file, 'wb') as output:
			for k in summaries:
				output.write(str(k)+'\n')

def Summarize(input_file, output_file):
	parser.clear_data()
	
	if 'http' in input_file:
		text = get_url_text(input_file)
	else:
		text, head = get_input_text(input_file, parser)
	text = unicode(text, 'utf-8')	

	write_summary(text, output_file[:-2], head)
	
if __name__ == '__main__':
	input_file = sys.argv[1]
	head, text = get_input_text(input_file)
	write_summary(head, text)