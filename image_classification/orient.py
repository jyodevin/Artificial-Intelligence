#Bharat Mallala, Jyothi Pranavi, Harshit Krishnakumar
from __future__ import division
import sys
from math import sqrt,log,exp
from random import choice
from copy import deepcopy
import numpy as np
import json
from numpy import *
import csv

#KNN
#****************************************************************************************
def file_open1(file_name):
	f = open(file1)
	data = []
	for line in f:
    		data_line = line.rstrip().split(' ')
    		data.append(data_line)
	f.close()
	return data

#KNN
def knn(model_file,test_data):
	labels=[]
	train_data=file_open1(model_file)
	k=18
	for test_image in test_data:
		distance={}
		for train_image in train_data:
			distance[train_image[0]+':'+train_image[1]]=sqrt(sum([((int(train_image[i])-int(test_image[i]))**2) for i in range(2,len(train_image))]))	      
		inter=sorted(distance,key=distance.get)[0:k+1]
		nn=[]
		for key in inter:
			nn.append(key[(key.index(':')+1):])
		count1=0
		for i in nn:
			count=nn.count(i)
			if count>count1:
				count1=count
				label = i
		labels.append(label)
	return labels

#accuracy
def accuracy1(predicted,test):
	accu_score=0
	for i,j in zip(predicted,test):
		#print(i)
		#print(j[1])
		if i==j[1]:
			accu_score+=1
	total_score=(accu_score/len(test))*100
	return total_score


#Adaboost
#*****************************************************************************************
class Adaboost():
	def __init__(self,data,model_file):
		self.data=data
		self.file = model_file
		self.class_pairs=[(0,90),(0,180),(0,270),(90,180),(90,270),(180,270)]
	def hyp(self,split_data,stump):
		h1={}
		h2={}
		for image in split_data:
			if image[stump[0]]>image[stump[1]]:
				h1[str(image[0])]=h1.get(str(image[0]),0)+1
			else:
				h2[str(image[0])]=h2.get(str(image[0]),0)+1
		if len(h1)==0:
			return list(h2.keys())
		elif len(h2)==0:
			return list(h1.keys())
		else:
			return [max(h1,key=h1.get),max(h2,key=h2.get)]
	def train(self,split_data):
		stumps=[]
		self.n = len(split_data)
		self.weights=[1/self.n]*self.n
		num=np.linspace(1,192,192)
		for i in range(0,100):
			stumps.append([int(choice(num)),int(choice(num))])
		
		#hypotheses
		
		ht=[]
		
		alpha=[]
		for t in range(0,100):
			labels=[]
			pairs=stumps[t]
			h = self.hyp(split_data,pairs)
			ht.append(h)
			#print(h.values())
			for i in range(self.n):
				if split_data[i][pairs[0]]>split_data[i][pairs[1]]:
					labels.append(h[0])
				else:
					labels.append(h[1])
			error=0
			for i in range(self.n):
				if int(labels[i])!=int(split_data[i][0]):
					error+=self.weights[i]
			for i in range(self.n):
				if int(labels[i])==int(split_data[i][0]):
					self.weights[i]*=(error/(1-error))
			#print(error)
			self.weights=[(w/sum(self.weights)) for w in self.weights]
			alpha.append(log((1-error)/(error)))
		return ht,alpha,stumps
	def adab(self):
		split_data={}
		for image in self.data:
			for item in self.class_pairs:
				if image[0] in item:
					split_data[str(item)]=split_data.get(str(item),tuple())+(image,)	
		hypo={}
		z={}
		stumps={}
		for pair in self.class_pairs:
			h,alpha,stump=self.train(split_data[str(pair)])
			hypo[str(pair)] = h
			z[str(pair)]=alpha
			stumps[str(pair)]=stump
		return hypo,z,stumps
	def test(self):
		with open(self.file, 'r') as param_file:
			hypo,alpha,stumps = json.load(param_file)
		#print('hypothesis',hypo.values())
		#pairs=[line[0] for line in parameters]
		classes=[]
		for image in self.data:
			results=[]
			for pair in self.class_pairs:
				labels={}
				for i in range(0,100):
					if image[stumps[str(pair)][i][0]] >= image[stumps[str(pair)][i][1]]:
						labels[hypo[str(pair)][i][0]] = labels.get(hypo[str(pair)][i][0],0)+alpha[str(pair)][i]
					else:
						labels[hypo[str(pair)][i][1]] = labels.get(hypo[str(pair)][i][1],0)+alpha[str(pair)][i]
						
				results.append(max(labels,key=labels.get))
				
			count1=0
			for result in results:
				count2=results.count(result)
				if count2>count1:
					count1=count2
					label=result
			classes.append(label)
		return classes
#-------------------------------------------------------------------------------------------------#

# Neural Net
#--------------------------------------------------------------------------------------------------#
def sigmoid(input):
	return array([[1/(1+exp(-x)) for x in dummy] for dummy in input])

def rmse(input, target):
	return sqrt(mean((input - target)**2))



def neural_network(data, train_test, model_file):
	# Splitting input data into training and labels
	if train_test == 'train':
		train_x = [array([int(pixel) for pixel in row[2:]]) for row in data] 
	
		# Normalizing to avoid math overflow
		temp = []
		for row in train_x:
			max_x = max(row)
			min_x = min(row)
			temp.append(concatenate((array([(2*(x-min_x)/(max_x-min_x))-1 for x in row]) , array([1]) ), axis = 0 ))
		train_x = temp
				
		train_y = []
		for row in data:
			if row[1] == '0':
				train_y.append(array([1.0,0.0,0.0,0.0]))
			elif row[1] == '90':
				train_y.append(array([0.0,1.0,0.0,0.0]))
			elif row[1] == '180':
				train_y.append(array([0.0,0.0,1.0,0.0]))
			elif row[1] == '270':
				train_y.append(array([0.0,0.0,0.0,1.0]))
			else:
				train_y.append(array([0.0,0.0,0.0,0.0]))
				print('error', row[1])
	
		# Initializing the Neural Network Weights
		
		n_inputs = shape(train_x)[1] # Including bias
		
		n_outputs = shape(train_y)[1]
		
		n_hidden_1 = 20 # Trying a random value
		
		eta = 0.05
		
		hidden_1_weights = array([random.normal(0,1/sqrt(n_inputs*n_outputs),n_inputs).tolist() for _ in range(n_hidden_1)])
		output_weights = array([random.normal(0,1/sqrt(n_hidden_1),n_hidden_1+1).tolist() for _ in range(n_outputs)]) # Extra 1 comes from the bias term
		
		#taking 10 epochs :
		
		for epoch in range(20):
			# Iterating through every row, for stochastic gradient descent:
			for i in range(len(train_x)):
				
				# Forward Propogation of input
				input = [train_x[i]]
				label = [train_y[i]]
		
				hidden_1_weights = around(hidden_1_weights, 10 )
				
				
				# input = [[round(x,2) for x in row]for row in input]
				
				hidden_1_outputs = sigmoid(dot(input, transpose(hidden_1_weights)))
				hidden_1_outputs = array([[round(x, 10) for x in row]+[1.00]  for row in hidden_1_outputs])
				
				output_outputs = sigmoid(dot(hidden_1_outputs, transpose(output_weights)))
				
				# error = rmse(output_outputs, label)
				
				
				# Back Propogation of Error
				delta_output = multiply(multiply((output_outputs - label), output_outputs), (1-output_outputs))
				delta_weights_2 = dot(transpose(delta_output) , hidden_1_outputs)
				
				hidden_1_outputs = array([ row[:-1]  for row in hidden_1_outputs]) # Removing bias term for this calculation
				
				delta_hidden = multiply(multiply(hidden_1_outputs.T, (1-hidden_1_outputs.T)), dot( transpose([row[:-1] for row in output_weights]), transpose(delta_output) ))
				
				delta_weights_1 = dot(delta_hidden, input)
				
				# Updating weights:
				hidden_1_weights = hidden_1_weights - (eta*delta_weights_1)
				output_weights = output_weights - (eta*delta_weights_2)
		# print(array([hidden_1_weights.tolist(), output_weights.tolist()]))
		save(model_file, array([hidden_1_weights.tolist(), output_weights.tolist()]))
	
	else:
		
		test_x = [array([int(pixel) for pixel in row[2:]]) for row in data] 
		test_labels = [row[0] for row in data] 
		
		model_params = load(model_file+'.npy')
		hidden_1_weights = model_params[0]
		output_weights = model_params[1]
	
		# Normalizing to avoid math overflow
		temp = []
		for row in test_x:
			max_x = max(row)
			min_x = min(row)
			temp.append(concatenate((array([(2*(x-min_x)/(max_x-min_x))-1 for x in row]) , array([1]) ), axis = 0 ))
		test_x = temp
		# Testing Data Predictions:
		predicted=[]
		for i in range(len(test_x)):
			row = [test_x[i]]
			hidden_1_weights = around(hidden_1_weights, 10 )		
			
			hidden_1_outputs = sigmoid(dot(row, transpose(hidden_1_weights)))
			hidden_1_outputs = array([[round(x, 10) for x in row]+[1.00]  for row in hidden_1_outputs])
			
			output_outputs = sigmoid(dot(hidden_1_outputs, transpose(output_weights)))
			decision = argmax(output_outputs)
			
			if decision == 0:
				test_labels[i]= test_labels[i] + ' 0'
				predicted.append('0')
			elif decision == 1:
				test_labels[i]= test_labels[i] + ' 90'
				predicted.append('90')
			elif decision == 2:
				test_labels[i]= test_labels[i] + ' 180'
				predicted.append('180')
			elif decision == 3:
				test_labels[i]= test_labels[i] + ' 270'
				predicted.append('270')
		#print(test_labels)
		naccuracy = accuracy1(predicted,data)
		print(naccuracy)
		
		with open("output.txt", "w") as output:
			for item in test_labels:
				output.write("%s\n" % item)






#opening a file
def file_open(file_name):
#reading a file
	f = open(file_name,'r')
	lines=f.readlines()
	#print('lines',len(lines))
	data_line=[line.strip().split(' ')[1:] for line in lines]
	data=[map(int,x) for x in data_line]
	f.close()
	return data

#accuracy score
def accuracy(predicted,test):
	accu_score=0
	for i,j in zip(predicted,test):
		#print('predicted',i)
		#print('test',j[0])
		if int(i)==int(j[0]):
			accu_score+=1
	total_score=(accu_score/len(test))*100
	return total_score

#main()	
train_test,file1,model_file,model = (sys.argv[1:5])
if train_test == 'train':
	train_data=file_open(file1)
	print('train',len(train_data))
	if model=='[nearest]':
		param_file=open(model_file,'w')
		files=open(file1,'r')
		lines=files.readlines()
		param_file.writelines(lines)
		param_file.close()
	if model == '[adaboost]':
		adaboost=Adaboost(train_data,model_file)
		hypothesis,alpha,stumps=adaboost.adab()
		with open(model_file,'w') as file:
			json.dump([hypothesis,alpha,stumps],file)
		file.close()
	if model == '[nnet]':
		data = []
		with open(file1) as text_file:
			for line in csv.reader(text_file,delimiter = " "):
				data.append(line)
		neural_network(data, train_test, model_file)
		
		text_file.close()
	if model == '[best]':
		data = []
		with open(file1) as text_file:
			for line in csv.reader(text_file,delimiter = " "):
				data.append(line)
		neural_network(data, train_test, model_file)
		
		text_file.close()

	
if train_test == 'test':
	test_data1=file_open1(file1)
	print('test',len(test_data1))
	#output_file=open("output.txt",'w')
	output_file=open("output.txt",'w')
	if model=='[nearest]':
    		results=knn(model_file,test_data1)
			#print(results[0:10])
		for i,j in zip(test_data1,results):
			output_file.write(' '.join([i[0],j])+"\n")
		accuracy_score=accuracy1(results,test_data1)
		output_file.close()
		print(accuracy_score)
	if model=='[adaboost]':
		test_data = file_open(file1)
		adaboost=Adaboost(test_data,model_file)
		results = adaboost.test()
		accuracy_score=accuracy(results,test_data)
		print('accuracy',accuracy_score)
		for i,j in zip(test_data1,results):
			output_file.write(' '.join([i[0],j])+"\n")
		output_file.close()
	if model == '[nnet]':
		data = []
		with open(file1) as text_file:
			for line in csv.reader(text_file,delimiter = " "):
				data.append(line)
		neural_network(data, train_test, model_file)
	
	if model == '[best]':
		print("Running nnet algorithm")
		data = []
		with open(file1) as text_file:
			for line in csv.reader(text_file,delimiter = " "):
				data.append(line)
		neural_network(data, train_test, model_file)



