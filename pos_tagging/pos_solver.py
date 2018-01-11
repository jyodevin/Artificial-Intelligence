###################################
# CS B551 Fall 2017, Assignment #3
#
# Bharat Mallala(bmallala), Jyothi Pranavi Devineni(jyodevin), Harshit Krishnakuamr(harkrish)
#
# (Based on skeleton code by D. Crandall)
#
#
####
# Put your report here!!
# -----------------Training Part--------------------------
#We divided the training part into three functions namely
# 1. ini_prob: to calculate the initial probabilities
# 2. tran_prob: to calculate the transition probabilities
# 3. emi_prob: to calculate the emission probabilities
# these functions will be called in the training function.
#----------------------------------------------------------
#--------------Simplified-------------------------------------
# As the states S1, S2,S3,....... are independent of each other, each POS is only dependent on its corresponding observed word.
# Hence we calculated the posterior probabilities as the maximum of the product of initial and emission probabilities of 12 respective states.
# We have excluded the total probabilities since it remains constant across all the posterior probabilities.
#----------------------------------------------------------------
#----------------------Variable elimination----------------------
# We have used forward-backward algorithm.
# have calculated both the apha part and beta part and combined the results from both 
# Here each POS is not only dependent on its respective observed word but also on its previous state.
# For the initial state we estimated the posterior to be the product of initial probabilities and the respective emission probabilities alone since it does not have a previous state i.e. no transition probabilities.
# For all the remaining states we calculated the posterior probability using the emission probability of the current state, transition probabilities from the previous state to current state, posterior of the previous state.
# we then store the posteriors of each state and loop forward.
# we then chose the maximum of the 12 posteriors obtained for every word in each iteration.
# If the emission probabilities are missing or are zero we replaced it with a very small value.
#----------------------------------------------------------------------------------------------
#-------------------------------------Viterbi Decoding-----------------------------------------------
#For the initial_state, we calculated the vi(t) as the product of the initial and emission probabilities according to the algorithm
#For all the remaining states, we calculated the vj(t) as the product of argmax(product of vi(t) of the previous state and the transition probability from the previous state to the current state) and the emission probability of the current state
#For returning the path, we used a dictionary called "parent" which stores the parent state of all the states encountered during the decoding process
from __future__ import division
#from numpy import product
from math import log

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
	def posterior(self, sentence, label):
		tran = 0
		emi = 0
		for i,word in zip(range(len(label)),sentence):
			if i+1 < len(label) and self.transition[label[i+1]+"/"+label[i]]!=0:
				tran += log(self.transition[label[i+1]+"/"+label[i]])
			emi += log(self.count_emission.get(word+"/"+label[i],0.00000001))
		log_pos = log(self.initial[label[0]]) + tran + emi
		return log_pos
#calculating initial probabilities
	def ini_prob(self,data):
		self.count={}
		self.initial ={}
		self.train_len = len(data)
		#print self.train_len
		
		self.speech = ['adj','adv','adp','conj','det','noun','num','pron','prt','verb','x','.']
		for i in range(self.train_len):
			pos= data[i][1][0]
			self.count[pos]=self.count.get(pos,0)+1
		#print self.count
		for word in self.speech:
			self.initial[word] = self.count.get(word)/self.train_len
			#print self.initial
    		#return self.initial

#calculating transition probabilities
	def tran_prob(self,data):
		self.count_numerator ={}
		self.count_denom = {}
		self.transition = {}
		for sentence in data:
			for i in range(len(sentence[1])):
				if i+1!=len(sentence[1]):
					self.count_numerator[sentence[1][i+1]+"after"+sentence[1][i]]=self.count_numerator.get(sentence[1][i+1]+"after"+sentence[1][i],0)+1 
				if i != len(sentence[1][i])-1:	
					self.count_denom[sentence[1][i]]=self.count_denom.get(sentence[1][i],0)+1 	
		for word2 in self.speech:
			for word1 in self.speech:
				self.transition[word2+"/"+word1] = self.count_numerator.get(word2 +"after"+ word1,0)/self.count_denom.get(word1,1)
		#print(self.transition["."+"/"+"adj"])

#calculating emission probabilities
	def emi_prob(self,data):
		#self.totalprob=0
		count_num ={}
		self.count_pos={}
		self.count_emission = {}
		for sentence,pos in data:
			for i in range(len(sentence)):
				count_num[sentence[i]+","+pos[i]] = count_num.get(sentence[i]+","+pos[i],0)+1
				self.count_pos[pos[i]]=self.count_pos.get(pos[i],0)+1 
			
		for sentence,pos in data:
			for word in sentence:
				for parts in self.speech:
					if count_num.get(word+","+parts,0)!=0:
						self.count_emission[word+"/"+parts]=count_num[word+","+parts]/self.count_pos[parts]
					else:
						self.count_emission[word+"/"+parts]=0.0000001/self.count_pos[parts]
		#for s in self.speech:
		#	self.totalprob += self.initial[s]*sum([self.count_emission.get(word +"/"+ s,0) for word in sentence])					
		#return count_emission
# Do the training!
    #
	
   
	def train(self, data):
		self.ini_prob(data)
		self.tran_prob(data)
		self.emi_prob(data)
				
	
    # Functions for each algorithm.
    #
	def simplified(self, sentence):
		self.marginalprob =[]
		
		for word1 in sentence:
			maxprob = 0
			self.marginal={}
			for word in self.speech:
				#if self.totalprob.get(word) != 0:
				self.marginal[word] = (self.count_emission.get(word1 +"/"+word,0.000001) * self.count_pos[word])/sum(self.count_pos.values()) #/self.totalprob.get(word)

			self.marginalprob.append(sorted(self.marginal,key=self.marginal.get,reverse = True)[0])
			
		#print self.marginalprob		
		return self.marginalprob

	def hmm_ve(self, sentence):
		pos_sequence=[]
		alpha={}
		beta ={}
		#calulating forward part
		for i in range(len(sentence)):
			for pos in self.speech:
				if i == 0:
					alpha["s"+str(i)+"="+pos]=self.initial[pos]*self.count_emission.get(sentence[i]+"/"+pos,0.000001)
				else:
					alpha["s"+str(i)+"="+pos] = self.count_emission.get(sentence[i]+"/"+pos,0.000001)*sum([alpha["s"+str(i-1)+"="+s]*self.count_emission.get(sentence[i-1]+"/"+s,0.000001)*self.transition[pos+"/"+s] for s in self.speech])
		#calulating backward part
		
		for  i in range(len(sentence)-1,-1,-1):
			for pos in self.speech:
				if i == len(sentence)-1:
					beta["s"+str(i)+"="+pos] = 1
				else:
					beta["s"+str(i)+"="+pos] = sum([beta["s"+str(i+1)+"="+s]*self.count_emission.get(sentence[i+1] +"/"+s,0.000001)*self.transition[s+"/"+pos]for s in self.speech])
		#calcualting final prob	
			
		for i  in range(len(sentence)):
			gama = {}
			for pos in self.speech:
				gama["s"+str(i)+"="+pos] = alpha["s"+str(i)+"="+pos]*beta["s"+str(i)+"="+pos]
			pos_tag=sorted(gama,key = gama.get,reverse = True)[0]			
			pos_sequence.append(pos_tag[pos_tag.index("=")+1:])
				 
		return pos_sequence

	def hmm_viterbi(self, sentence):
		mle = []
		v = {}
		parent = {}
		for i in range(len(sentence)):
			v_temp = {}
			for pos in self.speech:
				if i == 0:
					v[pos+"("+str(i)+")"] = self.initial[pos]*self.count_emission.get(sentence[i]
+"/"+pos,0.000001)
					v_temp[pos+"("+str(i)+")"] = v[pos+"("+str(i)+")"]
				else:
					v_mid = {}
					for s in self.speech:
						v_mid[s+"("+str(i-1)+")"] = v[s+"("+str(i-1)+")"]*self.transition[pos+"/"+s]
					parent[pos+"("+str(i)+")"] = sorted(v_mid,key=v_mid.get,reverse=True)[0]
					v[pos+"("+str(i)+")"] = max(v_mid.values())*self.count_emission.get(sentence[i]
+"/"+pos,0.000001)
					v_temp[pos+"("+str(i)+")"] = v[pos+"("+str(i)+")"]
		final_pos = sorted(v_temp,key=v_temp.get,reverse=True)[0]
		mle.append(final_pos[0:final_pos.index("(")])
		while True:
			if final_pos[final_pos.index("("):] != "(0)":
				parent1 = parent[final_pos]
				mle.insert(0,parent1[0:parent1.index("(")])
				final_pos = parent1
			else: 
				break
		return mle
		
		


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
	def solve(self, algo, sentence):
		if algo == "Simplified":
			return self.simplified(sentence)
		elif algo == "HMM VE":
			return self.hmm_ve(sentence)
		elif algo == "HMM MAP":
			return self.hmm_viterbi(sentence)
		else:
			print "Unknown algo!"
