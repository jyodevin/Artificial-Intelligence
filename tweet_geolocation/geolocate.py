#part2
#Bharat Malala, Jyothi Pranavi, Harshit Krishnakumar
from __future__ import division
import sys
import string
import math
#reading the three files
trainfile = sys.argv[1]
testfile = sys.argv[2]
outputfile = sys.argv[3]



train = open(trainfile,'r')
test = open(testfile, 'r')
output=open(outputfile,'w')
train_data = train.readlines()

train.close()

counts={}
symbols = "{}[](),._/!-:;$&*0123456789^@#%+=<>'"
len_train=len(train_data)

#cities=['losangelesca','manhattanny','sanfranciscoca','houstontx','sandiegoca','chicagoil','philadelphiapa','torontoontario','atlantaga','bostonma','orlandofl',\
#'washingtondc']
stop = ['theirs','him','have','im','see','might','been','my','just','further','me','dont','am','wouldnt','up','here','our','some','doing','it','being','hadn','haven','needn','weren','ve','no','from','too','she','or','because','had','these','yourself','themselves','how','before','to','off','ours','in','won','is','an','myself','until','if','into','same','while','both','through','couldnt','hers','only','was','mustnt','for','over','as','than','itself','does','between','of','few','most','nor','shan','yours','i','so','its','were','re','you','what','down','he','do','they','himself','after','each','doesnt','out','aint','other','why','did','their','be','should','such','who','yourselves','there','all','herself','them','hasnt','will','about','ourselves','that','having','wasnt','by','are','and','her','when','which','whom','your','with','own','has','shouldnt','once','then','arent','a','more','this','didnt','but','those','the','now','above','his','any','during','against','isnt','not','below','we','under','where',"i'm",'ill','very','on','again','can','at']

train1=[]
train2 =[]
bagwords=[]
countofwords ={}
countof_allwords={}
city_counts={}
city_prior={}
posterior={}
likelihood ={}
#newcount =[]
#ps = PorterStemmer()
cutoff=5
# 512 is the best cutoff with an accuracy of 41.2%


def trainfun(train_data):

    for i in range(len_train):
        train1.append(train_data[i].strip().split(" "))
        train2.append([word.translate(None,symbols).strip().replace("\r","") for word in train1[i]])
        train2[i] = filter(None,train2[i])
        #print(train2[1:5])
        #print(train2[i][3])
    #    train3.append([ps.stem(nword) for nword in train2[i]])
        for j in range(len(train2[i])):
            #print(tweet)
            train2[i][j]=train2[i][j].lower()
	    train2[i][j]=train2[i][j].decode('ascii','ignore').encode('ascii','ignore')
            if train2[i][j] not in stop:
                if j == 0:
                    city_counts[train2[i][j]]=city_counts.get(train2[i][j],0)+1
                    city_prior[train2[i][j]] = city_counts.get(train2[i][j])/len_train
                else:
                    counts[train2[i][j]]=counts.get(train2[i][j],0)+1
                    countofwords[train2[i][0]+':'+train2[i][j]]=countofwords.get(train2[i][0]+':'+train2[i][j],0)+1
           
    cities = city_counts.keys()                           
    #print cutoff
    for item in counts.items():
        if item[1]>cutoff and item[0] != '':
            		bagwords.append(item[0])
   
   
    for city in cities:

        for word in bagwords:
            countof_allwords[city] = countof_allwords.get(city,0)+countofwords.get(city+':'+word,0)
            likelihood[word+'/'+city] = (1+countofwords.get(city+':'+word,0))/(len(bagwords)+countof_allwords.get(city,0))
   
    
   
    return bagwords,likelihood,city_prior,cities
#------------------------------------------------------------------------
#for test
test_data = test.readlines()
len_test = len(test_data)
test.close()
test1 =[]
test2=[]

def testfun(test_data,bagwords,likelihood,city_prior,cities):
    final_city =[]
    for i in range(len_test):
        posterior1=[]
        test1.append(test_data[i].strip().split(" "))
        #print("test1",test1)
        test2.append([word.translate(None,symbols).strip().replace("\r","") for word in test1[i]])
        #print("test2",test2)
        test2[i] = filter(None,test2[i])
        #print(len(test2[i]),test2[i])
        for city in cities:
            posterior_city=city_prior.get(city)
            for j in range(len(test2[i])):
            #print(word)
                test2[i][j]=test2[i][j].lower()
		test2[i][j]=test2[i][j].decode('ascii','ignore').encode('ascii','ignore')
                if test2[i][j] not in stop and test2[i][j] in bagwords:
                    posterior_city*=likelihood.get(test2[i][j]+'/'+city)
            posterior1.append(posterior_city)
	label=cities[posterior1.index(max(posterior1))]
        final_city.append(label)
	output.write(label+','+test_data[i])
    
    
    
    output_format="{0:>20}:{1:>10}"
    for city in cities:
	word_association=[]
	for word in bagwords:
		total_prob=0
		for prior in city_prior.items():
			total_prob+=prior[1]*likelihood.get(word+'/'+prior[0]) 	
		word_association.append((city_prior.get(city)*likelihood.get(word+'/'+city))/total_prob)
	print '*'*50
	print "Top 5 associated words for",city.upper()
	print '*'*50
	for i in range(0,5):
		print output_format.format(bagwords[word_association.index(sorted(word_association,reverse=True)[i])],word_association[word_association.index(sorted(word_association,reverse=True)[i])])
	
   
   
    val = 0
    for i in range(len_test):
        if test2[i][0] == final_city[i]:
            val+=1

    print (len(bagwords))
    model_accuracy = (val/len_test)*100
    print "acuucray",model_accuracy,"%"

bagwords,likelihood,city_prior,cities=trainfun(train_data)
testfun(test_data,bagwords,likelihood,city_prior,cities) 
	
