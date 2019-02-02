import pandas as pd
from sklearn.model_selection import train_test_split
import math


true=1
false=0
label = 'accepted'
#cat={feature1:{'sunny':{'yes':2,'no':3},'rainy':{'yes':3,'no':2} }, .....,total:{'yes':9,'no':5}  }

def filterCategorical(data,features,cat_uniq):
	dic={}

	for f in features:
		if(f== label):
			continue
		dic[f]={}

		for val in cat_uniq[f]:
			toappend={}
			
			pos= len(data.loc[(data[f]==val) & (data[label]==true)])
			neg= len(data.loc[(data[f]==val) & (data[label]==false)])

			toappend[true]=pos
			toappend[false] =neg

			dic[f][val]=toappend

	pos= len(data.loc[data[label] == true])
	neg= len(data.loc[data[label] == false])

	toappend={}
	toappend[true]=pos
	toappend[false]= neg
	dic["total"]= toappend


	return dic


def mean(a):

	l= len(a)

	s= sum(a)

	return s/l

def stdev(a,mean):

	el=[]
	l= len(a)

	for i in a:
		el.append(pow(i-mean,2))

	s= sum(el)

	variance= s/(l-1)

	return math.sqrt(variance) 




#num= {feature:{'yes':[83,....,98],'no':[7,....,98]},}
def filterNumerical(data,features):
	dic={}

	for f in features:
		# dic[f]={}


		pos= list( set ( data.loc[ data[label] == true ][f] ) )
		neg= list( set ( data.loc[ data[label] == false ][f] ) )

		toappend={}
		toappend[true]= pos
		toappend[false] = neg
		
		dic[f]=toappend


	return dic

def summary_numerical(dic,features):
	dic_prob={}

	for f in features:
		dic_prob[f]={}
		pos_list= dic[f][true]
		neg_list= dic[f][false]

		pos_mean = mean(pos_list)
		pos_stdev= stdev(pos_list,pos_mean)
		
		# print("feature = ",f)
		# print("mean= ",pos_mean)
		# print("stdev= ",pos_stdev)
		dic_prob[f][true] = {'mean':pos_mean,'stdev':pos_stdev}

		neg_mean= mean(neg_list)
		neg_stdev = stdev(neg_list,neg_mean)

		dic_prob[f][false] ={'mean':neg_mean,'stdev':neg_stdev}


	return dic_prob



def gaussian(x,mean,stdev):
	e= - (pow((x-mean),2) / (2*pow(stdev,2)))

	exp= math.exp(e)

	prob = (1/(math.sqrt(2*math.pi) * stdev)) * exp

	return prob
	




def prob(fav,total):
	return fav/total

#cat={feature1: {'sunny':{'yes':2/9,'no':3/5}}}
def prob_cat(dic,features,cat_uniq):

	pos_total= dic['total'][true]
	neg_total = dic['total'][false]

	dic_prob={}

	for f in features:
		if(f== label):
			continue
		dic_prob[f]={}

		for val in cat_uniq[f]:
			toappend={}
			
			pos= prob(dic[f][val][true], pos_total)
			neg= prob(dic[f][val][false], neg_total)
			# pos= len(data.loc[(data[f]==val) & (data[label]==true)])
			# neg= len(data.loc[(data[f]==val) & (data[label]==false)])

			toappend[true]=pos
			toappend[false] =neg

			dic_prob[f][val]=toappend

	return dic_prob



def cond_prob(y,x,cols,categorical,numerical,cat_prob,num_summary,train_len):

	if(y==true):
		prob_label = prob(train_len[0],train_len[0]+train_len[1])

	if(y==false):
		prob_label = prob(train_len[1],train_len[0]+train_len[1])


	prob_final = prob_label
	for count,i in enumerate(cols):

		if(i in categorical):
			prob_final = prob_final * cat_prob[i][x[count]][y]

		if(i in numerical):
			mean = num_summary[i][y]['mean']
			stdev = num_summary[i][y]['stdev']

			prob_final = prob_final * gaussian(x[count],mean,stdev)

	return prob_final









def predict(x,cat_prob,num_summary,categorical,numerical,train_len):

	y_predict=[]
	cols= x.columns
	for row in x.itertuples():
		r= [i for i in row]
		r.pop(0)
		prob_yes= cond_prob(true,r,cols,categorical,numerical,cat_prob,num_summary,train_len)
		prob_no= cond_prob(false,r,cols,categorical,numerical,cat_prob,num_summary,train_len)

		if(prob_yes>prob_no):
			pred_val= true
		else:
			pred_val= false

		y_predict.append(pred_val)


	return y_predict



def accuracy(ytest,ypredict):
	c=0
	l =len(ytest)

	for count,i in enumerate(ytest):
		if(ytest[count] == ypredict[count]):
			c= c +1
	# print("count= ",c)
	# print("total= ",l)

	return c/l



data= pd.read_csv("data.csv")

l=['id','age','experience','income','zip','familysize','spending','education','mortgage','accepted','securityacc','cd','internetbanking','creditcard']


data.columns= l

x= data[['age','experience','income','zip','familysize','spending','education','mortgage','securityacc','cd','internetbanking','creditcard']]
y= data["accepted"]


xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2)


categorical = ['zip','education','securityacc','cd','internetbanking','creditcard','accepted']
numerical= ['age','experience','income','familysize','spending','mortgage']

cat_uniq={}

for i in categorical:
	cat_uniq[i]= list(set(data[i]))

l= [xTrain,yTrain]
data_train=pd.concat(l,axis=1)

# print("categorical")
cat_dic=filterCategorical(data_train,categorical,cat_uniq)
# print(cat_dic)
# print("\n\n\n\n\nnumerical")
num_dic=filterNumerical(data_train,numerical)
# print(num_dic)

# print("\n\ncat prob")
cat_prob=prob_cat(cat_dic,categorical,cat_uniq)
# print(cat_prob)
# print("accepted prob=  ",cat_prob['accepted'])

# print("\n\n\nnum prob")
num_summary=summary_numerical(num_dic,numerical)
# print(num_summary)


# print("true")
true_len=len(data_train.loc[data_train[label]==true])
# print("false")
false_len=len(data_train.loc[data_train[label]==false])
train_len= [true_len,false_len]

y_predict=predict(xTest,cat_prob,num_summary,categorical,numerical,train_len)

yTest= yTest.tolist()

print("accuracy= ",accuracy(yTest,y_predict)*100)















