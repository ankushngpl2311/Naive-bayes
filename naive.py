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

def prob_numerical(dic,features):
	dic_prob={}

	for f in features:
		dic_prob[f]={}
		pos_list= dic[f][true]
		neg_list= dic[f][false]

		pos_mean = mean(pos_list)
		pos_stdev= stdev(pos_list,pos_mean)
		
		print("feature = ",f)
		print("mean= ",pos_mean)
		print("stdev= ",pos_stdev)
		dic_prob[f][true] = {'mean':pos_mean,'stdev':pos_stdev}

		neg_mean= mean(neg_list)
		neg_stdev = stdev(neg_list,neg_mean)

		dic_prob[f][false] ={'mean':neg_mean,'stdev':neg_stdev}


	return dic_prob




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

print("categorical")
cat_dic=filterCategorical(data_train,categorical,cat_uniq)
print(cat_dic)
print("\n\n\n\n\nnumerical")
num_dic=filterNumerical(data_train,numerical)
print(num_dic)

print("\n\ncat prob")
print(prob_cat(cat_dic,categorical,cat_uniq))

print("\n\n\nnum prob")
print(prob_numerical(num_dic,numerical))


# print("true")
# print(len(data_train.loc[data_train[label]==true]))
# print("false")
# print(len(data_train.loc[data_train[label]==false]))














