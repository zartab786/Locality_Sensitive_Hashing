from __future__ import division
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
import time
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
import pandas as pd



#run code
#spark-submit --packages org.mongodb.spark:mongo-spark-connector_2.12:3.0.1 spark_mongo.py
#spark-submit --packages org.mongodb.spark:mongo-spark-connector_2.12:3.0.1 --conf spark.executor.instances=1 spark_mongo.py



#creating spark session

my_spark = SparkSession \
    .builder \
    .appName("myApp") \
    .config("spark.mongodb.input.uri", "mongodb://127.0.0.1/BDA_A3.coll") \
    .config("spark.mongodb.output.uri", "mongodb://127.0.0.1/BDA_A3.coll") \
    .getOrCreate()


#.config("spark.executor.instances", 2)\
#.config("spark.executor.cores", 2)\


#sample load to check for errors
df = my_spark.read.format("mongo").option("uri","mongodb://127.0.0.1/BDA_A3.questions1").load()

#df.show()


# p_df= df.toPandas()

# # print(p_df)

# #taking first 100000 rows
train2=df.limit(100000)

# train2.show()

# hold=train2.count()
# print(hold)

# #extracting questions and saving in list

q_id1= train2.select("qid1").rdd.flatMap(lambda x: x).collect()
q_col1=train2.select("question1").rdd.flatMap(lambda x: x).collect()

# print(q_id1)

q_id2=train2.select("qid2").rdd.flatMap(lambda x: x).collect()
q_col2=train2.select("question2").rdd.flatMap(lambda x: x).collect()


duplicate_status=train2.select("is_duplicate").rdd.flatMap(lambda x: x).collect()

# print(q_col2[0])


#ground truth
dict_duplicate={}
for i in range(len(q_id1)):

  hold= str(q_id1[i]) +""+ str(q_id2[i])

  dict_duplicate[hold]= duplicate_status[i]

# print(dict_duplicate)


#dictionary of questions
dict_questions={}
for i in range(len(q_id1)):

  dict_questions[q_id1[i]]= q_col1[i]

for i in range(len(q_id2)):

  dict_questions[q_id2[i]]= q_col2[i]

print(len(dict_questions))


#list_keys
key_id= list(dict_questions.keys())
# print(key_id)


#list of questions
list_questions=[]

for id in key_id:

  list_questions.append(dict_questions[id])


print(list_questions[0])


#using TF-IDF vectorization from Spark API

#creating dataframe from questions
cols=['id', 'question']
data=[]

for i in range(len(list_questions)):
  tup=(int(key_id[i]),list_questions[i])
  data.append(tup)

df2 = my_spark.createDataFrame(data).toDF(*cols)

# df2.show()

#first step tokenization
tokenizer = Tokenizer(inputCol="question", outputCol="words")
wordsData = tokenizer.transform(df2)

#calculating term frequency
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=10000)
featurizedData = hashingTF.transform(wordsData)

#calculating idf
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)


#sparse vectors in a list
hold=rescaledData.select("features").collect()

#list_vectors 
#storing vectors in dict
dict_vect={}
key_id=list(key_id)

for i in range(len(key_id)):
  dict_vect[key_id[i]]=hold[i][0]

# print(((dict_vect[1].toArray())))


#Creating Multiple HashTables

#generating random vectors to generate min-hash signatures
##function to generate random vectors

def rand_vect(num_vects, length_vect):

  size = (num_vects, length_vect)

  rand_vectors = np.random.uniform(-1, 1, size)

  return rand_vectors


#hashtables list
list_hashtable=[]

#computing dot-product of random vectors with each question vector 
# from tqdm.notebook import tqdm as tqdm

num_tables= 1  #number of hashtables to be made
list_dict_hash=[]  #to hold multiple minhash dicts

from tqdm import tqdm

for i in tqdm(range(num_tables)):
  dict_min_hash={}

  rand_vects= rand_vect(35,10000)

  for key in tqdm(dict_vect):

    #converting sparse array to numpy array
    quest= dict_vect[key].toArray()
    
    # to hold the minhash for the given vector
    minhash=""

      #function call to create random vectors 
    
    for vect in rand_vects:

      #taking dot product of question vector with random vector
      
      hold_dot= np.dot(quest,vect)

      if hold_dot > 0:

        minhash= minhash + str(0)
      
      else:

        minhash= minhash + str(1)


    dict_min_hash[key]=minhash

  list_dict_hash.append(dict_min_hash)
  
      
# # print((dict_min_hash[8]))
# # print(list_dict_hash[0])

#putting the min-hash values in HashTables

for i in tqdm(range(num_tables)):

  dict_lsh={}

  dict_min_hash=list_dict_hash[i]

  for key in tqdm(dict_min_hash):

    hold = dict_min_hash[key]

    #saves in list in case of collision multiple values get in a list

    if hold in dict_lsh:
      hold2= dict_lsh[hold]
      hold2.append(key)
      
      dict_lsh[hold]= hold2

    else:

      hold2=[]
      hold2.append(key)

      dict_lsh[hold]=hold2
    
  list_hashtable.append(dict_lsh)


 #checking candidate pairs

fp=0
tp=0

dict_candidate_pairs={}

for i in tqdm(range(num_tables)):

  dict_lsh=list_hashtable[i]

  for key in tqdm(dict_lsh):

    hold = dict_lsh[key]

    if len(hold)>1:  #collision i.e candidate pairs exist

      for val1 in hold:

        for val2 in hold:

          hold2= str(val1)+""+str(val2)
          hold3= str(val2)+""+str(val1)
          #checking for true positive

          if hold2 in dict_duplicate and hold2 not in dict_candidate_pairs:
            if dict_duplicate[hold2]==1:
                tp+=1
            else:
                fp+=1
            
            dict_candidate_pairs[hold2]=1
            dict_candidate_pairs[hold3]=1

          elif hold3 in dict_duplicate and hold3 not in dict_candidate_pairs:
            if dict_duplicate[hold3]==1:
                tp+=1
            else:
                fp+=1
            
            dict_candidate_pairs[hold2]=1
            dict_candidate_pairs[hold3]=1
          
          elif hold2 not in dict_candidate_pairs and hold3 not in dict_candidate_pairs:
            fp+=1

            dict_candidate_pairs[hold2]=1
            dict_candidate_pairs[hold3]=1
  


print(tp,fp)

#calculate precision
pre= tp/(tp+fp)

print("Precision: ",pre)


#total duplicate pairs in ground truth
count_tot=0
for key in dict_duplicate:

  if dict_duplicate[key]==1:
    count_tot+=1


#calculate recall

rec= tp/(count_tot)

print("Recall : ",rec)


#truth labels
tp=0
fp=0
fn=0
tn=0
for key in dict_candidate_pairs:

  if key in dict_duplicate:
    if dict_duplicate[key]==1:
      tp+=1
    elif dict_duplicate[key]==0:
      fp+=1
  else:
      fp+=1
  

for key in dict_duplicate:
  if key not in dict_candidate_pairs and dict_duplicate[key]==1:
    fn+=1
  elif key not in dict_candidate_pairs and dict_duplicate[key]==0:
    tn+=1



from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# cf_matrix = confusion_matrix(y_true, y_predicted)

cf_matrix2=[[tn,fp],[fn,tp]]
fig = plt.figure(figsize=(10,7))

heatmap=sns.heatmap(cf_matrix2, annot=True,linewidths=.5, fmt='.0f')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

