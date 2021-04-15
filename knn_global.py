from math import *
from random import randint

def load_data(size=None):
    dataset=[]
    raw_datas=[]
    with open("./TD3/data.csv","r") as f:
        lines=f.readlines()
        if not size:
            size=len(lines)
        while len(raw_datas)!=size:
            raw_datas.append(lines.pop(randint(0,len(lines)-1)).strip().split(","))
        for data in raw_datas:
            dataset.append(Element([float(data[0]),float(data[1]),float(data[2]),float(data[3]),float(data[4]),float(data[5])],data[6]))
    return dataset



class Element():
    def __init__(self,data,classification):
        self.data=data
        self.classification=classification

    def __str__(self):
        a='('
        for el in self.data:
            a+=str(el)+','
        a+=')'+self.classification
        return a

    def dist(self,other):
        sum=0
        for i in range(len(self.data)):
            sum+=(self.data[i]-other.data[i])**2
        return sqrt(sum)



def nearest_neighbors(element,dataset,k):
    dists={}
    for other in dataset:
        if element!=other:
            dists[other]=element.dist(other)
    #print("knn:")
    #for flower,dist in sorted(dists.items(),key=lambda x:x[1])[:k]:
    #    print(flower,dist)
    
    return [element for element,dist in sorted(dists.items(),key=lambda x:x[1])[:k]]
    

def prediction(neighbors):
    weights={}
    for neighbor in neighbors:
        if neighbor.classification not in weights:
            weights[neighbor.classification]=1
        else:
            weights[neighbor.classification]+=1
    #print(weights)
    #print(sorted(weights.items(),key=lambda x:x[1],reverse=True)[0][0])
    return sorted(weights.items(),key=lambda x:x[1],reverse=True)[0][0]




def training(dataset):
    
    k_vals={}
    for k in range(1,round(len(dataset)/3)):
        accuracy=0
        for el in dataset:
            #print('Subject:',flower)
            expected=prediction(nearest_neighbors(el,dataset,k))
            #print('Classification:',flower.classification,'Excpected:',expected)
            #input()
            if el.classification==expected:
                accuracy+=1
        k_vals[k]=accuracy/len(dataset)*100
        
        print('k=',k,"accuracy=",accuracy,'/',len(dataset),str(accuracy/len(dataset)*100)+"%")
    k_vals=dict(sorted(k_vals.items(),key=lambda x:x[1],reverse=True))
    print(k_vals)
    print("Best k:",list(k_vals.keys())[:10])
    return list(k_vals.keys())[:10]

def evaluation(k_vals,dataset):
    k_results={}
    for k in k_vals:
        accuracy=0
        for el in dataset:
            expected=prediction(nearest_neighbors(el,dataset,k))
            if el.classification==expected:
                accuracy+=1
        k_results[k]=accuracy/len(dataset)*100
        
        #print('k=',k,"accuracy=",accuracy,'/',len(dataset),str(accuracy/len(dataset)*100)+"%")
    return k_results

'''
TO DO LIST

Do confusion matrix

'''


def knn():
    dataset=load_data()
    k_training=training(dataset)
    result=evaluation(k_training,dataset[100:150])
    print(result)


knn()