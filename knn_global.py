from math import sqrt,ceil
from random import randint
from numpy import dot, array, mean, max, minimum, std
import numpy as np
from numpy.linalg import norm

class Element():
    def __init__(self,data,classification):
        self.data=data
        self.classification=classification

    def __str__(self):
        a='('
        for el in self.data:
            a+=str(el)+','
        a+=')'
        if self.classification:
            a+=self.classification
        return a

    def euclidian_dist(self,other):
        sum=0
        for i in range(len(self.data)):
            sum+=(self.data[i]-other.data[i])**2
        return sqrt(sum)

    def normalized1_dist(self,other,mean,std):
        sum=0
        for i in range(len(self.data)):
            sum+=(((self.data[i]-mean[i])/std[i])-((other.data[i]-mean[i])/std[i]))**2
        return sqrt(sum)

    def normalized2_dist(self,other,train_max):
        sum=0
        for i in range(len(self.data)):
            sum+=((self.data[i])/train_max[i]-(other.data[i])/train_max[i])**2
        return sqrt(sum)

    def normalized3_dist(self, other, train_min, train_max):
        sum=0
        for i in range(len(self.data)):
            sum+=(((self.data[i]-train_min[i])/(train_max[i]-train_min[i]))-((other.data[i]-train_min[i])/(train_max[i]-train_min[i])))**2
        return sqrt(sum)

    def cos_sim(self,other):
        return 1-dot(self.data,other.data)/(norm(self.data)*norm(other.data))

    def nearest_neighbors(self, dataset, k, method_name, dist_meth, train_mean, train_std, train_min, train_max):
        dists={}    # other:dist
        for other in dataset:
            if self.data!=other.data:
                    if method_name=="euclidian_dist":
                        dists[other]=dist_meth(self,other)
                    elif method_name=="normalized1_dist":
                        dists[other]=dist_meth(self,other, train_mean, train_std)
                    elif method_name=="normalized2_dist":
                        dists[other]=dist_meth(self,other, train_max)
                    elif method_name=="normalized3_dist" :
                        dists[other]=dist_meth(self,other, train_min , train_max)
                    elif method_name=="cos_sim":
                        dists[other]=dist_meth(self,other)

        return dict(sorted(dists.items(),key=lambda x:x[1])[:k]) #Send them back cut by k and sorted by distance result







def load_data(filename,split_prc=None,merge_file=None,random=True):
    raw_datas=[]

    with open(filename,"r") as f:
        lines=f.readlines()
        if merge_file:
            with open(merge_file,"r") as merge:
                lines+=merge.readlines()
        if random:
            while len(lines)>0:
                raw_datas.append(lines.pop(randint(0,len(lines)-1)).strip().split(","))
        else:
            for line in lines:
                raw_datas.append(line.strip().split(","))
        if split_prc:
            train_dataset=[]
            eval_dataset=[]
            for data in raw_datas[:round(len(raw_datas)*split_prc/100)]:
                train_dataset.append(Element([float(data[0]),float(data[1]),float(data[2]),float(data[3]),float(data[4]),float(data[5])],data[6]))
            for data in raw_datas[round(len(raw_datas)*split_prc/100):]:
                eval_dataset.append(Element([float(data[0]),float(data[1]),float(data[2]),float(data[3]),float(data[4]),float(data[5])],data[6]))
            return train_dataset, eval_dataset
        else:
            validation_set=[]
            for data in raw_datas:
                try:
                    validation_set.append(Element([float(data[0]),float(data[1]),float(data[2]),float(data[3]),float(data[4]),float(data[5])],data[6]))
                except:
                    validation_set.append(Element([float(data[0]),float(data[1]),float(data[2]),float(data[3]),float(data[4]),float(data[5])],None))

            return validation_set
    


def prediction(neighbors):
    weights={}
    for neighbor,dist in neighbors.items():
        if neighbor.classification not in weights:
            weights[neighbor.classification]=1/dist
        else:
            weights[neighbor.classification]+=1/dist
    return sorted(weights.items(),key=lambda x:x[1],reverse=True)[0][0]




def training(train_dataset, eval_dataset):
    k_possible=[x for x in range(1,ceil(sqrt(len(train_dataset))))]
    #dist_meth_possible={"euclidian_dist":Element.euclidian_dist, "cos_sim":Element.cos_sim, "normalized1_dist": Element.normalized1_dist , "normalized2_dist": Element.normalized2_dist, "normalized3_dist" :Element.normalized3_dist,}
    dist_meth_possible={"normalized3_dist" :Element.normalized3_dist}

    k_global_results={}

    arr=array([obj.data for obj in train_dataset+eval_dataset])
    train_mean = mean(arr,axis=0)
    train_std = std(arr,axis=0)
    train_max = np.max(arr,axis=0)
    train_min = np.min(arr,axis=0)

    for method_name,dist_method in dist_meth_possible.items():
        k_results={}
        for k in k_possible:
            accuracy=0
            for el in eval_dataset:

                expected=prediction(el.nearest_neighbors(train_dataset, k, method_name, dist_method, train_mean, train_std,train_min, train_max ))

                if el.classification==expected:
                    accuracy+=1
                    
            k_results[method_name+"_"+str(k)]=accuracy/len(eval_dataset)*100
            print('Meth:',method_name,'k=',k,"accuracy=",accuracy,'/',len(eval_dataset),str(k_results[method_name+"_"+str(k)])+"%")
        
        print("Global results:")
        k_global_results|=k_results
        k_global_results=dict(sorted(k_global_results.items(),key=lambda x:x[1],reverse=True))
        print(k_global_results)
        print("Best couples:",list(k_global_results.keys())[:10])

    
    return list(k_global_results.keys())[:10]


def validation(train_dataset, validation_dataset, k, method_name, dist_method,outputfile=None):
    arr=array([obj.data for obj in train_dataset+validation_dataset])
    train_mean = mean(arr,axis=0)
    train_std = std(arr,axis=0)
    train_max = np.max(arr,axis=0)
    train_min = np.min(arr,axis=0)
    
    accuracy=0
    content=[]
    for el in validation_dataset:
        expected=prediction(el.nearest_neighbors(train_dataset, k, method_name, dist_method, train_mean, train_std,train_min, train_max ))
        if outputfile:
            content.append(expected+"\n")
        else:
            if el.classification==expected:
                accuracy+=1

    if outputfile:
        with open(outputfile,"w+") as file:
            file.writelines(content)
    else:
        result=accuracy/len(validation_dataset)*100
        print('k=',k,"accuracy=",accuracy,'/',len(validation_dataset),str(result)+"%")
        return result
    
    




def main():
    train_dataset=load_data("./data.csv",merge_file="./preTest.csv") #,merge_file="./preTest.csv"
    #train_dataset,validation=load_data("./preTest.csv",70)
    #k_training=training(train_dataset+eval_dataset, validation)
    validation_dataset=load_data("./finalTest.csv",random=False)

    result=validation(train_dataset,validation_dataset, 15, 'normalized3_dist',Element.normalized3_dist,"labels.txt")
    #print(result)


main()