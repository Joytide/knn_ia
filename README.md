# KNN Report

https://github.com/Joytide/knn_ia

## Abstract

This report aims to reconstruct my thought process and give broad overview of the code written for a handmade k nearest neighbour algorithm.

In this report and in the code, I will be referring to an element as a instance of an element class as well as a row of data in the datasets available. A element 6-dimension vector to represent its data and is either of a known label, or unknown label and we have to predict its classification.

#### Datasets

We are given 3 datasets:

- A base dataset made up of 800 rows with known label
- A pre-test dataset made up of 800 rows with known label
- A final dataset with unknown labels.



Our goal will be to use the two first dataset to identify best distance methods, k and weights for the classification process, and then use these parameters to predict the final dataset's labels.

## Parameters

### Distance methods

$$
a\text{ and }b\text{ are both 6-dimensionnal vectors illustrating an element.}\\
\mu\text{, }\sigma\text{, }min\text{, }max\text{ are all calculated with the combination of the training and evaluation dataset.}
$$

##### Euclidian distance

$$
Distance (a,b)=\sqrt{\sum_{i=1}^6{(a_i-b_i)^2}}
$$



##### Normalized distance through mean and standard

$$
\text{}Distance (a,b)=\sqrt{\sum_{i=1}^6{(\frac{a_i-\mu_i}{\sigma_i}-\frac{b_i-\mu_i}{\sigma_i})^2}}
$$



##### Normalized distance through max

$$
\text{}Distance (a,b)=\sqrt{\sum_{i=1}^6{(\frac{a_i}{max_i}-\frac{b_i}{max_i})^2}}
$$



##### Normalized distance through max and min

$$
\text{}Distance (a,b)=\sqrt{\sum_{i=1}^6{(\frac{a_i-min_i}{max_i-min_i}-\frac{b_i-min_i}{max_i-min_i})^2}}
$$



##### Cosine similarity

$$
\text{}Distance (a,b)=\frac{\sum_{i=1}^6{a_i.b_i}}{\sqrt{\sum_{i=1}^6{a_i^2}}.{\sqrt{\sum_{i=1}^6{a_i^2}}}}
$$



### K

K is the number of significative neighbours that we will take into account when choosing for the expected label. It can be as low as 1 (but 1 suggests massive overfitting) but its upper bound can be much more variable, to be safe we will take a the upper bound to be the ceil of the square root of the length of the training dataset (here ~29 for ~800 rows).

We will iterate to find the best k and the best distance function in the following training function:

```python
def training(train_dataset, eval_dataset):
    k_possible=[x for x in range(1,ceil(sqrt(len(train_dataset))))]
    dist_meth_possible={"euclidian_dist":Element.euclidian_dist, "cos_sim":Element.cos_sim, "normalized1_dist": Element.normalized1_dist , "normalized2_dist": Element.normalized2_dist, "normalized3_dist" :Element.normalized3_dist,}

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
            k_results[method_name+"_"+str(k)]=accuracy/len(eval_dataset)*100				                                                 print('Meth:', method_name, 'k=', k, "accuracy=", accuracy, '/', len(eval_dataset), str(k_results[method_name + "_" + str(k)]) + "%")

        k_global_results|=k_results
        k_global_results=dict(sorted(k_global_results.items(),key=lambda x:x[1],reverse=True))
   return list(k_global_results.keys())[:10]
```



### Weights

Weights are expected to balance our calculations in 2 places:

- **Giving higher weights to closer neighbours during the label process.**

This is very simple to implement as shown in the code snippet below, each classification is given the inverse of the distance of the element, meaning a closer element will increase the probability that this label is chosen.

```python
for neighbor,dist in neighbors.items():
    if neighbor.classification not in weights:
        weights[neighbor.classification]=1/dist
    else:
        weights[neighbor.classification]+=1/dist
```

- **Giving higher or lower weights to certain column of a vector.**

This is based purely on statistics and I haven't had the time to do it, but some column in the vectors can have strong  or low correlation in the choosing of the class, as we can see in the statistical summary in the iris dataset:

```
Summary Statistics:
	         Min  Max   Mean    SD   Class Correlation
   sepal length: 4.3  7.9   5.84  0.83    0.7826   
    sepal width: 2.0  4.4   3.05  0.43   -0.4194
   petal length: 1.0  6.9   3.76  1.76    0.9490  (high!)
    petal width: 0.1  2.5   1.20  0.76    0.9565  (high!)
```

So only using the 2 last columns in the vector to predict might remove false-positive errors and increase accuracy.



## Calculations

Now, we have to split our datasets into 2 or 3 datasets: the training dataset, the evaluation dataset and the validation dataset, and run tests to gather the best distance method and k, they will be displayed as ```method_k```.

##### First test

For the first test, we use 80% and 20% of the first dataset for the training and evaluation dataset, the results are very clear: the cosine similarity functions wins heavily by placing 10 cosine similarity function 

```python
Best couples: ['cos_sim_6', 'cos_sim_5', 'cos_sim_7', 'cos_sim_8', 'cos_sim_9', 'cos_sim_10', 'cos_sim_11', 'cos_sim_12', 'cos_sim_13', 'cos_sim_14']
```

After a few runs, we always find the same results, cosine similarity with a k of around 6 seems to be the best fit!

##### Second test

Let's try it to validate it with our second dataset, using the first one as our training and the second one as our evaluation.

```python
Best couples: ['normalized3_dist_15', 'normalized3_dist_23', 'normalized3_dist_20', 'normalized3_dist_21', 'normalized3_dist_24', 'normalized3_dist_28', 'normalized3_dist_19', 'normalized3_dist_22', 'normalized3_dist_27', 'normalized3_dist_25']
# Meth: cos_sim k= 6 accuracy= 52 / 803 6.47571606475716%
```

Only Euclidian distance and very, very low cosine similarity. This is very unexpected.

##### Third test

Let's try again with a switch between the 2 dataset.

```python
Best couples: ['normalized3_dist_13', 'normalized3_dist_14', 'normalized3_dist_15', 'normalized3_dist_19', 'normalized3_dist_23', 'normalized3_dist_24', 'normalized3_dist_25', 'normalized3_dist_12', 'normalized3_dist_17', 'normalized3_dist_20']
# Meth: cos_sim k= 9 accuracy= 204 / 803 25.404732254047325% is the best cosine similarity
```

Okay, it seems that cosine similarity doesn't work well on a unknown dataset, or "in the wild".

##### Third test

Okay, cosine similarity seems to work but also seems to overfit a lot... In this case we can try to merge the two datasets and see what happens.

```python
Best couples: ['cos_sim_9', 'cos_sim_11', 'cos_sim_8', 'cos_sim_10', 'cos_sim_12', 'cos_sim_7', 'cos_sim_13', 'cos_sim_14', 'cos_sim_15', 'cos_sim_28']
```

As expected, cosine similarity works very well in the known, but in the wild it will probably under perform, we need to choose the safest method and k for our final validation dataset, so we're going with the third normalized distance method, the min-max.

##### Best k

After a few test runs with evaluation data taken from the same dataset, it seems that any k between 7 and 15 will yield a ~90% accuracy.

After testing it in the wild, by taking evaluation data from another dataset (using first dataset as training and second as evaluation for test 1 and vice versa for test 2), the best k seems to be at least 15 in both cases, since the accuracy increases from ~50% to ~85% from $$k=1$$ to $$k=15$$ and then stays around that number for test 1, probably thanks to the weight of distance, meaning that more distant neighbours doesn't not mean more noise since closer ones are much more weighted. For test 2 it's always at ~85% but peaks around 13 to 15 too. So let's go with that and generate our final data.





## Final estimation

We're now ready to finally estimate the labels of the final dataset. We are using the third normalized functions with a k of 15 and appending to the output file with the validation method:

```python
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
```

























