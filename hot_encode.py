import pandas as pd
import numpy as np
import csv
import numpy as np
import string
data = pd.read_csv(r'C:\Users\USER\Desktop\Gitpull\Hot_encoding_gene\test.csv')
df_1= pd.DataFrame(data, columns= ['Sequence 1'])
df_2= pd.DataFrame(data, columns= ['Sequence 2'])
df_3= pd.DataFrame(data, columns= ['Abundance Ratio'])
test_num=data.to_numpy() #convert to numpy

cat_array=df_1.to_numpy() #to obtain sequence1, for sequence 2 replace with df_2.to_numpy())
print(len(cat_array))
num_cat=len(cat_array)
def get_matrix(j):
    mat = []
    check=[]
    result1=[]
    # len_doc = 26
    for i in range(0, len(j)): 
        check=j[i]
        # print(check) #need to be single character
        for letter in string.ascii_uppercase: #A-Z 26 english alphabet
            if check==letter:
                ans=1
            else:
                ans=0
            mat.append(ans)  
        # print(result1)  
    return mat
result=[]
for i in range(0, num_cat):
    test1=cat_array[i]
    final_mat = []
    result_mat = []
    for j in test1:
        # print('\n',j)
        ans_1=get_matrix(j)
     
    final_mat.append(ans_1) 
        # print('I=',i)
    result.append(final_mat)

with open('sequence1.csv', 'w', newline = '') as csvfile:
    my_writer = csv.writer(csvfile, delimiter = ' ')
    my_writer.writerows(result)