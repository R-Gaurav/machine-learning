import string, numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
from sklearn.decomposition import PCA, RandomizedPCA, ProjectedGradientNMF
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import random_projection
from random import sample, shuffle
from scipy.linalg import svd,inv
from sklearn.feature_selection import VarianceThreshold
import linecache

################################################################ DATA SECTION ########################################################################
np.set_printoptions ( formatter={'float':'{: 0.2f}'.format})
# Reading files of features, total 5 normalized training features present.
# data changed from test to train

'''
train_f1 = r'/home/raga/project_2/nus_object/low_level_features/Test_CH.txt'
train_f2 = r'/home/raga/project_2/nus_object/low_level_features/Test_CM55.txt'
train_f3 = r'/home/raga/project_2/nus_object/low_level_features/Test_CORR.txt'
train_f4 = r'/home/raga/project_2/nus_object/low_level_features/Test_EDH.txt'
train_f5 = r'/home/raga/project_2/nus_object/low_level_features/Test_WT.txt'
'''

'''
train_f1 = r'/home/raga/project_2/LITE/LITE_NEW_TEST/Normalized_CH_Lite_Test_new.dat'
train_f2 = r'/home/raga/project_2/LITE/LITE_NEW_TEST/Normalized_CM55_Lite_Test_new.dat'
train_f3 = r'/home/raga/project_2/LITE/LITE_NEW_TEST/Normalized_CORR_Lite_Test_new.dat'
train_f4 = r'/home/raga/project_2/LITE/LITE_NEW_TEST/Normalized_EDH_Lite_Test_new.dat'
train_f5 = r'/home/raga/project_2/LITE/LITE_NEW_TEST/Normalized_WT_Lite_Test_new.dat'
'''

train_f1 = r'/home/raga/project_2/SCENE/SCENE_NEW_TEST/Test_CH_new.txt'
train_f2 = r'/home/raga/project_2/SCENE/SCENE_NEW_TEST/Test_CM_new.txt'
train_f3 = r'/home/raga/project_2/SCENE/SCENE_NEW_TEST/Test_CORR_new.txt'
train_f4 = r'/home/raga/project_2/SCENE/SCENE_NEW_TEST/TEST_EDH_new.txt'
train_f5 = r'/home/raga/project_2/SCENE/SCENE_NEW_TEST/Test_WT_new.txt'


### Creating matrices V1, V2, V3, V4, V5 #######################################
### Where Vi = Di * n where Di = dimension of Vi features and n = total number of training examples

### Vi = Pi * H where Vi = Di * n , and H = d * n where d = reduced dimension of fused multi modal features d <= min(Di)
### Pi is therefore of order Di * d

def create_v_function( train_feature, no_of_images ):
    lst=[]
    i=1
    n = 0
    train_file=open(train_feature)
    for line in train_file:
        n = len( line.split() )
        lst.append([float(val) for val in line.split()])
        if i==no_of_images:
            break
        i=i+1
    """
    counter=1
    for i in lst:
        print counter,type(i),len(i)
        counter+=1
    """
    V=np.matrix(lst, dtype=float) # Here V is n * Di 
    pca = RandomizedPCA( n_components = int( .75*n ) ).fit( V )
    pca.transform( V )
    V=V.transpose() # We need V to be of dimension Di * n
    return V

def get_v_function(n):
    # all the corresponding V matrices have been obtained

    #print 'n', n
    
    V1=create_v_function(train_f1, n)
    V2=create_v_function(train_f2, n)
    V3=create_v_function(train_f3, n)
    V4=create_v_function(train_f4, n)
    V5=create_v_function(train_f5, n)

    '''print V1.shape
    print V2.shape
    print V3.shape
    print V4.shape
    print V5.shape'''
    

    return V1, V2, V3, V4, V5

def initialise_p_function(D,d):
    #P=np.random.rand(D,d)
    P=[]
    for i in range(D):
        Q=[]
        for j in range(d):
            if i == j:
                Q.append(1)
            else :
                Q.append(0)
        P.append(Q)
    P=np.matrix(P)
    return P

def initialise_h_function(d,n):
    H=[]
    for i in range(d):
        Q=[]
        for j in range(n):
            if i == j:
                Q.append(1)
            else :
                Q.append(0)
        H.append(Q)
    H=np.matrix(H)
    return H


def update_p_function(P,V,H,alpha,lamda):
    temp1=np.dot(P,H)
    '''print type(temp1)
    print V.shape
    print temp1.shape'''
    temp1=V-temp1
    temp2=H.transpose()

    temp1=np.dot(temp1,temp2)-lamda*P
    temp1=temp1*alpha

    #print P
    #print temp1
    return temp1+P

def update_h_function(H_old,P1_old,P2_old,P3_old,P4_old,P5_old,V1,V2,V3,V4,V5,alpha,lamda):
    temp1=np.dot(P1_old,H_old)
    temp2=np.dot(P2_old,H_old)
    temp3=np.dot(P3_old,H_old)
    temp4=np.dot(P4_old,H_old)
    temp5=np.dot(P5_old,H_old)

    temp1=V1-temp1
    temp2=V2-temp2
    temp3=V3-temp3
    temp4=V4-temp4
    temp5=V5-temp5

    P1_old=P1_old.transpose()
    P2_old=P2_old.transpose()
    P3_old=P3_old.transpose()
    P4_old=P4_old.transpose()
    P5_old=P5_old.transpose()

    temp1=np.dot(P1_old,temp1)
    temp2=np.dot(P2_old,temp2)
    temp3=np.dot(P3_old,temp3)
    temp4=np.dot(P4_old,temp4)
    temp5=np.dot(P5_old,temp5)

    temp=temp1+temp2+temp3+temp4+temp5-lamda*H_old
    temp=alpha*temp

    return H_old+temp
    
  
def check_ph_function(P1_old,P1_new,P2_old,P2_new,P3_old,P3_new,P4_old,P4_new,P5_old,P5_new,H_old,H_new, difference_parameter): #For checking whether P matrix differs from its previous value and accordingly stop updation

        # Checking the disimilarity between old and new values of Pi....
        
    temp1=P1_new-P1_old
    for k in temp1.flat:
        if abs(k)>difference_parameter:
            t1=False
            break
        else:
            t1=True

    temp2=P2_new-P2_old
    for k in temp2.flat:
        if abs(k)>difference_parameter:
            t2=False
            break
        else:
            t2=True

    temp3=P3_new-P3_old
    for k in temp3.flat:
        if abs(k)>difference_parameter:
            t3=False
            break
        else:
            t3=True

    temp4=P4_new-P4_old
    for k in temp4.flat:
        if abs(k)>difference_parameter:
            t4=False
            break
        else:
            t4=True

    temp5=P5_new-P5_old
    for k in temp5.flat:
        if abs(k)>difference_parameter:
            t5=False
            break
        else:
            t5=True

        # Now for checking disimilarity between old and new values of H....

    temp6=H_new-H_old
    for k in temp6.flat:
        if abs(k)>difference_parameter:
            t6=False
            break
        else:
            t6=True

    if t1&t2&t3&t4&t5&t6:
        return True
    else:
        return False

        
            
    
def optimise_p_function(d,n,alpha,lamda,difference_parameter):
    V1, V2, V3, V4, V5=get_v_function(n)
    tpl1=V1.shape
    tpl2=V2.shape
    tpl3=V3.shape
    tpl4=V4.shape
    tpl5=V5.shape

    H_old=initialise_h_function(d,n)

    
    P1_old=initialise_p_function(tpl1[0],d)
    P2_old=initialise_p_function(tpl2[0],d)
    P3_old=initialise_p_function(tpl3[0],d)
    P4_old=initialise_p_function(tpl4[0],d)
    P5_old=initialise_p_function(tpl5[0],d)

    P1_new=update_p_function(P1_old,V1,H_old,alpha,lamda)
    P2_new=update_p_function(P2_old,V2,H_old,alpha,lamda)
    P3_new=update_p_function(P3_old,V3,H_old,alpha,lamda)
    P4_new=update_p_function(P4_old,V4,H_old,alpha,lamda)
    P5_new=update_p_function(P5_old,V5,H_old,alpha,lamda)

    H_new=update_h_function(H_old,P1_old,P2_old,P3_old,P4_old,P5_old,V1,V2,V3,V4,V5,alpha,lamda)
    
    
    # If this condition is true then update the matrix else exit since the matrix is updated
    # According to the gradient descent method, all the Parameter Pi and H should be updated together......
    
    while(check_ph_function(P1_old,P1_new,P2_old,P2_new,P3_old,P3_new,P4_old,P4_new,P5_old,P5_new,H_old,H_new,difference_parameter)==False):
        P1_old=P1_new
        P1_new=update_p_function(P1_old,V1,H_old,alpha,lamda)

        P2_old=P2_new
        P2_new=update_p_function(P2_old,V2,H_old,alpha,lamda)

        P3_old=P3_new
        P3_new=update_p_function(P3_old,V3,H_old,alpha,lamda)

        P4_old=P4_new
        P4_new=update_p_function(P4_old,V4,H_old,alpha,lamda)

        P5_old=P5_new
        P5_new=update_p_function(P5_old,V5,H_old,alpha,lamda)

        H_old=H_new
        H_new=update_h_function(H_old,P1_old,P2_old,P3_old,P4_old,P5_old,V1,V2,V3,V4,V5,alpha,lamda)

    """
    print "P1 Matrix"
    print P1_new.shape
    print P1_new

    print "P2 Matrix"
    print P2_new.shape
    print P2_new

    print "P3 Matrix"
    print P3_new.shape
    print P3_new

    print "P4 Matrix"
    print P4_new.shape
    print P4_new

    print "P5 Matrix"
    print P5_new.shape
    print P5_new

    print "H Matrix"
    print H_new.shape
    print H_new
    """
    print "h calculated"

    return (P1_new,P2_new,P3_new,P4_new,P5_new,H_new)
    
"""
def main():
    optimise_p_function(10,50,0.002,0.003,0.0001)
    

if __name__=='__main__':
    main()
"""
