import unittest
import numpy as np
import scipy.sparse, pickle
from sklearn.metrics import pairwise_distances
import datetime
from lmnn import LMNN
from lmnn import python_LMNN
import final_model
import final_model1
import final_model_test_inf
from sklearn.svm import SVC
from sklearn.metrics import *
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier as RFC

n_train=5976#4482#2988#1000#5976  #8964#5976#14940 #17928
n_test=4024#3018#2012#600#4024   #6036#4024#10060 #12072
#C_list=[1e2,1e3,1e4,1e5]
C_list = [1e4]
#gamma_list=[.01,.001,.0001]
gamma_list=[0.01]
C_gamma_list=[]
for i in C_list:
    for j in gamma_list:
        C_gamma_list.append((i,j))


def class_separation(X, labels):
  unique_labels, label_inds = np.unique(labels, return_inverse=True)
  ratio = 0
  for li in xrange(len(unique_labels)):
    Xc = X[label_inds==li]
    Xnc = X[label_inds!=li]
    ratio += pairwise_distances(Xc).mean() / pairwise_distances(Xc,Xnc).mean()
  return ratio / len(unique_labels)
    


def test_data(d,alpha,lamda):
    k = 10
    #np.random.seed(1234)
    p_h=final_model.optimise_p_function(d,n_train,alpha,lamda,0.0001) # take training points , H training points obtained
    points=p_h[5]
    P = p_h[:5]  #All P  matrices obtained
    points=points.transpose()
    points=np.array(points)
    #f=open(r'/home/raga/project_2/nus_object/Train_labels.txt') #take training labels
    #f = open(r'/home/raga/project_2/LITE/Train_labels_lite_new.txt')
    f = open(r'/home/raga/project_2/SCENE/Train_labels_scene_new.txt')
    labels=f.readlines()[:n_train]
    labels=[int(i.strip()) for i in labels]
    labels=np.array(labels)
    f.close()
    
    #f=open(r'/home/raga/project_2/nus_object/Test_labels.txt') #take test labels
    #f=open(r'/home/raga/project_2/LITE/Test_labels_lite_new.txt')
    f = open(r'/home/raga/project_2/SCENE/Test_labels_scene_new.txt')
    test_labels=f.readlines()[:n_test]
    test_labels=[int(i.strip()) for i in test_labels]
    test_labels=np.array(test_labels)
    f.close()

    test_points=final_model1.optimise_p_function(d,n_test,alpha,lamda,0.0001)[5] #take train labels , H test_points obtained
    #test_points = final_model_test_inf.optimise_h_function(P,d,n_test,alpha,lamda,0.0001)
    test_points=test_points.transpose()
    #test_points=np.array(test_points)
    #return (points,labels,test_points,test_labels)

    # Test both impls, if available.
    '''
    for LMNN_cls in set((LMNN, python_LMNN)):
      lmnn = LMNN_cls(k=k, learn_rate=1e-6,max_iter=150)
      lmnn.fit(points, labels, verbose=False)
      print "lmnn has been fitted"
      #csep = class_separation(lmnn.transform(), labels)
      #self.assertLess(csep, 0.25)
      points=lmnn.transform(points)
      test_points=lmnn.transform(test_points)
      #return (points,labels,test_points,test_labels)
      return (points,labels,test_points)
    '''
    return (points, labels, test_points,test_labels)



if __name__ == '__main__':

    #alpha_list=[0.001,0.0003]
    alpha_list = [0.0003]
   # lamda_list=[0.5,0.01,0.001]
    lamda_list = [0.01]
    alpha_lamda_list=[]

    for i in alpha_list:
        for j in lamda_list:
            alpha_lamda_list.append((i,j))
    f= open('final_results_lite.txt','w')
    for d in range(50,60,10):
        print '####################################################################################'
        time_now = datetime.datetime.now()
        f.write('Time start\n')
        f.write(str(time_now))
        f.write('\n')
        print 'Values for d=', d
        for al in alpha_lamda_list:
            d_a_l_val = '\n'+'New Combination'+' '+str(d)+' '+str(al[0])+ ' '+str(al[1])+'\n'
            f.write(d_a_l_val)
            print '*******************************************************************************'
            print 'Values for alpha and lamda', al[0],al[1]
    
            m=test_data(d,al[0],al[1])
            f.write('H obtained \n')
            f.write('\n')
            time_now_h = datetime.datetime.now()
            f.write(str(time_now_h))
            f.write('\n')
            f.write('SVM Classification start \n')
            f.write('\n')
	    dump_list = {}
            for j in C_gamma_list:
                clf =SVC(C=j[0],gamma=j[1])
                clf = clf.fit(m[0],m[1])
                pred = clf.predict(m[2])
		f.write(str(j[0])+ ' '+ str(j[1])+'\n')
	        f.write( ' '.join( [str(i) for i in pred] ))	
		f.write('\n')
                f.write('SVM done \n')
                time_now_svm = datetime.datetime.now()
                f.write(str(time_now_svm))
                
                print 'm3_len', len(m[3]), 'pred_len', len(pred)
                acc =  accuracy_score(m[3],pred)
                print 'Accuracy: %.3f   C:%10d'%(acc,j[0]),'gamma:',j[1]
                #pickle.dump(m[0],open('h_old_scene15k_file.p','wb'))            

		
	
		print '*********KNN***********'
                clf =KNeighborsClassifier(n_neighbors=10)
	        clf = clf.fit(m[0],m[1])
		pred = clf.predict(m[2])
		acc =  accuracy_score(m[3],pred)
		print 'Knn acc', acc
	'''	   
		print '**********RFC*********'
		clf = RFC( n_estimators=20 )
		clf = clf.fit( m[0],m[1] )
		pred = clf.predict( m[2] )
		acc = accuracy_score( m[3],pred )
		print acc
	'''
    f.close()
