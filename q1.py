import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from prettytable import PrettyTable
import matplotlib.pyplot as plt
t = PrettyTable(['degree', 'bias','bias^2','variance'])

with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

#print (data)

data=np.array(data)
y=[]
for i in range (0,5000):
    y.append(i)


np.random.shuffle(data)

y=np.array(y)

#print (data.shape)
#print (data)

datt = np.transpose(data)
X=datt[0]
y=datt[1]

#for i in range (0,100):
#    print (X[i],"    ",y[i])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


#print (X_train)
#print (y_train)
xtrainlist=np.split(X_train,10)
ytrainlist=np.split(y_train,10)
#print (xtrainlist)
#print ()
#print ()
#print ()
#print (X_train)


polyres=[]
i=2
degree=[]
variance=[]
allbias=[]
allbiassq=[]
biaslist=[]
varlist=[]
deglist=[]
meansqlist=[]
for i in range(1,10) :
    polyres=[]
    for ind in range(0,10):
        xsub=xtrainlist[ind]
#    print(xsub)
        xsub=xsub.reshape(-1,1)
        X_test=X_test.reshape(-1,1)
#    print(xsub)
        poly = PolynomialFeatures(i)
        k=poly.fit_transform(xsub)
        xtest=poly.fit_transform(X_test)
        ysub=ytrainlist[ind]
#    print(ysub)
        reg = LinearRegression().fit(k, ysub)
#        print(reg.score(k,ysub))
#        print((reg.coef_))
        resu=reg.predict(xtest)
        polyres.append(resu)


    row=[]
    row.append(i)
    polyres=np.array(polyres) 
#    print (polyres.shape)
    
    #calculate bias 
    sump = np.sum(polyres,axis=0)
    sump = np.divide(sump,10)
#    print(y_test.shape,"  ",X_test.shape," ",sump.shape)
    plt.scatter(X_test, sump,label="predicted output")
    plt.scatter(X_test,y_test,label="actual output")
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.legend()
    plt.show()
    sqbias = np.square(sump-y_test)
#    print(sump.shape)
#    print(bias.shape)
    finsqbias=np.mean(sqbias)
#--    print("bias= ",finbias)
    finbias=np.sqrt(finsqbias)
    row.append(finbias)
    row.append(finsqbias)
    #calculate variance 
    sump = np.sum(polyres,axis=0)
    sump = np.divide(sump,10)
    meansq = np.square(sump)

    sqr = np.square(polyres)
    sqrsum = np.sum(sqr,axis=0)
    sqrmean = np.divide(sqrsum,10)

    var= sqrmean - meansq
    finvar=np.mean(var)

    biaslist.append(finbias)
    varlist.append(finvar)
    deglist.append(i)
    meansqlist.append(finsqbias+finvar)

#    print("var= ",finvar)
    row.append(finvar)
    t.add_row(row)

plt.plot(deglist, biaslist,label="bias")
plt.plot(deglist,varlist,label="variance")
plt.plot(deglist,meansqlist,label="error")
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.legend()
plt.show()


print(t)
print(meansqlist)




#        print(y_test)

#        reg = LinearRegression().fit(k, y)



#print ("lololololololololol        ",polyres)


