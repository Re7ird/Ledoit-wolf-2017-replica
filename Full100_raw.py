##Package-----------------------------------

import numpy as np
import pandas as pd
import nonlinshrink as nls

def showmatrixinfo(matrix):
    print("Matrix shape: ",np.shape(matrix))
    print(matrix)
    return
    


##-------Test Function---------------------##

def test_result(omega,method):
    
    ### Use estimated covariance matrix to calculate w
    Omega= np.matrix(omega)
    w=(Omega.I@np.ones((100,1)))/(np.ones((1,100))@Omega.I@np.ones((100,1)))
    print("\n"+method+":")
    print("weight_vector:")
    print(np.shape(w))
    
    ###########calculate payoff and excess payoff(ri-rf)
    
    payoff=w.T@yt_test*100
    
    exc_payoff=payoff-rf_daily
    print("Return: "+method)
    print(type(exc_payoff.tolist()[0]))
    print("lenth:")
    print(len(exc_payoff.tolist()[0]))
    
    
    return exc_payoff.tolist()[0]

##-------------Covariance Estimation----------------------##

def run_1N():
    N=100
    omega=np.identity(100)

    return1N=test_result(omega,"1/N")
    return return1N

def run_Samp():
    ## ---------Sample Covariance Estimation
    omega=np.cov(yt_est)
    print("Matrix shape:")
    print(np.shape(omega))

    returnSamp=test_result(omega,"Samp")
    return returnSamp
    
def run_Lin():
    ##--------Lin(Ledoit-Wolf 2004 methpd) shrinkage Estimation
    from sklearn.covariance import ledoit_wolf

    omega=ledoit_wolf(yt_est.T)[0]

    returnLin=test_result(omega,"Lin")
    return returnLin

##-------NonLin shrinkage Estimation--------------

def run_NonLin():
    import nonlinshrink as nls

    omega=nls.shrink_cov(yt_est.T)
    print(np.shape(omega))

    returnNonLin=test_result(omega,"NonLin")
    return returnNonLin

##--------Single factor estimation---------------------##

def run_SF():
        ##   Generate Factor
    equalw=np.array([[0.01]*100])
    print("equal portfolio: ",np.shape(equalw))
    factor=equalw@yt_est
    print("Factor matrix shape: ",np.shape(factor))


    ##estimate SigmaF
    var_f=np.var(factor,ddof=1)      #variance of factor,

    np.savetxt("SF_factor.csv",factor,delimiter=",")

    ##compute cov(Ri,Rf),the covariance of stocks and factor
    var_if=np.cov(yt_est,factor)[-1,:-1]
    var_if=np.matrix(var_if)          #convert to 1X100 matrix


    SigmaSF=var_if.T*var_if/var_f  
    for i in range(100):
        SigmaSF[i,i]=np.cov(yt_est)[i,i]


    showmatrixinfo(SigmaSF)    ####Seems correct

    returnSF=test_result(SigmaSF,"SF")
    return returnSF,SigmaSF

##-----------------FAMA FRENCH estimation------------------------

def run_FF():
    #####First, Generate 3-factors array.


    from sklearn.linear_model import LinearRegression

    LG=LinearRegression()

    LG.fit(FFfactors,yt_est.T)      ##FFfactor matirx is a (250,3) matrix!
    betas=LG.coef_
    print("Beta matrix: ",np.shape(betas))

    var_ff=np.cov(FFfactors.T)      ##Covariance of FAMA FRENCH 3 Factor model.

    SigmaF=betas@var_ff@betas.T

    ###As same as SF, the diagonal need add residual,or replace by var(Ri)

    for i in range(100):
        SigmaF[i,i]=np.cov(yt_est)[i,i]

    print("SigmaF: ")
    showmatrixinfo(SigmaF)

    returnFF=test_result(SigmaF,"FF")
    return returnFF


def run_POET():

#---------POET estimation-----------------#
    from sklearn.decomposition import PCA
    
    print("generating components:")
    pca = PCA(n_components=5, copy=True)
    pca.fit(yt_est)
    factors=pca.components_
    print('PCA Componets:\n', pca.components_)
    print(np.shape(pca.components_))
    print('\nEigenvalues:', pca.explained_variance_)
    print('Variance explaination:', pca.explained_variance_ratio_)
    print("Add up: ",np.sum(pca.explained_variance_ratio_))


    ##Regression:
    from sklearn.linear_model import LinearRegression

    print("\nRegression:")
    LG2=LinearRegression()

    LG2.fit(factors.T,yt_est.T)
    betas=LG2.coef_
    print("Beta matrix: ",np.shape(betas))

    var_fs=np.cov(factors)

    SigmaF=betas@var_fs@betas.T

    ###As same as SF, the diagonal need add residual,or replace by var(Ri)

    for i in range(100):
        SigmaF[i,i]=np.cov(yt_est)[i,i]

    showmatrixinfo(SigmaF)

    returnPOET=test_result(SigmaF,"POET")
    return returnPOET

def run_NLSF(SigmaSF):
    ## NL-SF
    eigenvalue, eigenvectors = np.linalg.eig(SigmaSF)
    print(np.shape(eigenvalue))
    print(np.shape(eigenvectors))

    print(np.allclose(SigmaSF,eigenvectors@np.diag(eigenvalue)@eigenvectors.T))

    diag=np.identity(100)
    diag2=np.zeros((100,100))
    for i in range(100):
        diag[i,i]=pow(eigenvalue[i],-1/2)
        diag2[i,i]=pow(eigenvalue[i],1/2)
    ##Generate Yt x Sigma_SF to the power of -1/2
    SigmaSF2=eigenvectors@diag@eigenvectors.T  ##(1/2)
    SigmaSF3=eigenvectors@diag2@eigenvectors.T  ##(-1/2)
    SigmaC_hat=nls.shrink_cov(yt_est.T@SigmaSF2)

    #Reincorporating the structure.
    SigmaNLSF=SigmaSF3@SigmaC_hat@SigmaSF3
    returnNLSF=test_result(SigmaNLSF,"NL-SF")
    return returnNLSF


##------------Initialize-------------------#

return1N=[]
returnSamp=[]
returnLin=[]
returnNonLin=[]
returnSF=[]
returnFF=[]
returnPOET=[]
returnNLSF=[]

def run(fulldata):
    yt_est=fulldata[0]
    yt_test=fulldata[1]
    FFdata=fulldata[2]
    
    return1N.extend(run_1N())
    returnSamp.extend(run_Samp())
    returnLin.extend(run_Lin())
    returnNonLin.extend(run_NonLin())
    returnSF.extend(run_SF()[0])
    returnFF.extend(run_FF())
    returnPOET.extend(run_POET())
    returnNLSF.extend(run_NLSF(run_SF()[1]))
    
    return


#####--------------Running test----------------------------------#

##------Data import-------------------------
FF_u1 = pd.read_csv('data/FF_universe1.csv')
stocks_u1=pd.read_csv('data/stocks100_u1.csv')

FF_u2= pd.read_csv('data/FF_universe2.csv')
stocks_u2=pd.read_csv('data/stocks100_u2.csv')

#-------------Iteration---------------------

FF_u=FF_u1
stocks_u=stocks_u1

for i in range(230):
    FFdata = FF_u.iloc[i*21:271+i*21,]

    stocksdata=stocks_u.iloc[i*21:271+i*21,1:101].to_numpy()
    yt_est=stocksdata[:250,].T
    yt_test=stocksdata[250:,].T

    index=["Mkt-RF","SMB","HML"]
    FFfactors=FFdata.loc[:,index].iloc[:250,].to_numpy()
    rf_rate=FFdata.loc[:,"RF"].iloc[250:,].to_numpy()
    rf_daily=rf_rate/250                          

    data=[yt_est,yt_test,FFdata]
    run(data)
    print("Tetst "+str(i+1)+" complete!")


print("Universe 1 Complete!")

FF_u=FF_u2
stocks_u=stocks_u2

for i in range(230):
    FFdata = FF_u.iloc[i*21:271+i*21,]

    stocksdata=stocks_u.iloc[i*21:271+i*21,1:101].to_numpy()
    yt_est=stocksdata[:250,].T
    yt_test=stocksdata[250:,].T

    index=["Mkt-RF","SMB","HML"]
    FFfactors=FFdata.loc[:,index].iloc[:250,].to_numpy()
    rf_rate=FFdata.loc[:,"RF"].iloc[250:,].to_numpy()
    rf_daily=rf_rate/250                          

    data=[yt_est,yt_test,FFdata]
    run(data)
    print("Tetst "+str(i+231)+" complete!")


print("Universe 2 Complete!")





    
ret_mat=np.array([return1N,returnSamp,returnLin,returnNonLin,returnSF,returnFF,returnPOET,returnNLSF])

print("Return maxtrix:")
print(np.shape(ret_mat))


##----------------Calculating and plotting results------------------##

AV=np.mean(ret_mat,axis=1)*250
AV=AV.T.tolist()
    
SD=np.std(ret_mat,axis=1)*pow(250,.5)
SD=SD.T.tolist()

SR=[]
for i in range(len(SD)):
    SR.append(AV[i]/SD[i])

def prtb():
    print("Table 1")
    print("Performance measures for various estimators of the GMV portfolio")
    print("{}".format('''Period: January 19, 1973 to December 31, 2011" '''))
    print("\t1/N \tSample \tLin \tNolin \tSF \tFF \tPOET \tNL-SF")
    print("{:-^70}".format(""))
    print("{:^70}".format("N=100"))
    print("{:-^70}".format(""))
    print("AV \t{:.2f} \t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(
        AV[0],AV[1],AV[2],AV[3],AV[4],AV[5],AV[6],AV[7])
    )
    print("SD \t{:.2f} \t{:.2f}\t{:.2f}\t{:.2f}\t{:.1f}\t{:.2f}\t{:.2f}\t{:.2f}".format(
        SD[0],SD[1],SD[2],SD[3],SD[4],SD[5],SD[6],SD[7]))
    print("SR \t{:.2f} \t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(
        SR[0],SR[1],SR[2],SR[3],SR[4],SR[5],SR[6],SR[7]))
        
    return


prtb()