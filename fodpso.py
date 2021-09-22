import numpy as st
from numpy import zeros
from numpy import ones
import inspect
import sympy as sy
import random
import time




N_PAR=8;
def fodpso(func=None,xmin=-100*ones((1,N_PAR)),xmax=100*ones((1,N_PAR)),type='min',population=20,nswarms=5,iterations=500,alfa=0.6):
    start = time.time();
    fun=lambda x1,x2 :eval(func) ;
    print(func);
    



    N_PAR=len(inspect.getfullargspec(fun)[0]);
    print(N_PAR);



    N_GER = iterations;

    PHI1 = 1.5;
    PHI2 = 1.5;
    W = 1;

    MIN_SWARMS=round(nswarms/2);
    N_SWARMS=nswarms;
    MAX_SWARMS=nswarms*2;

    SWARMS=st.zeros((MAX_SWARMS,1));  #SWARMS=st.zeros(MAX_SWARMS);

    for i in range(0,MAX_SWARMS-1):
        if (i<=N_SWARMS):
             SWARMS[i]=1;



    N_MIN=round(population/2);
    N_Init=population;
    N_MAX=population*2;

    N=N_Init*ones((MAX_SWARMS,1)); #N=N_Init*st.ones(MAX_SWARMS,1);

    SCmax=round(N_GER/20);
    SC=st.zeros([MAX_SWARMS,1]); #SC=st.zeros(MAX_SWARMS,1);


    X_MAX=xmax;
    X_MIN=xmin;
    #print(X_MIN.shape)

    vmin=round(-(max(xmax)-min(xmin))/(population));
    vmax=round((max(xmax)-min(xmin))/(population));

    if type == 'min' :
         gbestvalue = 1000000*st.ones((MAX_SWARMS,1)); #gbestvalue = 1000000*st.ones(MAX_SWARMS,1);
    elif type == 'max':
        gbestvalue = -1000000*st.ones((MAX_SWARMS,1)); #gbestvalue = -1000000*st.ones(MAX_SWARMS,1);


    gbestvalueaux=gbestvalue;

    fit = st.zeros((MAX_SWARMS,int(st.max(N).item()),1));
    x=zeros((MAX_SWARMS,int(st.max(N).item()),N_PAR,1));
    #print(MAX_SWARMS);
    v=st.zeros((MAX_SWARMS,int(st.max(N).item()),N_PAR,1));
    #print(v.shape);
    vbef1=st.zeros((MAX_SWARMS,int(st.max(N).item()),N_PAR,1));
    vbef2=st.zeros((MAX_SWARMS,int(st.max(N).item()),N_PAR,1));
    vbef3=st.zeros((MAX_SWARMS,int(st.max(N).item()),N_PAR,1));

    #print(x.shape);
    for i in range(0,N_SWARMS):
        #print("H");
        x[i] = zeros((int(N[i].item()), N_PAR,1)) ;
        #x(i).lvalue=mlrose.CustomFitness(N[i],N_PAR);
        #print(x);
        #print(st.size(x));
        fit[i,0:int(N[i].item()),0]=0; #maybe replace 1 with 0
        #print(fit[i]);
        v[i]=zeros((int(N[i].item()), N_PAR,1));
        vbef1[i]=zeros((int(N[i].item()), N_PAR,1));
        vbef2[i]=zeros((int(N[i].item()), N_PAR,1));
        vbef3[i]=zeros((int(N[i].item()), N_PAR,1));


    xaux=x;
    fitBest=fit;
    vaux=v;
    vbef1aux=vbef1;
    vbef2aux=vbef2;
    vbef3aux=vbef3;

    xBest=st.zeros((MAX_SWARMS,int(st.max(N).item()),N_PAR,1));
    gBest=st.zeros((MAX_SWARMS,int(st.max(N).item()),N_PAR,1));

    nger=1;

    fitBefore=fit;

    Nkill=st.zeros((MAX_SWARMS))#,int(st.max(N).item()),N_PAR,1));

    xBEST=st.zeros((MAX_SWARMS,int(st.max(N).item()),N_PAR,1));
    fitting=st.zeros((N_GER));
    gBestT=st.zeros((N_GER,N_PAR,1));

    for i in range(0,N_SWARMS):
        #print("H");
        xaux[i,0:int(N[i].item()),0:N_PAR,0] = inicializaSwarm(N[i], N_PAR, X_MIN, X_MAX);
        x=xaux;
        for j in range(0,int(N[i].item())):
            fit[i][j]=0;

            if N_PAR == 1:
                fit[i][j]=fun(x[i][j,0]);
            elif N_PAR == 2 :
                fit[i][j]=fun(x[i][j,0],x[i][j,1]);
            elif N_PAR == 3 :
                fit[i][j]=fun(x[i][j,0],x[i][j,1],x[i][j,2]);
            elif N_PAR == 4 :
                fit[i][j]=fun(x[i][j,0],x[i][j,1],x[i][j,2],x[i][j,3]);
            elif N_PAR == 5 :
                fit[i][j]=fun(x[i][j,0],x[i][j,1],x[i][j,2],x[i][j,3],x[i][j,4]);
            elif N_PAR == 6 :
                fit[i][j]=fun(x[i][j,0],x[i][j,1],x[i][j,2],x[i][j,3],x[i][j,4],x[i][j,5]);
            elif N_PAR == 7 :
                fit[i][j]=fun(x[i][j,0],x[i][j,1],x[i][j,2],x[i][j,3],x[i][j,4],x[i][j,5],x[i][j,6]);
            elif N_PAR == 8:
                fit[i][j]=fun(x[i][j,0],x[i][j,1],x[i][j,2],x[i][j,3],x[i][j,4],x[i][j,5],x[i][j,6],x[i][j,7]);
            elif N_PAR == 9:
                fit[i][j]=fun(x[i][j,0],x[i][j,1],x[i][j,2],x[i][j,3],x[i][j,4],x[i][j,5],x[i][j,6],x[i][j,7],x[i][j,8]);
            else:
                print("Incorrect number of variables");
            fitBest[i][j]=fit[i][j];

        if type=='min' :
            [a,b]=[min(fit[i]),st.argmin(fit[i])];
        elif type=='max' :
            [a,b]=[max(fit[i]),st.argmax(fit[i])];


        gBest[i]=x[i,b,:];

        gbestvalue[i] = fit[i][b];
        gbestvalueaux[i]=gbestvalue[i];
        xBest[i,0:int(N[i].item()),0:N_PAR,0] = inicializaSwarm(N[i], N_PAR, X_MIN, X_MAX);


    fitaux=fit;
    xBestaux=xBest;
    gBestaux=gBest;
    fitBestaux=fitBest;

    while nger<=N_GER :
        #print("H");
        for i in range(0,MAX_SWARMS):
            #print("H");
            xBestaux[i]=xBest[i];
            vbef1aux[i]=vbef1[i];
            vbef2aux[i]=vbef2[i];
            vbef3aux[i]=vbef3[i];

            randnum1 = st.random.rand(int(N[i].item()), N_PAR,1);
            #print(randnum1);
            randnum2 = st.random.rand(int(N[i].item()), N_PAR,1);

            gaux=ones((int(N[i].item()),1));
            #print(gaux.reshape(gaux.shape[0],1,1).shape)
            #print(N[i]);
            vaux[i,0:int(N[i].item()),:] = W*(alfa*v[i,0:int(N[i].item()),:] + (1/2)*alfa*(1-alfa)*vbef1[i,0:int(N[i].item()),:] + (1/6)*alfa*(1-alfa)*(2-alfa)*vbef2[i,0:int(N[i].item()),:] + (1/24)*alfa*(1-alfa)*(2-alfa)*(3-alfa)*vbef3[i,0:int(N[i].item()),:]) + randnum1 *(PHI1 *(xBest[i,0:int(N[i].item()),:]-x[i,0:int(N[i].item()),:])) + randnum2 *(PHI2 *(gaux.reshape(gaux.shape[0],1,1)*gBest[i,0:int(N[i].item()),:]-x[i,0:int(N[i].item()),:]));
            #print(temp.shape);
            vaux[i] = ( (vaux[i] <= vmin)*vmin ) + ( (vaux[i] > vmin)*vaux[i] );
            vaux[i] = ( (vaux[i] >= vmax)*vmax ) + ( (vaux[i] < vmax)*vaux[i] );

            vbef3aux[i]=vbef2[i];
            vbef2aux[i]=vbef1[i];
            vbef1aux[i]=v[i];
            #print(xaux[i].shape)
            #print(x[i].shape)
            #print(vaux[i].shape)
            xaux[i,0:int(N[i].item()),:,0] = x[i,0:int(N[i].item()),:,0]+vaux[i,0:int(N[i].item()),:,0];

            for j in range(0,int(N[i].item())) :
                for k in range(0,N_PAR):
                    if xaux[i][j][k] < X_MIN[k]:
                        xaux[i,j,k] = X_MIN[k];
                    elif xaux[i,j,k] > X_MAX[k] :
                        xaux[i,j,k] = X_MAX[k];

            for j in range(0,int(N[i].item())):
                fitaux[i][j]=0;

                if N_PAR == 1:
                    fitaux[i][j]=fun(x[i][j,0]);
                elif N_PAR == 2 :
                    fitaux[i][j]=fun(x[i,j,0],x[i,j,1]);
                    #print(st.min(fitaux));
                    print(x[i][j,0]);
                elif N_PAR == 3 :
                    fitaux[i][j]=fun(x[i][j,0],x[i][j,1],x[i][j,2]);
                elif N_PAR == 4 :
                    fitaux[i][j]=fun(x[i][j,0],x[i][j,1],x[i][j,2],x[i][j,3]);
                elif N_PAR == 5 :
                    fitaux[i][j]=fun(x[i][j,0],x[i][j,1],x[i][j,2],x[i][j,3],x[i][j,4]);
                elif N_PAR == 6 :
                    fitaux[i][j]=fun(x[i][j,0],x[i][j,1],x[i][j,2],x[i][j,3],x[i][j,4],x[i][j,5]);
                elif N_PAR == 7 :
                    fitaux[i][j]=fun(x[i][j,0],x[i][j,1],x[i][j,2],x[i][j,3],x[i][j,4],x[i][j,5],x[i][j,6]);
                elif N_PAR == 8:
                    fitaux[i][j]=fun(x[i][j,0],x[i][j,1],x[i][j,2],x[i][j,3],x[i][j,4],x[i][j,5],x[i][j,6],x[i][j,7]);
                elif N_PAR == 9:
                    fitaux[i][j]=fun(x[i][j,0],x[i][j,1],x[i][j,2],x[i][j,3],x[i][j,4],x[i][j,5],x[i][j,6],x[i][j,7],x[i][j,8]);
                else:
                    print("Incorrect number of variables");


                if type=='min':
                    if fitaux[i][j] < fitBestaux[i][j] :
                        fitBestaux[i][j] = fitaux[i][j];
                        xBestaux[i,j,:] = xaux[i,j,:];
                elif type=='max':
                    if fitaux[i][j] > fitBestaux[i][j]:
                        fitBestaux[i][j] = fitaux[i][j];
                        xBestaux[i,j,:] = xaux[i,j,:];

            #print(fitaux);
            if type=='min':
                [a,b]=[min(fit[i]),st.argmin(fit[i])];
                if (a < gbestvalueaux[i]):
                    gBestaux[i]=xaux[i,b,:];
                    gbestvalueaux[i]=fitaux[i,b,:];
                    #print(fitaux[i,b,:]);


                [fitting[nger-1],indexM]=[min(gbestvalueaux),st.argmin(gbestvalueaux)];#min(gbestvalueaux);
                #print(X_MIN)
                gBestT[nger-1,0:N_PAR] = gBestaux[indexM,0];
            elif type=='max' :
                [a,b]=[max(fit[i]),st.argmax(fit[i])];
                if (a > gbestvalueaux[i]) :
                    gBestaux[i]=xaux[i,b,:];
                    gbestvalueaux[i] = fitaux[i,b,:];


                [fitting[nger-1],indexM]=[max(gbestvalueaux),st.argmax(gbestvalueaux)];#max(gbestvalueaux);
                gBestT[nger-1,0:N_PAR] = gBest[indexM,0];


            if (nger>1) :
                if gbestvalueaux[i]>=gbestvalue[i] :
                    SC[i]=SC[i]+1;
                    if (SC[i]==SCmax) :
                        if N[i]>N_MIN :

                            if type=='min' :
                                [a,b]=[min(fitaux[i]),st.argmin(fitaux[i])];#max(fitaux[i]);
                            elif type=='max' :
                                [a,b]=[max(fitaux[i]),st.argmax(fitaux[i])];#min(fitaux[i]);


                            if b==1 :
                                xaux[i,0:int(N[i].item())-1,:]=xaux[i,1:int(N[i].item()),:];
                                xBestaux[i,0:int(N[i].item())-1,:]=xBestaux[i,1:int(N[i].item()),:];
                                vaux[i,0:int(N[i].item())-1,:]=vaux[i,1:int(N[i].item()),:];
                                vbef1aux[i,0:int(N[i].item())-1,:]=vbef1aux[i,1:int(N[i].item()),:];
                                vbef2aux[i,0:int(N[i].item())-1,:]=vbef2aux[i,1:int(N[i].item()),:];
                                vbef3aux[i,0:int(N[i].item())-1,:]=vbef3aux[i,1:int(N[i].item()),:];
    #                         end

                            if b==int(N[i].item()):
                                xaux[i,0:b-1,:]=xaux[i,0:b-1,:];
                                xBestaux[i,0:b-1,:]=xBestaux[i,0:b-1,:];
                                vaux[i,0:b-1,:]=vaux[i,0:b-1,:];
                                vbef1aux[i,0:b-1,:]=vbef1aux[i,0:b-1,:];
                                vbef2aux[i,0:b-1,:]=vbef2aux[i,0:b-1,:];
                                vbef3aux[i,0:b-1,:]=vbef3aux[i,0:b-1,:];
    #                         end
    #
                            if (b>1 and b<int(N[i].item())) :
                                xaux[i,0:int(N[i].item())-1,:]=st.vstack((xaux[i][0:(b-1),:],xaux[i][(b):int(N[i].item()),:]));
                                xBestaux[i,0:int(N[i].item())-1,:]=st.vstack((xBestaux[i][0:(b-1),:],xBestaux[i][(b):int(N[i].item()),:]));
                                vaux[i,0:int(N[i].item())-1,:]=st.vstack((vaux[i][0:(b-1),:],vaux[i][(b):int(N[i].item()),:]));
                                vbef1aux[i,0:int(N[i].item())-1,:]=st.vstack((vbef1aux[i][0:(b-1),:],vbef1aux[i][(b):int(N[i].item()),:]));
                                vbef2aux[i,0:int(N[i].item())-1,:]=st.vstack((vbef2aux[i][0:(b-1),:],vbef2aux[i][(b):int(N[i].item()),:]));
                                vbef3aux[i,0:int(N[i].item())-1,:]=st.vstack((vbef3aux[i][0:(b-1),:],vbef3aux[i][(b):int(N[i].item()),:]));
    #                         end

                            N[i]=N[i]-1;
                            Nkill[i]=Nkill[i]+1;
                            SC[i]=st.fix(SCmax*(1-1/(Nkill[i]+1)));
                        else :
                            if (N_SWARMS>MIN_SWARMS) :
                                SWARMS[i]=0;
                                N_SWARMS=N_SWARMS-1;
                                SC[i]=0;
                                N[i]=0;
    #                         end
    #                     end
    #                 end
                else :
                    if (Nkill[i]>0) :
                        Nkill[i]=Nkill[i]-1;
    #                 end
                    if (N[i]<N_MAX) :
                        N[i]=N[i]+1;
                        xaux[i][N[i],1:N_PAR]=inicializaSwarm(1, N_PAR, X_MIN, X_MAX);
                        xBestaux[i][N[i],1:N_PAR]=inicializaSwarm(1, N_PAR, X_MIN, X_MAX);
                        vaux[i][N[i],1:N_PAR]=st.zeros(1,N_PAR);
                        vbef1aux[i][N[i],1:N_PAR]=st.zeros(1,N_PAR);
                        vbef2aux[i][N[i],1:N_PAR]=st.zeros(1,N_PAR);
                        vbef3aux[i][N[i],1:N_PAR]=st.zeros(1,N_PAR);
                        fitaux[i][N[i],1]=0;
    #
    #
                        if N_PAR == 1:
                            fitaux[i][N[i],1]=fun(xaux[i][j,1]);
                        elif N_PAR == 2 :
                            fitaux[i][N[i],1]=fun(xaux[i][j,1],xaux[i][j,2]);
                        elif N_PAR == 3 :
                            fitaux[i][N[i],1]=fun(xaux[i][j,1],xaux[i][j,2],xaux[i][j,3]);
                        elif N_PAR == 4 :
                            fitaux[i][N[i],1]=fun(xaux[i][j,1],xaux[i][j,2],xaux[i][j,3],xaux[i][j,4]);
                        elif N_PAR == 5 :
                            fitaux[i][N[i],1]=fun(xaux[i][j,1],xaux[i][j,2],xaux[i][j,3],xaux[i][j,4],xaux[i][j,5]);
                        elif N_PAR == 6 :
                            fitaux[i][N[i],1]=fun(xaux[i][j,1],xaux[i][j,2],xaux[i][j,3],xaux[i][j,4],xaux[i][j,5],xaux[i][j,6]);
                        elif N_PAR == 7 :
                            fitaux[i][N[i],1]=fun(xaux[i][j,1],xaux[i][j,2],xaux[i][j,3],xaux[i][j,4],xaux[i][j,5],xaux[i][j,6],xaux[i][j,7]);
                        elif N_PAR == 8:
                            fitaux[i][N[i],1]=fun(xaux[i][j,1],xaux[i][j,2],xaux[i][j,3],xaux[i][j,4],xaux[i][j,5],xaux[i][j,6],xaux[i][j,7],xaux[i][j,8]);
                        elif N_PAR == 9:
                            fitaux[i][N[i],1]=fun(xaux[i][j,1],xaux[i][j,2],xaux[i][j,3],xaux[i][j,4],xaux[i][j,5],xaux[i][j,6],xaux[i][j,7],xaux[i][j,8],xaux[i][j,9]);
                        else:
                            print("Incorrect number of variables");
    #                     end
    #
                        fitBestaux[i][N[i],1]=fitaux[i][N[i],1];
                        [a,b]=[min(faux[i]),st.argmin(faux[i])];#min(fitaux[i]);
                        gBestaux[i]=xaux[i][b,:];
                        gbestvalueaux[i] = fitaux[i][b];
                    #end
                #end
                #print(Nkill.shape);
                if (Nkill[i]==0)and(N_SWARMS<MAX_SWARMS) :
                    prob=random.random()/N_SWARMS;
                    create_swarm=random.random();
                    if (create_swarm<prob) :
                        SC[i]=0;
                        SWARM_ALIVE=st.argwhere(SWARMS[:,0]==0);
                        #print(SWARM_ALIVE);
                        if (SWARM_ALIVE.shape[0]>0):
                            SWARMS[SWARM_ALIVE[0].item(),0]=1;
                            N_SWARMS=N_SWARMS+1;
                            N[SWARM_ALIVE[0]]=N_Init;
                            #print(SWARM_ALIVE[0]);
                            xaux[SWARM_ALIVE[0].item(),0:int(N[SWARM_ALIVE[0].item()].item()),0:N_PAR,0]=inicializaSwarm(N[SWARM_ALIVE[0].item()], N_PAR, X_MIN, X_MAX);
                            vaux[SWARM_ALIVE[0].item(),0:int(N[SWARM_ALIVE[0].item()].item()),0:N_PAR,0]=st.zeros((int(N[SWARM_ALIVE[0].item()].item()),N_PAR));
                            vbef1aux[SWARM_ALIVE[0].item(),0:int(N[SWARM_ALIVE[0].item()].item()),0:N_PAR,0]=st.zeros((int(N[SWARM_ALIVE[0].item()].item()),N_PAR));
                            vbef2aux[SWARM_ALIVE[0].item(),0:int(N[SWARM_ALIVE[0].item()].item()),0:N_PAR,0]=st.zeros((int(N[SWARM_ALIVE[0].item()].item()),N_PAR));
                            vbef3aux[SWARM_ALIVE[0].item(),0:int(N[SWARM_ALIVE[0].item()].item()),0:N_PAR,0]=st.zeros((int(N[SWARM_ALIVE[0].item()].item()),N_PAR));
                            xBestaux[SWARM_ALIVE[0].item(),0:int(N[SWARM_ALIVE[0].item()].item()),0:N_PAR,0] = inicializaSwarm(N[SWARM_ALIVE[0].item()], N_PAR, X_MIN, X_MAX);
                            for j in range (0,int(N[SWARM_ALIVE[0].item()].item())):
        #
                                fitaux[SWARM_ALIVE[0].item(),j,0]=0;
        #
                                if N_PAR == 1:
                                    fitaux[SWARM_ALIVE[0].item(),j,0]=fun(xaux[SWARM_ALIVE[0].item(),j,0]);
                                elif N_PAR == 2 :
                                    fitaux[SWARM_ALIVE[0].item(),j,0]=fun(xaux[SWARM_ALIVE[0].item(),j,0],xaux[SWARM_ALIVE[0].item(),j,1]);
                                elif N_PAR == 3 :
                                    fitaux[SWARM_ALIVE[0].item(),j,0]=fun(xaux[SWARM_ALIVE[1]][j,1],xaux[SWARM_ALIVE[1]][j,2],xaux[SWARM_ALIVE[1]][j,3]);
                                elif N_PAR == 4 :
                                    fitaux[SWARM_ALIVE[0].item(),j,0]=fun(xaux[SWARM_ALIVE[1]][j,1],xaux[SWARM_ALIVE[1]][j,2],xaux[SWARM_ALIVE[1]][j,3],xaux[SWARM_ALIVE[1]][j,4]);
                                elif N_PAR == 5 :
                                    fitaux[SWARM_ALIVE[0].item(),j,0]=fun(xaux[SWARM_ALIVE[1]][j,1],xaux[SWARM_ALIVE[1]][j,2],xaux[SWARM_ALIVE[1]][j,3],xaux[SWARM_ALIVE[1]][j,4],xaux[SWARM_ALIVE[1]][j,5]);
                                elif N_PAR == 6 :
                                    fitaux[SWARM_ALIVE[0].item(),j,0]=fun(xaux[SWARM_ALIVE[1]][j,1],xaux[SWARM_ALIVE[1]][j,2],xaux[SWARM_ALIVE[1]][j,3],xaux[SWARM_ALIVE[1]][j,4],xaux[SWARM_ALIVE[1]][j,5],xaux[SWARM_ALIVE[1]][j,6]);
                                elif N_PAR == 7 :
                                    fitaux[SWARM_ALIVE[0].item(),j,0]=fun(xaux[SWARM_ALIVE[1]][j,1],xaux[SWARM_ALIVE[1]][j,2],xaux[SWARM_ALIVE[1]][j,3],xaux[SWARM_ALIVE[1]][j,4],xaux[SWARM_ALIVE[1]][j,5],xaux[SWARM_ALIVE[1]][j,6],xaux[SWARM_ALIVE[1]][j,7]);
                                elif N_PAR == 8:
                                    fitaux[SWARM_ALIVE[0].item(),j,0]=fun(xaux[SWARM_ALIVE[1]][j,1],xaux[SWARM_ALIVE[1]][j,2],xaux[SWARM_ALIVE[1]][j,3],xaux[SWARM_ALIVE[1]][j,4],xaux[SWARM_ALIVE[1]][j,5],xaux[SWARM_ALIVE[1]][j,6],xaux[SWARM_ALIVE[1]][j,7],xaux[SWARM_ALIVE[1]][j,8]);
                                elif N_PAR == 9:
                                    fitaux[SWARM_ALIVE[0].item(),j,0]=fun(xaux[SWARM_ALIVE[1]][j,1],xaux[SWARM_ALIVE[1]][j,2],xaux[SWARM_ALIVE[1]][j,3],xaux[SWARM_ALIVE[1]][j,4],xaux[SWARM_ALIVE[1]][j,5],xaux[SWARM_ALIVE[1]][j,6],xaux[SWARM_ALIVE[1]][j,7],xaux[SWARM_ALIVE[1]][j,8],xaux[SWARM_ALIVE[1]][j,9]);
                                else:
                                    print("Incorrect number of variables");
        #                         end
        #
                                fitBestaux[SWARM_ALIVE[0].item(),j,0]=fitaux[SWARM_ALIVE[0].item(),j,0];
        #                     end
                            [a,b]=[min(fitaux[SWARM_ALIVE[0].item()]),st.argmin(fitaux[SWARM_ALIVE[0].item()])];#min(fitaux[SWARM_ALIVE[0].item()]);
                            gBestaux[SWARM_ALIVE[0].item()]=xaux[SWARM_ALIVE[0].item(),b,:];
                            gbestvalueaux[SWARM_ALIVE[0].item()] = fitaux[SWARM_ALIVE[0].item(),b];
        #
                            SC[SWARM_ALIVE[0].item()]=0;
        #
                            fit[SWARM_ALIVE[0].item()]=fitaux[SWARM_ALIVE[0].item()];
                            x[SWARM_ALIVE[0].item()]=xaux[SWARM_ALIVE[0].item()];
                            v[SWARM_ALIVE[0].item()]=vaux[SWARM_ALIVE[0].item()];
                            vbef1[SWARM_ALIVE[0].item()]=vbef1aux[SWARM_ALIVE[0].item()];
                            vbef2[SWARM_ALIVE[0].item()]=vbef2aux[SWARM_ALIVE[0].item()];
                            vbef3[SWARM_ALIVE[0].item()]=vbef3aux[SWARM_ALIVE[0].item()];
                            xBest[SWARM_ALIVE[0].item()]=xBestaux[SWARM_ALIVE[0].item()];
                            gBest[SWARM_ALIVE[0].item()]=gBestaux[SWARM_ALIVE[0].item()];
                            gbestvalue[SWARM_ALIVE[0].item()]=gbestvalueaux[SWARM_ALIVE[0].item()];
        #                 end
        #             end
    #         end
    #     end
    # end
    #
        fit=fitaux;
        x=xaux;
        v=vaux;
        vbef1=vbef1aux;
        vbef2=vbef2aux;
        vbef3=vbef3aux;
        xBest=xBestaux;
        gBest=gBestaux;
        gbestvalue=gbestvalueaux;
        #
        nger=nger+1;
        #
        #
    xbest=gBestT[N_GER-1,:];
    fit=fitting[N_GER-1];
    print("fit=");
    print(fit);
    print("time=");
    end = time.time()
    print(end - start)
    print("xbest=");
    print(xbest);
    #xbest=0;
    #fit=0;
    #time=0;




def inicializaSwarm(N, N_PAR, V_MIN, V_MAX):
    swarm = zeros((int(N.item()),N_PAR));
    for i in range (0,int(N.item())):
        for j in range (0,N_PAR):
            swarm[i][j] = ( V_MAX[j]-V_MIN[j] ) + V_MIN[j]*st.random.rand(1,1) ;
    return swarm;
fodpso('10+5*x1**2-0.8*x2', st.array([-10,-20]),st.array([20, 40]), "min");
