import numpy as np
from numpy import linalg
import math
import scipy
from scipy.linalg import sqrtm
import itertools
import matplotlib.pyplot as plt

#THIS IS VERY SIMILAR TO THE RESTRICTED CODE, BUT I GENERALIZE THE ROOTHAAN
#EQUATION TO THE PAIRED POPLE-NESBET EQUATIONS

#exponent coefficients and contractions from basissetexchange
coe=[3.425250914,0.6239137298,0.1688554040]
d=[0.1543289673,0.5353281423,0.4446345422]
r=1

r=0.8
Et=[]
Enr=[]
Eea=[]
Eer=[]
xs=[]
Pa=[[.1,0],[0,0]]
Pb=[[0,0],[0,.1]]

for w in range(0,500):
    print(w)
    r+=0.01
    xs.append(r)
    #FIRST, CALCULATE SINGLE ELECTRON INTEGRALS:

    #this thing is the normalized overlap fucntion, equation (A.9) in Szabo
    overlap = lambda a,b,dr : ((2*b/3.1415926)**(.75))*((2*a/3.1415926)**(.75))\
            *((3.1415926/(a+b))**1.5)*math.exp(-a*b/(a+b)*dr**2)

    #initial form of overlap matrix
    S=[[0,0],[0,0]]

    #loop through each element of the S array, and loop through each interaction of \
    #  the three primitive gaussians which make up the CGF
    for i in range(0,2):    #each row of S matrix
        for j in range(0,2):        #each column of S matrix
            if i==j:
                dr=0
            else:
                dr=r
            for n in range(0,3):    #each primitive gaussian of Si
                for m in range(0,3):    #each primitive gaussian of Sj
                    S[i][j] += d[m]*d[n]*overlap(coe[m],coe[n],dr)
    S=np.array(S)


    #This is kinetic single-electron terms, eq (A.11)
    kinetic = lambda a,b,dr : ((2*b/3.1415926)**(.75))*((2*a/3.1415926)**(.75))\
                            *a*b*(3-2*a*b*(dr**2)/(a+b))/(a+b)\
                            *((3.1415926/(a+b))**1.5)*math.exp(-a*b/(a+b)*dr**2)

    #same form as above
    T=[[0,0],[0,0]]
    for i in range(0,2):    #each row of T matrix
        for j in range(0,2):        #each column of T matrix
            if i==j:
                dr=0
            else:
                dr=r
            for n in range(0,3):    #each primitive gaussian of Ti
                for m in range(0,3):    #each primitive gaussian of Tj
                    T[i][j] += d[m]*d[n]*kinetic(coe[m],coe[n],dr)
    T=np.array(T)


    #now it is useful to make a function which calculates the new center of
    #a gaussian which resulted from two other gaussians multiplied
    center = lambda a,b,ra,rb : (a*ra +b*rb)/(a+b)

    #This calculates nuclear attraction potential, eq (A.33)
    attraction = lambda a,b,ra,rb,rc : -((2*b/3.1415926)**(.75))*((2*a/3.1415926)**(.75))\
                            *(2*3.1415926/(a+b))*math.exp(-(a*b/(a+b))*(ra-rb)**2)\
                            *0.5*(3.1415926/((a+b)*(center(a,b,ra,rb)-rc)**2))**0.5\
                            *math.erf(((a+b)*(center(a,b,ra,rb)-rc)**2)**0.5)

    #same form as above, one matrix for each nucleus
    V_1=[[0,0],[0,0]]
    V_2=[[0,0],[0,0]]
    for i in range(0,2):    #each row of T matrix
        for j in range(0,2):        #each column of T matrix
            if i==0:
                ra=0
            else:
                ra=r
            if j==0:
                rb=0
            else:
                rb=r
            for n in range(0,3):    #each primitive gaussian of Ti
                for m in range(0,3):    #each primitive gaussian of Tj
                    #OK, here is a problem: if both basis functions are centered
                    #on the same nucleus whose potential we are calculating,
                    #then we get a divided by 0 error in the attraction function.
                    #this is remedied by allowing nuclear coord to be very close to
                    #the center of the basis funcs, but not quite at it.
                    #This method reproduces the book's table for r=1.4
                    V_1[i][j] += d[m]*d[n]*attraction(coe[m],coe[n],ra,rb,0.00000000000001)
                    V_2[i][j] += d[m]*d[n]*attraction(coe[m],coe[n],ra,rb,r+.00000000000001)
    V_1=np.array(V_1)
    V_2=np.array(V_2)
    H=T+V_1+V_2




    #CALCULATE TWO-ELECTRON INTEGRALS
    #first two lines are just normalization
    twointegrals = lambda a, b, c, d, ra, rb, rc, rd : \
            ((2*b/3.1415926)**(.75))*((2*a/3.1415926)**(.75))\
            *((2*c/3.1415926)**(.75))*((2*d/3.1415926)**(.75))\
            *((2*3.1415926**2.5)/((a+b)*(c+d)*(a+b+c+d)**.5))\
            *math.exp(-((a*b/(a+b))*(ra-rb)**2) - (c*d/(c+d))*(rc-rd)**2)\
            *.5*(3.1415926/(((a+b)*(c+d)/(a+b+c+d))*(center(a,b,ra,rb)-center(c,d,rc,rd))**2))**.5\
            *math.erf((((a+b)*(c+d)/(a+b+c+d))*(center(a,b,ra,rb)-center(c,d,rc,rd))**2)**.5)


    #we don't need to calculate every possible combination because there are a lot of
    #equavalencies. but they are not hard to calculate for minimal basis H2, and itll be
    #easier to reference later so will do it anyway
    two=[[[[0,0],[0,0]],[[0,0],[0,0]]],[[[0,0],[0,0]],[[0,0],[0,0]]]]
    for k,l,p,u in itertools.product(range(0,2),range(0,2),range(0,2),range(0,2)):
        for n,m,i,j in itertools.product(range(0,3),range(0,3),range(0,3),range(0,3)):
            if k==0:
                rk=0
            else:
                rk=r

            if l==0:
                rl=0
            else:
                rl=r

            if p==0:
                rp=0
            else:
                rp=r

            if u==0:
                ru=0
            else:
                ru=r

            two[k][l][p][u]+= d[m]*d[n]*d[i]*d[j]\
            *twointegrals(coe[m],coe[n],coe[i],coe[j],rk,rl,rp,ru+.0000000001)

    #Now we need to diagonalize S and find the transformation matrix X
    eigenvalues,eigenvectors=np.linalg.eig(S)

    #it seems like the eigenvalues and vectors are not in the same order
    #this orders them correctly
    if eigenvectors[0][0] == np.matmul(S,eigenvectors[0])[0]/eigenvalues[0]:
        pass
    else:
        temp=eigenvalues[0]
        eigenvalues[0]=eigenvalues[1]
        eigenvalues[1]=temp

    #this is small s
    s=np.diag(eigenvalues)

    #Ut is transpose conjugate of U matrix, where U matrix is the eigenvectors as the columns
    Ut= np.conj(np.array([[eigenvectors[0][0],eigenvectors[0][1]],[eigenvectors[1][0],eigenvectors[1][1]]]))
    U = np.transpose(Ut)
    #this makes the inverse root of the diagonal s matrix: inhalfs
    halfs=scipy.linalg.sqrtm(s)
    inhalfs=halfs
    inhalfs[0][0]= 1/(halfs[0][0])
    inhalfs[1][1]= 1/(halfs[1][1])
    #form the transformation matrix using canonical orthogonalization
    X = np.matmul(np.matmul(U,inhalfs),Ut)
    Xt= np.conj(np.transpose(X))

    #NOW WE CAN BEGIN THE ITERATIVE PROCESS

    #initial guesses at Fs,Ps:
    Pa=np.array(Pa)
    Pb=np.array(Pb)
    P=Pa+Pb
    Ga=[[0,0],[0,0]]
    Gb=[[0,0],[0,0]]
    for i,j, in itertools.product(range(0,2),range(0,2)):
        for k,l in itertools.product(range(0,2),range(0,2)):
            Ga[i][j]+= ((P[k][l])*two[i][j][l][k]-Pa[k][l]*two[i][k][l][j])
            Gb[i][j]+= ((P[k][l])*two[i][j][l][k]-Pb[k][l]*two[i][k][l][j])

    Ga=np.array(Ga)
    Gb=np.array(Gb)

    Fa=H+Ga
    Fb=H+Gb

    E=[]

    for n in range(0,20):
        Fpa=np.matmul(np.matmul(Xt,Fa),X)
        Fpb=np.matmul(np.matmul(Xt,Fb),X)

        eigenvaluesa,eigenvectorsa=np.linalg.eig(Fpa)
        eigenvaluesb,eigenvectorsb=np.linalg.eig(Fpb)
        Cpa= np.transpose(np.array(eigenvectorsa))
        Cpb= np.transpose(np.array(eigenvectorsb))

        Ca=np.matmul(X,Cpa)
        Cb=np.matmul(X,Cpb)

        for i,j in itertools.product(range(0,2),range(0,2)):
            Pa[i][j]=Ca[i][0]*np.conj(Ca[j][0])
            Pb[i][j]=Cb[i][0]*np.conj(Cb[j][0])
        Pa=np.array(Pa)
        Pb=np.array(Pb)
        P=Pa+Pb
        Ga=[[0,0],[0,0]]
        Gb=[[0,0],[0,0]]
        for i,j, in itertools.product(range(0,2),range(0,2)):
            for k,l in itertools.product(range(0,2),range(0,2)):
                Ga[i][j]+= ((P[k][l])*two[i][j][l][k]-Pa[k][l]*two[i][k][l][j])
                Gb[i][j]+= ((P[k][l])*two[i][j][l][k]-Pb[k][l]*two[i][k][l][j])

        Ga=np.array(Ga)
        Gb=np.array(Gb)

        Fa=H+Ga
        Fb=H+Gb


    if r>.9999999 and r<1.00000001:
        Eattracone=0
        Erepulsone=0
        for i,j in itertools.product(range(0,2),range(0,2)):
            Eattracone+=P[i][j]*H[j][i]
            Erepulsone+= 0.5*(Pa[i][j]*Ga[j][i]+Pb[i][j]*Gb[j][i])

        Eone=Eattracone+Erepulsone
        Nucrepulsone=1/r
        Etotone=Eone+Nucrepulsone
        print("Made it to r=1")


    Eattrac=0
    Erepuls=0
    E=0

    for i,j in itertools.product(range(0,2),range(0,2)):
        Eattrac+=P[i][j]*H[j][i]
        Erepuls+= 0.5*(Pa[i][j]*Ga[j][i]+Pb[i][j]*Gb[j][i])
    E= Eattrac+Erepuls
    Nucrepuls=1/r
    Etot= E+ Nucrepuls
    Et.append(Etot)
    Enr.append(Nucrepuls)
    Eer.append(Erepuls)
    Eea.append(Eattrac)

index=Et.index(min(Et))
xs[index]


with open('output_unrestricted.txt', 'w') as f:
    f.write("This script computes the energy of H2 using Unrestricted Hartree-Fock\
 Theory with minimal basis set STO-3G.\n")
    f.write("\n")
    f.write("\n")
    f.write("With internuclear distance r=1 Angstrom, the energy breakdown in Hartree Atomic Units is:\n")
    f.write("       Total Energy : "+str(Etotone)+"\n")
    f.write("       Nuclear Repulsion Energy : "+str(Nucrepulsone)+"\n")
    f.write("       Electronic Energy : "+str(Eone)+"\n")
    f.write("       Electron Repulsion Energy : "+str(Erepulsone)+"\n")
    f.write("       Electron-Nuclear Attraction Energy : "+str(Eattracone)+"\n")
    f.write("\n")
    f.write("\n")
    f.write("The optimum bond length is found to be r="+str(xs[index])+" AU.\n")
    f.write("Energy breakdown at this minimum:\n")
    f.write("       Total Energy : "+str(Et[index])+"\n")
    f.write("       Nuclear Repulsion Energy : "+str(Enr[index])+"\n")
    f.write("       Electronic Energy : "+str(Eea[index]+Eer[index])+"\n")
    f.write("       Electron Repulsion Energy : "+str(Eer[index])+"\n")
    f.write("       Electron-Nuclear Attraction Energy : "+str(Eea[index])+"\n")



plt.plot(xs,Et, label= "Total Energy",marker='None', color='black', linestyle='-')
# plt.plot(xs,Enr, label= "Nucleus-Nucleus Repulsion Energy (Hartree AU)",marker='None', color='red', linestyle='-')
# plt.plot(xs,Eea, label= "Nuclei-Electron Attraction Energy (Hartree AU)",marker='None', color='blue', linestyle='-')
# plt.plot(xs,Eer, label= "Electron-Electron Repulsion Energy (Hartree AU)",marker='None', color='purple', linestyle='-')

plt.title("Energy of H2 vs bond length")
plt.xlabel("Bond length (AU)")
plt.ylabel("Energy (Hartree AU)")
#plt.legend()#loc='upper center')
outfile = 'energy_unrestricted.png'
plt.savefig(outfile)#,bbox_inches='tight')
