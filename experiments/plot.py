import matplotlib.pyplot as plt
import glob
import re
import sys
import os

args =sys.argv
datasetname = args[1]

save_dir = 'figure/'

gradfiles = glob.glob(datasetname+'/'+datasetname+'_grad_*') #update

accgradfiles = glob.glob(datasetname+'/'+datasetname+'_acc_grad_*')

coorfilename = datasetname+'/'+datasetname+'_coor.txt' #update

gradfiles.remove(datasetname+'/'+datasetname+'_grad_10.txt')

if args[2]=='epoch':
    savefilename = save_dir + datasetname + '_epoch.png'
elif args[2]=='time':
    savefilename = save_dir + datasetname + '_time.png'
else:
    print("Option Error") 


y_coor=[]
time_coor = []
y_grad = {}
time_grad = {}
y_accgrad = {}
time_accgrad = {}


with open(coorfilename) as f:            
    for line in f:
        lin = line.split(' ')
        y_coor.append(float(lin[1]))
        time_coor.append(float(lin[2]))

for gradfilename in gradfiles:
    with open(gradfilename) as f:
        step = re.search('grad_(.*).txt',gradfilename).group(1) #update
        y_list = []
        time_list = []
        for line in f:
            lin = line.split(' ')
            y_list.append(float(lin[1]))
            time_list.append(float(lin[2]))
        y_grad[step] = y_list
        time_grad[step] = time_list

for accgradfilename in accgradfiles:
    with open(accgradfilename) as f:
        mu = re.search('acc_grad_(.*).txt',accgradfilename).group(1) #update
        y_list = []
        time_list = []
        for line in f:
            lin = line.split(' ')
            y_list.append(float(lin[1]))
            time_list.append(float(lin[2]))
        y_accgrad[mu] = y_list
        time_accgrad[mu] = time_list



#figure
fig, ax = plt.subplots()

#plot
if args[2]=="epoch":
    
    for step in y_grad:
        ax.plot(range(len(y_grad[step])), y_grad[step] ,marker='x', markersize = 3, markevery=50, linestyle='--',label='GD('+str(step)+')',linewidth=0.5)
    
    for mu in y_accgrad:
        ax.plot(range(len(y_accgrad[mu])), y_accgrad[mu] ,marker='s', markersize = 3, markevery=50, linestyle='--',label='AGD('+str(mu)+')',linewidth=0.5)
    
    
    ax.plot(range(len(y_coor)), y_coor,marker='o', markersize = 3, markevery=50, linestyle=':',label='Proposed',linewidth=0.5)

    ax.legend()
    plt.xlim([0,len(y_coor)])
    ax.set_xlabel('epoch')

elif args[2]=="time":
    for step in time_grad:
        ax.plot(time_grad[step], y_grad[step] ,marker='x', markersize = 3, markevery=50, linestyle='--',label='GD('+str(step)+')',linewidth=0.5)

    for mu in time_accgrad:
        ax.plot(time_accgrad[mu], y_accgrad[mu] ,marker='s', markersize = 3, markevery=50, linestyle='--',label='AGD('+str(mu)+')',linewidth=0.5)
    
    ax.plot(time_coor, y_coor ,marker='o', markersize = 3, markevery=50, linestyle=':',label='Proposed',linewidth=0.5)
    
    ax.legend(loc = 'upper right')
    ax.set_xlabel('time[sec]')

ax.set_yscale("log")
plt.subplots_adjust(left = 0.15)


ax.set_ylabel('L_D-L_D*')

plt.title(args[1])
fig.savefig(os.path.join(save_dir,savefilename))
print("Finished plotting " + datasetname + "_" + args[2] + ".png")
