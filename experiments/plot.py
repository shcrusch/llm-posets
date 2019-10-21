import matplotlib.pyplot as plt
import glob
import re
import sys
import os

args =sys.argv
datasetname = args[1]

save_dir = '/home/hayashi/workspace/tbm-python/experiments/figure/'+datasetname+'/'

gradlist = glob.glob('/home/hayashi/workspace/tbm-python/experiments/'+datasetname+'/grad_*')
print('file for gradient descent:')
print(gradlist)

coorfilename = '/home/hayashi/workspace/tbm-python/experiments/'+datasetname+'/coor.txt'
print(coorfilename)
if args[2]=='kl' and args[3]=='epoch':
    savefilename = save_dir + datasetname + '_kl_epoch.png'
elif args[2]=='kl' and args[3]=='time':
    savefilename = save_dir + datasetname + '_kl_time.png'
elif args[2]=='sg' and args[3]=='epoch':
    savefilename = save_dir + datasetname + '_sg_epoch.png'
elif args[2]=='sg' and args[3]=='time':
    savefilename = save_dir + datasetname+'_sg_time.png'
else:
    print("Option Error") 

y_grads=[]
steps=[]
y_coor=[]
times_grads=[]
times_coor=[]
if args[2]=="kl":
    
    for gradfilename in gradlist:
        with open(gradfilename) as f:
            step = re.search('grad_(.*).txt',gradfilename).group(1)
            print(step)
            steps.append(step)

            y_grad=[]
            for line in f:
                lin = line.split(' : KL divergence: ')
                lin = lin[-1].split('  time : ')
                y_grad.append(float(lin[0]))
         
            y_grads.append(y_grad)
    
    with open(coorfilename) as f:
            
        y_coor=[]
        for line in f:
            lin = line.split(' : KL divergence: ')
            lin = lin[-1].split('  time : ')
            y_coor.append(float(lin[0]))

elif args[2]=="sg":
    for gradfilename in gradlist:
        with open(gradfilename) as f:
            step = re.search('grad_(.*).txt',gradfilename).group(1)
            steps.append(step)
            y_grad=[]
            for line in f:
                lin = line.split(' Squared Gradient: ')
                y_grad.append(float(lin[-1]))
            y_grads.append(y_grad)
    with open(coorfilename) as f:
        y_coor=[]
        for line in f:
            lin = line.split(' Squared Gradient: ')
            y_coor.append(float(lin[-1]))



if args[3]=="time":

    for gradfilename in gradlist:
        with open(gradfilename) as f:
            step = re.search('grad_(.*).txt',gradfilename).group(1)
            steps.append(step)
            time_grad=[]
            for line in f:
                lin = line.split(' Squared Gradient: ')
                lin = lin[0].split('  time : ')
                time_grad.append(float(lin[-1]))
            times_grads.append(time_grad)

    with open(coorfilename) as f:
        time_coor=[]
        for line in f:
            lin = line.split(' Squared Gradient: ')
            lin = lin[0].split('  time : ')
            time_coor.append(float(lin[-1]))

#figure
fig, ax = plt.subplots()

#plot
if args[3]=="epoch":

    for i in range(len(gradlist)):
        ax.plot(range(len(y_grads[i])), y_grads[i] ,marker='x', markersize = 6, markevery=50, linestyle='--',label=steps[i],linewidth=0.5)
    
    ax.plot(range(len(y_coor)), y_coor,marker='o', markersize = 5, markevery=50, linestyle=':',label="coor",linewidth=0.5)
    ax.legend()
    plt.xlim([0,len(y_grads[0])])
    ax.set_xlabel('epoch')
elif args[3]=="time":

    for i in range(len(gradlist)):
        ax.plot(times_grads[i], y_grads[i] ,marker='x', markersize = 6, markevery=50, linestyle='--',label=steps[i],linewidth=0.5)
    ax.plot(time_coor, y_coor ,marker='o', markersize = 5, markevery=50, linestyle=':',label="coor",linewidth=0.5)

    ax.legend()
    ax.set_xlabel('time[sec]')

#ax.set_yscale("log")
if args[2]=='kl':
    ax.set_ylabel('KL divergence')
elif args[2]=='sg':
    ax.set_ylabel('Squared gradient')

fig.savefig(os.path.join(save_dir,savefilename))
