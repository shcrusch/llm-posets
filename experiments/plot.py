import matplotlib.pyplot as plt
import glob
import re
import sys
import os

args =sys.argv
datasetname = args[1]

save_dir = '/home/hayashi/workspace/tbm-python/experiments/figure/'+datasetname+'/'

gradlist = glob.glob('/home/hayashi/workspace/tbm-python/experiments/'+datasetname+'/Grad_*') #update

coorfilename = '/home/hayashi/workspace/tbm-python/experiments/'+datasetname+'/Coor.txt' #update

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
grad_steps=[]
y_coor=[]
times_grads=[]
times_coor=[]


if args[2]=="kl":

    with open(coorfilename) as f:            
        y_coor=[]
        for line in f:
            lin = line.split(' : KL divergence: ')
            lin = lin[-1].split('  time : ')
            y_coor.append(float(lin[0]))
            y_star = y_coor[-1]
        y_coor = list(map(lambda y:y-y_star,y_coor))

    for gradfilename in gradlist:
        with open(gradfilename) as f:
            step = re.search('Grad_(.*).txt',gradfilename).group(1) #update
            grad_steps.append(step)
            y_grad=[]
            for line in f:
                lin = line.split(' : KL divergence: ')
                lin = lin[-1].split('  time : ')
                y_grad.append(float(lin[0]))
            y_grad = list(map( lambda y:y - y_star,y_grad))
            y_grads.append(y_grad)


elif args[2]=="sg":

    for gradfilename in gradlist:
        with open(gradfilename) as f:
            step = re.search('grad_(.*).txt',gradfilename).group(1)
            grad_steps.append(step)
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
            step = re.search('Grad_(.*).txt',gradfilename).group(1) #update
            grad_steps.append(step)
            time_grad=[]
            for line in f:
#                lin = line.split(' Squared Gradient: ')
                lin = line.split('  time : ') #update lin[0] ->line
                time_grad.append(float(lin[-1]))
            times_grads.append(time_grad)


    with open(coorfilename) as f:
        time_coor=[]
        for line in f:
#            lin = line.split(' Squared Gradient: ')
            lin = line.split('  time : ') #update lin[0] -> line
            time_coor.append(float(lin[-1]))

#figure
fig, ax = plt.subplots()

#plot
if args[3]=="epoch":
    
    for i in range(len(gradlist)):
        ax.plot(range(len(y_grads[i])), y_grads[i] ,marker='x', markersize = 6, markevery=50, linestyle='--',label=grad_steps[i],linewidth=0.5)

    ax.plot(range(len(y_coor)), y_coor,marker='o', markersize = 5, markevery=50, linestyle=':',label='Proposed',linewidth=0.5)

    ax.legend()
    plt.xlim([0,len(y_grads[0])])
    ax.set_xlabel('epoch')

elif args[3]=="time":
    if args[2] == "kl":
        for i in range(len(gradlist)):
            ax.plot(times_grads[i], y_grads[i] ,marker='x', markersize = 6, markevery=50, linestyle='--',label=grad_steps[i],linewidth=0.5)

        ax.plot(time_coor, y_coor ,marker='o', markersize = 5, markevery=50, linestyle=':',label='Proposed',linewidth=0.5)
    elif args[2] == "sg":
        for i in range(len(gradlist)):
            ax.plot(times_grads[i], y_grads[i] ,marker='x', markersize = 6, markevery=50, linestyle='--',label=grad_steps[i],linewidth=0.5)


        ax.plot(time_coor, y_coor ,marker='o', markersize = 5, markevery=50, linestyle=':',label='Proposed',linewidth=0.5)
    
    ax.legend(loc = 'upper right')
    ax.set_xlabel('time[sec]')

ax.set_yscale("log")
plt.subplots_adjust(left = 0.15)

if args[2]=='kl':
    ax.set_ylabel('KL divergence')
elif args[2]=='sg':
    ax.set_ylabel('Squared gradient')
plt.title(args[1])
fig.savefig(os.path.join(save_dir,savefilename))
print("Finished plotting")
