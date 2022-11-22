import matplotlib.pyplot as plt
loss_=[]
with open("loss_.txt","r",encoding="utf-8") as fr:
    for line in fr.readlines():
        line_=str(line.strip())
        loss_.append(float(line_))
plt.plot(loss_)
plt.show()



