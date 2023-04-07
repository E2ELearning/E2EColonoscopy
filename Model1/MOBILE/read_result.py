import pandas as pd

df = pd.read_csv('cnnlstm_result.txt', sep = ' |:|/', header = None, engine = 'python',keep_default_na=False)
df =df.select_dtypes(include='number')
df.columns = ['Train_loss','Train_reg','Train_cls','Train_F1',
              'Valid_loss','Valid_reg','Valid_cls','Valid_F1',
              'Test_loss','Test_reg','Test_cls','Test_F1']
print(df)

import matplotlib.pyplot as plt
import numpy as np

# for i in range(5):
#     legends = ['Train','Valid']

#     fig, ax1 = plt.subplots(figsize=(8,4))
    
#     ax1.set_xlabel('Sequence Length')
  
#     ax1.set_ylabel('Regression Loss')
#     ax1.plot(range(1,11), df['Train_reg'][i*10:(i+1)*10],marker = 'o', markersize = 4)
#     ax1.plot(range(1,11), df['Valid_reg'][i*10:(i+1)*10],marker = 'o', markersize = 4)
#     ax1.set_ylim([0,0.03])
    
#     ax2 = ax1.twinx()
#     ax2.set_ylabel('F1 score')
#     ax2.plot(range(1,11), df['Train_F1'][i*10:(i+1)*10], marker = 'o', markersize = 4, linestyle='dashed')
#     ax2.plot(range(1,11), df['Valid_F1'][i*10:(i+1)*10], marker = 'o', markersize = 4,linestyle='dashed')

#     ax2.set_ylim([0.7,1])
#     ax1.legend(legends, loc = 'lower left')
#     fig.tight_layout()
    
#     plt.savefig('alpha_{}.png'.format(pow(10,-i)))
#     plt.close()

for j in range(10):
    legends = ['Train','Valid']

    fig, ax1 = plt.subplots(figsize=(8,4))
    
    ax1.set_xlabel(r'$\alpha$')
    ax1.set_ylabel('Regression Loss')
    ax1.set_ylim([0,0.03])
    ax1.set_xscale('log')
    xticks = [pow(10,-x) for x in range(5)]
    train_reg = [df['Train_reg'][i*10+j] for i in range(5)]
    train_f1 = [df['Train_F1'][i*10+j] for i in range(5)]
    Valid_reg = [df['Valid_reg'][i*10+j] for i in range(5)]
    Valid_f1 = [df['Valid_F1'][i*10+j] for i in range(5)]
    
    ax1.plot(xticks, train_reg, marker = 'o', markersize = 4)
    ax1.plot(xticks, Valid_reg, marker = 'o', markersize = 4)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('F1 score')
    ax2.set_ylim([0.7,1])
    ax2.plot(xticks, train_f1, marker = 'o', markersize = 4, linestyle='dashed')
    ax2.plot(xticks, Valid_f1, marker = 'o', markersize = 4, linestyle='dashed')
    
    ax1.legend(legends, loc = 'lower left')
    fig.tight_layout()
    plt.savefig('seq_{}.png'.format(j+1))
    plt.close()