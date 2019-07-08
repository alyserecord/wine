import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')


def boxplot(df,vals,col,target_col,title,ylabel):
    
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    lst = []
    for val in vals:
        lst.append(df[df[col]==val][target_col])
    lst.append(df[~df[col].isin(vals)][target_col])   

    plt.boxplot(x = lst)
    vals.append('Other')
    plt.xticks([idx+1 for idx,num in enumerate(vals)], vals,fontsize = 14,rotation=30,ha='right')
    plt.yticks(fontsize = 12)
    plt.ylabel(ylabel,fontsize=18)
    plt.title(title,fontsize = 24)
    # plt.ylim(0,510)
    # plt.savefig('../figures/test.jpg')
    fig.subplots_adjust(bottom=0.18)
    plt.show()

def boxplot_multiple_cols(df,vals,cols,target_col,title,ylabel):
    
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    lst = []
    for val,col in zip(vals,cols):
        lst.append(df[df[col]==val][target_col])   

    plt.boxplot(x = lst)
    plt.xticks([idx+1 for idx,num in enumerate(vals)], cols,fontsize = 14,rotation=30,ha='right')
    plt.yticks(fontsize = 12)
    plt.ylabel(ylabel,fontsize=18)
    plt.title(title,fontsize = 24)
    # plt.ylim(0,510)
    # plt.savefig('../figures/test.jpg')
    fig.subplots_adjust(bottom=0.18)
    plt.show()

if __name__=='__main__':
    df = pd.read_csv('../data/50x50/sorted_df.csv')
    vals = ['Pinot Noir','Chardonnay','Cabernet Sauvignon']
    col = 'varietal'
    target_col = 'price'
    title = 'Price by Varietal'
    ylabel = 'Price ($)'
    boxplot(df,vals,col,target_col,title,ylabel)

    vals = [1,1,1]
    cols = ['red','white','sparking']
    target_col = 'price'
    title = 'Price by Type'
    ylabel = 'Price ($)'
    boxplot_multiple_cols(df,vals,cols,target_col,title,ylabel)


