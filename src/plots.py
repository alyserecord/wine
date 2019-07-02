import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')


def boxplot_stupid(df,vals,cols):
    
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    dfs = {}
    for col,val in enumerate(zip(cols,vals)):
        'df_{}'.format(col) = df[df[col] == val]
        dfs['df_{}'.format(col)] = val
    plt.boxplot(x=[d[v] for d,v in dfs.items()])
    plt.show()


            
