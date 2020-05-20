from glob import glob
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
    

csv_path = os.path.join('facerec/csv_data/')


def csv_writer(linalg_norm, name, file_name):
    information = {'Date': [], 'Linalg norm': [], 'Name': []}
    df = pd.DataFrame(information, columns= ['Date', 'Linalg norm', 'Name'])

    if os.path.isfile(csv_path + file_name):
        df = open(csv_path + file_name)
        df = pd.read_csv(df, sep = ',')

    date = datetime.datetime.now()
    date = datetime.datetime.strftime(date, '%Y/%m/%d/%H/%M/%S')
    data = {'Date': date, 'Linalg norm': linalg_norm, 'Name': name}
    df = df.append(data, ignore_index = True) 

    df.to_csv(csv_path + file_name, index = False, header=True)


def plot_csv_data(name, file_name):
    df = open(csv_path + file_name)
    df = pd.read_csv(df, sep = ',')

    df1 = df[df['Name'] == name]
    df2 = df[df['Name'] != name]

    ax = plt.gca()
    ax.set_ylim([0,1])
    #ax.invert_yaxis()
    ax.plot(df1.index, df1["Linalg norm"], label=name, color='blue')
    ax.plot(df2.index, df2["Linalg norm"], label='False positive', color='red')
    #plt.xticks(rotation=90)
    plt.legend(loc=1)
    plt.savefig(os.path.join('facerec/csv_data/' + file_name[:-4] + '.png'))
    plt.show()