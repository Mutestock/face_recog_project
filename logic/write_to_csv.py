from glob import glob
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
    

csv_path = os.path.join('facerec/csv_data/recognition_linalg_data.csv')


def csv_writer(linalg_norm):
    information = {'Date': [], 'Linalg norm': []}
    df = pd.DataFrame(information, columns= ['Date', 'Linalg norm'])

    if os.path.isfile(csv_path):
        df = open(csv_path)
        df = pd.read_csv(df, sep = ',')

    date = datetime.datetime.now()
    date = datetime.datetime.strftime(date, '%Y/%m/%d/%H/%M/%S')
    data = {'Date': date, 'Linalg norm': linalg_norm}
    df = df.append(data, ignore_index = True) 

    df.to_csv(csv_path, index = False, header=True)


def plot_csv_data():
    df = open(csv_path)
    df = pd.read_csv(df, sep = ',')

    #Simple plot
    df.plot(x="Date", y="Linalg norm")
    plt.gca().invert_yaxis()
    plt.xticks(rotation=90)
    plt.show()

    #Hard plot
    pd.plotting.register_matplotlib_converters()

    time = [datetime.datetime(int(i.split('/')[0].strip()),int(i.split('/')[1].strip()),int(i.split('/')[2].strip()),int(i.split('/')[3].strip()),int(i.split('/')[4].strip()),int(i.split('/')[5].strip())) for i in df['Date']]

    delay = [float(i) for i in df['Linalg norm']]

    plt.plot(time, delay)
    plt.gca().invert_yaxis()
    plt.grid(b='on')
    plt.xticks(rotation=90)

    plt.savefig('face_recog_project/facerec/csv_data/csv_pic.png')
    plt.show()