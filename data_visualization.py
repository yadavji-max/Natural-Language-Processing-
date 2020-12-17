import pandas as pd
from tkinter import *
import matplotlib.pyplot as plt
from pandas import ExcelWriter
from pandas import ExcelFile
import seaborn as sns

df = pd.read_excel('sentiment.xlsx')

def graph1():
    sns.countplot(x='Sentiment',hue="Sentiment",data=df)
    plt.show()
    
master = Tk()
master.title("Data Visualization")
master.geometry("200x200")

Label(master, text="Graphs").grid(row=0)
Button(master, text='Show Graphs', command=graph1).grid(row=1,column=0, pady=4)
Button(master, text='Quit', command=master.quit).grid(row=1,column=2, pady=4)
master.geometry("250x100")
mainloop()
