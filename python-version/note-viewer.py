import io
from Symbol import *
from tkinter import *
from tkinter.ttk import *
import os




if __name__ == "__main__":

    training_data_path = "/Users/ivy/Desktop/Senior_Seminar/manuscript/training-data"
    filepaths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(training_data_path) for f in filenames if os.path.splitext(f)[1] == '.txt']

    symbols_list = list(map((lambda n: Symbol(n)), filepaths))
    