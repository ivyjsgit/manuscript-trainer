import io
import os
import tensorflow as tf
import random

class Symbol:
    symbol_class = "unknown_symbol"
    filepath="unknown"
    def __init__(self, filename):
        self.filename = os.path.basename(filename)
        splitted_by_slashes=filename.split("/")
        self.symbol_class=splitted_by_slashes[len(splitted_by_slashes)-2]



if __name__ == "__main__":
    training_data_path = "/Users/ivy/Desktop/Senior_Seminar/HOMUS-Bitmap"
    filepaths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(training_data_path) for f in filenames if os.path.splitext(f)[1] == '.png']
    symbols_list = list(map((lambda n: Symbol(n)), filepaths))

    #Shuffle the list
    shuffled_symbol_list = random.sample(symbols_list,len(symbols_list))

    #Get the classes and list of file paths
    shuffled_paths=[]
    shuffled_classes=[]

    for symbol in shuffled_symbol_list:
        shuffled_paths.append(symbol.filepath)
        shuffled_classes.append(symbol.symbol_class)
