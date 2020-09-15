from Line import *
from Point import *
from tkinter import *
from tkinter.ttk import * 

class Symbol:
    name = "Unnamed Symbol"
    lines = []
    def __init__(self, filename):
        with open(filename) as f:
            self.name = f.readline()
            lines = f.read().split("\n")
            self.lines = self.turn_text_into_points(lines)            
    def turn_text_into_points(self,text_lines):
        lines = []
        for line_of_text in text_lines:
            splitted_by_semicolon = line_of_text.split(";")
            splitted_by_semicolon.remove('')

            for i in range(0,len(splitted_by_semicolon)-1):
                current_pair = splitted_by_semicolon[i].split(",")
                next_pair = splitted_by_semicolon[i+1].split(",")
                line = Line(Point(current_pair[0],current_pair[1]), Point(next_pair[0],next_pair[1]))

                lines.append(line)

        return lines