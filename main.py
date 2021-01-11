from SudokuCV import SudokuCV
import cv2

from tkinter import *
from tkinter import filedialog
from PIL import  Image, ImageTk


class GUI(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)
        w,h = 400, 500
        master.minsize(width=w, height=h)
        master.maxsize(width=w, height=h)
        master.title
        self.pack()

        self.file = Button(self, text='Browse', command=self.choose)
        self.choose = Label(self, text="Choose file").pack()
        #Replace with your image
        self.image = Image.open('Default.png')
        self.image = self.image.resize((350, 350))
        self.image = ImageTk.PhotoImage(self.image)
        self.label = Label(image=self.image)

        self.s = SudokuCV()
        self.file.pack()
        self.label.pack()

    def choose(self):
        ifile = filedialog.askopenfile(parent=self,mode='rb',title='Choose a file')
        path = Image.open(ifile)
        path = path.resize((350, 350))

        self.image2 = ImageTk.PhotoImage(path)
        self.label.configure(image=self.image2)
        self.label.image=self.image2

        solved = self.s.solve_sudoku_pic(str(ifile.name))
        solved = cv2.resize(solved, (350, 350))
        solved = Image.fromarray(solved)

        self.image2 = ImageTk.PhotoImage(solved)
        self.label.configure(image=self.image2)
        self.label.image=self.image2




root = Tk()
root.title("SudokuCV")
app = GUI(master=root)


app.mainloop()
root.destroy()



root.mainloop()