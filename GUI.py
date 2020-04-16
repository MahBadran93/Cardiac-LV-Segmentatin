# -*- coding: utf-8 -*-

import  tkinter as tk

#creating our main user interface 

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        master.title('Segment LV Cardiac')
        self.pack()
        self.create_widgets()
        
    def interface(self):
        tk.Frame
    def create_widgets(self):
        self.frame = tk.Scrollbar(self)
        self.hi_there = tk.Button(self)
        self.hi_there["text"] = "Hello World\n(click me)"
        self.hi_there["command"] = self.say_hi
        self.hi_there.pack(side="top")

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.pack(side="bottom")

    def say_hi(self):
        print("hi there, everyone!")

root = tk.Tk()
app = Application(master=root)
app.mainloop()
            