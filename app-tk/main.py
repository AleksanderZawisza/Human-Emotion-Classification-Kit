import tkinter as tk
# from tkinter import ttk

from load_page import LoadPage
from base_page import BasePage
from prediction_page import PredictionPage
from settings_page import SettingsPage


class TkinterApp(tk.Tk):
    # __init__ function for class tkinterApp
    def __init__(self, *args, **kwargs):
        # __init__ function for class Tk
        tk.Tk.__init__(self, *args, **kwargs)

        # creating a container
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # canvas = tk.Canvas(container, width=800, height=500)
        # canvas.grid()

        # initializing frames to an empty array
        self.frames = {}

        # iterating through a tuple consisting of the different page layouts
        for F in (BasePage, LoadPage, PredictionPage, SettingsPage):
            frame = F(container, self)

            # initializing frame of that object from all pages
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(BasePage)

    # to display the current frame passed as parameter
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

    def go_back(self):
        frame = self.frames[BasePage]
        frame.tkraise()


if __name__ == "__main__":
    # Driver Code
    app = TkinterApp()
    app.mainloop()
