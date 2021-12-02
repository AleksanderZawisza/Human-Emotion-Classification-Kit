import tkinter as tk
from tkinter import ttk, filedialog


LARGEFONT = ("Verdana", 30)


class LoadPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        # label of frame Layout 2
        label = ttk.Label(self, text="Wczytaj zdjęcia", font=LARGEFONT)

        # putting the grid in its place by using grid
        label.grid(row=0, column=0, padx=100, pady=40)

        button1 = ttk.Button(self, text="Wczytaj nowe zdjęcia",
                             command=lambda: open_file())

        # putting the button in its place by using grid
        button1.grid(row=1, column=0, padx=50, pady=20)

        button2 = ttk.Button(self, text="Powrót",
                             command=lambda: controller.go_back())

        # putting the button in its place by using grid
        button2.grid(row=2, column=0, padx=50, pady=20)

        def open_file():
            filename = filedialog.askopenfile(mode='rb', title='Wybierz zdjęcia')
