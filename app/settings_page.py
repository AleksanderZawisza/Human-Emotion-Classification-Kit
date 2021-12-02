import tkinter as tk
from tkinter import ttk

LARGEFONT = ("Verdana", 30)


class SettingsPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        model = []

        # label of frame Layout 2
        label = ttk.Label(self, text="Ustawienia predykcji", font=LARGEFONT)

        # putting the grid in its place by using grid
        label.grid(row=0, column=0, padx=100, pady=40)

        button1 = ttk.Radiobutton(self, text="Model 1", variable=model, value=1)

        # putting the button in its place by using grid
        button1.grid(row=1, column=0, padx=50, pady=10)

        button2 = ttk.Radiobutton(self, text="Model 2", variable=model, value=2)

        # putting the button in its place by using grid
        button2.grid(row=2, column=0, padx=50, pady=10)

        button3 = ttk.Radiobutton(self, text="Model 3", variable=model, value=3)

        # putting the button in its place by using grid
        button3.grid(row=3, column=0, padx=50, pady=10)

        button4 = ttk.Button(self, text="Powrót",
                             command=lambda: controller.go_back())

        # putting the button in its place by using grid
        button4.grid(row=4, column=0, padx=50, pady=20)
