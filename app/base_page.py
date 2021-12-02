import tkinter as tk
from tkinter import ttk

from app.load_page import LoadPage
from app.settings_page import SettingsPage
from app.prediction_page import PredictionPage

LARGEFONT = ("Verdana", 30)


class BasePage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        # label of frame Layout 2
        label = ttk.Label(self, text="Human Emotion Recognition Kit", font=LARGEFONT)

        # putting the grid in its place by using grid
        label.grid(row=0, column=0, padx=100, pady=40)

        button1 = ttk.Button(self, text="Uruchom predykcję",
                             command=lambda: controller.show_frame(PredictionPage))

        # putting the button in its place by using grid
        button1.grid(row=1, column=0, padx=50, pady=20)

        # button to show frame 2 with text layout2
        button2 = ttk.Button(self, text="Wczytaj zdjęcia",
                             command=lambda: controller.show_frame(LoadPage))

        # putting the button in its place by using grid
        button2.grid(row=2, column=0, padx=50, pady=20)

        button3 = ttk.Button(self, text="Zmień ustawienia predykcji",
                             command=lambda: controller.show_frame(SettingsPage))

        # putting the button in its place by using grid
        button3.grid(row=3, column=0, padx=50, pady=20)
