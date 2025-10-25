# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 10:12:32 2024

@author: Vik
"""

import tkinter as tk
from tkinter import Button, Canvas, W
import PIL
from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('mnist_model.h5')

def predict_digit(img):
    img = img.resize((28,28), PIL.Image.ANTIALIAS)
    img = img.convert('L')
    img = np.invert(img)
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1).astype('float32') / 255.0
    prediction = model.predict(img)
    return np.argmax(prediction), np.max(prediction)

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0

        self.canvas = Canvas(self, width=280, height=280, 
                             bg="white")
        self.label = tk.Label(self, text="Draw...", 
                              font=("Helvetica", 48))
        self.classify_btn = Button(self, 
                                   text="Recognize", 
                            command=self.classify_handwriting)
        self.button_clear = Button(self, text="Clear", 
                                   command=self.clear_all)

        self.canvas.grid(row=0, column=0, pady=2, sticky=W)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, 
                               padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        self.canvas.bind("<B1-Motion>", self.draw_lines)

        self.image = PIL.Image.new("RGB", (280, 280), 
                                   "white")
        self.draw = ImageDraw.Draw(self.image)

    def draw_lines(self, event):
        self.x, self.y = event.x, event.y
        r = 10
        self.canvas.create_oval(self.x - r, self.y - r, 
                self.x + r, self.y + r, fill='black', 
                outline='black')
        self.draw.ellipse([self.x - r, self.y - r, 
                           self.x + r, self.y + r], 
                          fill="black")

    def classify_handwriting(self):
        digit, acc = predict_digit(self.image)
        self.label.configure(text=str(digit) + ', ' + 
                             str(int(acc * 100)) + '%')

    def clear_all(self):
        self.canvas.delete("all")
        self.image = PIL.Image.new("RGB", (280, 280), 
                                   (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)
        self.label.configure(text="Draw...")

app = App()
app.mainloop()

