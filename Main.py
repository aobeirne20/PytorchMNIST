import numpy as np
import tkinter as tk
import PIL
import MNIST_NeuralNet


class App:

    def __init__(self):

        master = tk.Tk()
        self.init_ui(master)

        master.mainloop()

    def init_ui(self, master):
        master.title("Numbr Readr")
        master.geometry("900x500")
        master.resizable(0, 0)

        outline = tk.Frame(master, bg='#6A6A6A')
        outline.pack()

        user_canvas = DrawingCanvas(outline)
        nn_frame = tk.Frame(outline, bg='#1F1F1F', width=350, height=150)
        data_frame = tk.Frame(outline, bg='#3B3E3F', width=350, height=120)
        buttons_frame = tk.Frame(outline, bg='#3B3E3F', width=350, height=40)

        user_canvas.grid(column=0, rowspan=3, padx=50, pady=50)
        nn_frame.grid(row=0, column=1, padx=(0, 50), pady=(50, 0))
        data_frame.grid(row=1, column=1, padx=(0, 50), pady=(0, 90))
        buttons_frame.grid(row=2, column=1, padx=(0, 50), pady=(0, 50), sticky=tk.W)

        self.init_bind(user_canvas)

    def init_bind(self, user_canvas):
        user_canvas.bind("<B1-Motion>", user_canvas.draw_event)
        user_canvas.bind("<Button-1>", user_canvas.draw_event)


class DrawingCanvas(tk.Canvas):

    def __init__(self, outline):
        tk.Canvas.__init__(self, outline)
        self.config(bg='#E4E4E4', width=400, height=400, cursor='dot')
        self.ink_matrix = np.zeros((400, 400))
        self.img = tk.PhotoImage(width=400, height=400)
        self.create_image((200, 200), image=self.img, state="normal")
        for x in range(0, 401):
            for y in range(0, 401):
                self.img.put('#FFFFFF', (x, y))

    def draw_event(self, event):
        self.focus_set()
        self.gaussian_pen(event.x, event.y)

    def gaussian_pen(self, center_x, center_y):
        for x in range(0, 15):
            for y in range(0, 15):
                ink_dark = 255 - int(300 * np.exp(-1 * ((x ** 2) / (2 * (10 ** 2)) + (y ** 2) / (2 * (10 ** 2)))))
                for (pos_x, pos_y) in list(((x,y), (-x,y), (-x,-y), (x,-y))):
                    quad1 = self.img.get(center_x + pos_x, center_y + pos_y)
                    if ink_dark < 0:
                        ink_dark = 0
                    if ink_dark > 100:
                        pass
                    elif ink_dark < quad1[0]:
                        ink_hex = '#%02x%02x%02x' % (ink_dark, ink_dark, ink_dark)
                        self.img.put(ink_hex, (center_x + pos_x, center_y + pos_y))
                        self.ink_matrix[center_y + pos_y, center_x + pos_x] = ink_dark
                        print(self.ink_matrix)



















app = App()

