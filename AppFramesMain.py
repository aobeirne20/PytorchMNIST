import numpy as np
import tkinter as tk
import Main


class App:

    def __init__(self, master):

        self.init_ui(master)
        self.init_bind()
        self.outline = None
        self.ink_matrix = np.zeros((28, 28))
        self.draw_canvas = None
        self.value_disp = None

    def init_ui(self, master):

        self.outline = tk.Frame(master, bg='#3B3E3F')
        self.outline.master.title("Numbr Readr")
        self.outline.master.geometry("900x500")
        self.outline.master.resizable(0,0)
        self.outline.pack()

        self.draw_frame = tk.Canvas(self.outline, bg='#E4E4E4', width=392, height=392, cursor='dot')
        buttons_frame = tk.Frame(self.outline, bg='#3B3E3F', width=350, height=40)
        self.output_frame = tk.Frame(self.outline, bg='#1F1F1F', width=350, height=150)
        data_frame = tk.Frame(self.outline, bg='#2F3937', width=350, height=120)

        self.draw_frame.grid(column=0, rowspan=3, padx=58, pady=58)
        self.output_frame.grid(row=0, column=1, padx=(0, 50), pady=(50, 0))
        data_frame.grid(row=1, column=1, padx=(0, 50), pady=(10, 80))
        buttons_frame.grid(row=2, column=1, padx=(0, 50), pady=(0, 50), sticky=tk.W)

        clear_button = tk.Button(buttons_frame, text='CLEAR', command=self.clear_canvas, bg='#3C3C3C', fg='#BDBDBD')
        clear_button.pack(side=tk.LEFT, padx=(0, 50))

        run_button = tk.Button(buttons_frame, text='RUN', command=self.run_net, bg='#3C3C3C', fg='#BDBDBD')
        run_button.pack(side=tk.LEFT)

        self.num_input = tk.Canvas(self.output_frame, bg='#E4E4E4', width=112, height=112)
        self.num_output = tk.Frame(self.output_frame, bg='#E4E4E4', width=112, height=112)
        self.num_input.grid(row=0, column=0, padx=(19, 44), pady=19)
        self.num_output.grid(row=0, column=1, padx=(44, 19), pady=19)

    def init_bind(self):
        self.draw_frame.bind("<B1-Motion>", self.mouse_click)
        self.draw_frame.bind("<Button-1>", self.mouse_click)

    def mouse_click(self, event):
        self.draw_frame.focus_set()
        pixel_x = event.x // 14
        pixel_y = event.y // 14
        if self.ink_matrix[pixel_y, pixel_x] == 0:
            self.render_canvas(pixel_x, pixel_y)
            self.render_mini_canvas(pixel_x, pixel_y)
            self.ink_matrix[pixel_y, pixel_x] = 1

    def render_canvas(self, pixel_x, pixel_y):
        self.draw_frame.create_rectangle(pixel_x*14, pixel_y*14, (pixel_x+1)*14, (pixel_y+1)*14, fill='black')

    def render_mini_canvas(self, pixel_x, pixel_y):
        self.num_input.create_rectangle(pixel_x*4, pixel_y*4, (pixel_x+1)*4, (pixel_y+1)*4, fill='black')

    def clear_canvas(self):
        self.draw_frame.delete('all')
        self.num_input.delete('all')
        self.ink_matrix = np.zeros((28, 28))

    def run_net(self):
        probs = Main.use_net(self.ink_matrix)
        print(probs)
        value = np.argmax(probs)
        if self.value_disp is not None:
            self.value_disp.destroy
        self.value_disp = tk.Text(self.num_output, height=1, width=1)
        self.value_disp.pack()
        self.value_disp.insert(tk.END, value)


root = tk.Tk()
app = App(root)
root.mainloop()

