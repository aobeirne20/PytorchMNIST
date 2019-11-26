
import tkinter as tk


class App:

    def __init__(self, master):

        self.init_ui(master)
        self.init_bind()
        self.outline = None

    def init_ui(self, master):

        self.outline = tk.Frame(master, bg='#3B3E3F')
        self.outline.master.title("Numbr Readr")
        self.outline.master.geometry("900x500")
        self.outline.master.resizable(0,0)
        self.outline.pack()

        self.draw_frame = tk.Frame(self.outline, bg='#E4E4E4', width=400, height=400, cursor='dot')
        buttons_frame = tk.Frame(self.outline, bg='cyan', width=350, height=40)
        output_frame = tk.Frame(self.outline, bg='black', width=350, height=150)
        data_frame = tk.Frame(self.outline, bg='#2F3937', width=350, height=120)

        self.draw_frame.grid(column=0, rowspan=3, padx=50, pady=50)
        output_frame.grid(row=0, column=1, padx=(0, 50), pady=(50, 0))
        data_frame.grid(row=1, column=1, padx=(0, 50), pady=(10, 80))
        buttons_frame.grid(row=2, column=1, padx=(0, 50), pady=(0, 50))

    def mouse_click(self, event):
        self.draw_frame.focus_set()
        print("clicked at", event.x, event.y)

    def init_bind(self):
        self.draw_frame.bind("<Button-1>", self.mouse_click)


root = tk.Tk()
app = App(root)
root.mainloop()

