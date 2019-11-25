import tkinter as tk


class App:

    def __init__(self, master):

        outline = tk.Frame(master)
        outline.master.geometry("800x400")
        outline.pack()

        draw_frame = tk.Frame(outline, bg='white', width=300, height=300, padx=100, pady=100)
        buttons_frame = tk.Frame(outline, bg='cyan', width=400, height=50)
        output_frame = tk.Frame(outline, bg='black', width=400, height=200)
        data_frame = tk.Frame(outline, bg='gray', width=400, height=150)

        draw_frame.grid(column=0, rowspan=3)
        output_frame.grid(row=0, column=1)
        data_frame.grid(row=1, column=1)
        buttons_frame.grid(row=2, column=1)





    def say_hi(self):
        print("hi there, everyone!")

root = tk.Tk()

app = App(root)

root.mainloop()




