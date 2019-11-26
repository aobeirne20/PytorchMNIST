import tkinter as tk


def setup(conf):
    gui = tk.Tk()
    gui.geometry("{}x{}".format(conf.width, conf.height))
    gui.resizable(0, 0)
    gui.title(conf.name)
    canvas = tk.Canvas(
        gui,
        width=conf.width,
        height=conf.height,
        bg='#000000',
        highlightthickness=0,
        borderwidth=0)

    canvas.pack(fill="both")
    canvas.create_rectangle(0, 0, conf.width, conf.height, fill="#2F3937", outline="#2F3937")

    return gui, canvas

class TkinterConf():
    def __init__(self, width, height, name=None):
        self.width = width
        self.height = height
        self.name = "Black Box" if name is None else name


Conf = TkinterConf(1000, 700, 'MNIST Drawpad')
gui, canvas = setup(Conf)
gui.update()
gui.mainloop()


