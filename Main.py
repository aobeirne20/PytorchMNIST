import numpy as np
import tkinter as tk
from PIL import Image, ImageTk, ImageGrab
from MNIST_NeuralNet import NetWrapper, MNISTData, Net


class App:

    def __init__(self):

        master = tk.Tk()
        self.init_ui(master)

        self.MNIST = MNISTData()
        self.NN = NetWrapper()
        self.NN.load('Era2')
        self.NN.net.eval()
        self.input_28 = None

        master.mainloop()

    def init_ui(self, master):
        master.title("Numbr Readr")
        master.geometry("900x500")
        master.resizable(0, 0)

        outline = tk.Frame(master, bg='#6A6A6A')
        outline.pack()

        user_canvas = DrawingCanvas(outline)
        nn_frame = tk.Frame(outline, bg='#3B3E3F', width=350, height=150)
        data_frame = tk.Frame(outline, bg='#1F1F1F', width=350, height=120)
        buttons_frame = tk.Frame(outline, bg='#1F1F1F', width=350, height=40)

        user_canvas.grid(column=0, rowspan=3, padx=50, pady=50)
        nn_frame.grid(row=0, column=1, padx=(0, 50), pady=(50, 0))
        data_frame.grid(row=1, column=1, padx=(0, 50), pady=(0, 90))
        data_frame.grid_propagate(0)
        buttons_frame.grid(row=2, column=1, padx=(0, 50), pady=(0, 50), sticky=tk.W)
        buttons_frame.grid_propagate(0)

        input_bg = tk.PhotoImage(file='BlankOutput.pgm')
        input_box = tk.Label(nn_frame, bg='#000000', width=110, height=110, image=input_bg, bd=0)
        input_box.photo = input_bg
        output_box = tk.Label(nn_frame, bg='#000000', width=110, height=110, image=input_bg, bd=0)
        output_box.photo = input_bg
        output_box.grid_propagate(0)

        input_box.grid(row=0, column=0, padx=(20, 45), pady=20)
        output_box.grid(row=0, column=1, padx=(45, 20), pady=20)

        prob_frame_list = []
        prob_text_list = []
        for n in list((0, 1, 2, 3, 4, 5, 6, 7, 8, 9)):
            prob_frame_list.append(tk.Frame(data_frame, bg='#1F1F1F', width=80, height=24))
            prob_frame_list[n].grid(row=n % 5, column=n // 5, padx=(20,20))
            prob_frame_list[n].grid_propagate(0)
            prob_text_list.append(tk.Label(prob_frame_list[n], text="", fg="white", bg='#1F1F1F'))
            prob_text_list[n].grid(row=0, column=0)
            
        run_button = tk.Button(buttons_frame, text='RUN', command=lambda: self.run_nn(user_canvas, input_box, output_box, prob_text_list), bg='#3C3C3C', fg='#BDBDBD')
        run_button.pack(side=tk.LEFT, padx=(0, 10))

        clear_button = tk.Button(buttons_frame, text='CLEAR', command=lambda: user_canvas.clear_canvas(input_box), bg='#3C3C3C', fg='#BDBDBD')
        clear_button.pack(side=tk.LEFT, padx=(10, 50))

        save_button = tk.Button(buttons_frame, text='SAVE', command=lambda: self.save_nn(), bg='#3C3C3C', fg='#BDBDBD')
        save_button.pack(side=tk.LEFT, padx=(10, 10))

        learn_button = tk.Button(buttons_frame, text='LEARN', command=lambda: self.learn_nn(learn_field), bg='#3C3C3C', fg='#BDBDBD')
        learn_button.pack(side=tk.LEFT, padx=(10, 10))

        learn_field = tk.Entry(buttons_frame, width=10)
        learn_field.pack(side=tk.LEFT, padx=(10, 10))

        self.init_bind(user_canvas)

    def save_nn(self):
        self.NN.save("Era3")

    def init_bind(self, user_canvas):
        user_canvas.bind("<B1-Motion>", user_canvas.draw_event)
        user_canvas.bind("<Button-1>", user_canvas.draw_event)

    def run_nn(self, user_canvas, input_box, output_box, prob_text_list):
        input_array = Image.fromarray(np.array(user_canvas.ink_matrix.astype('uint8')))
        self.input_28 = input_array.resize((28, 28), resample=Image.LANCZOS)
        nn_input = np.array(self.input_28)
        input_110 = self.input_28.resize((110, 110))

        input_image = ImageTk.PhotoImage(input_110)
        input_box.config(image=input_image)
        input_box.photo = input_image

        prob_matrix = np.exp(np.array((self.NN.use(nn_input, self.MNIST.transform))))
        np.set_printoptions(precision=2, suppress=True)

        output_text = tk.Label(output_box, bg='#000000', bd=0, text=f" {np.argmax(prob_matrix)}", fg="white", font=("",75))
        output_text.grid(row=0, column=0)

        for n in list((0, 1, 2, 3, 4, 5, 6, 7, 8, 9)):
            prob_text_list[n].configure(text=f"{n} : {prob_matrix[0, n]*100:.6f}")
            prob_text_list[n].grid(row=0, column=0)

    def learn_nn(self, learn_field):
        learn_input = int(learn_field.get())
        if (learn_input > 9) or (learn_input < 0):
            print("Invalid entry")
            pass
        self.NN.user_learn(np.array(self.input_28), learn_input)


class DrawingCanvas(tk.Canvas):

    def __init__(self, outline):
        tk.Canvas.__init__(self, outline)
        self.config(bg='#E4E4E4', width=400, height=400, cursor='dot')
        self.ink_matrix = np.zeros((400, 400))
        self.img = tk.PhotoImage(file="BlankCanvas.pgm")
        self.create_image((200, 200), image=self.img, state="normal")

    def draw_event(self, event):
        self.focus_set()
        self.gaussian_pen(event.x, event.y)

    def clear_canvas(self, input_box):
        self.ink_matrix = np.zeros((400, 400))
        self.img = tk.PhotoImage(file="BlankCanvas.pgm")
        self.create_image((200, 200), image=self.img, state="normal")

    def gaussian_pen(self, center_x, center_y):
        for x in range(0, 20):
            for y in range(0, 20):
                ink_dark = 255 - int(400 * np.exp(-1 * ((x ** 2) / (2 * (15 ** 2)) + (y ** 2) / (2 * (15 ** 2)))))
                for (pos_x, pos_y) in list(((x,y), (-x,y), (-x,-y), (x,-y))):
                    if 0 <= center_x + pos_x < 400 and 0 <= center_y + pos_y < 400:
                        quad1 = self.img.get(center_x + pos_x, center_y + pos_y)
                        if ink_dark < 0:
                            ink_dark = 0
                        if ink_dark > 75:
                            pass
                        elif ink_dark < quad1[0]:
                            ink_hex = '#%02x%02x%02x' % (ink_dark, ink_dark, ink_dark)
                            self.img.put(ink_hex, (center_x + pos_x, center_y + pos_y))
                            self.ink_matrix[center_y + pos_y, center_x + pos_x   ] = 255 - ink_dark




















app = App()

