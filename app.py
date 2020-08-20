from tkinter import *
import PIL
from PIL import Image, ImageDraw
import numpy as np
import cv2
import consts
from keras.models import load_model
import tools as tl
model = load_model('mnist.h5')

def predict():
    pix = np.array(image1.getdata()).reshape(image1.size[0], image1.size[1],3)/255
    R,G,B = cv2.split(pix)
    resized = cv2.resize(R, consts.resize_shape)
    input = np.reshape(resized,consts.for_one)

    cv.create_text(20, 20, text=tl.get_mnist_prediction(model.predict(input)))




def activate_paint(e):
    global lastx, lasty
    cv.bind('<B1-Motion>', paint)
    lastx, lasty = e.x, e.y


def paint(e):
    global lastx, lasty
    x, y = e.x, e.y
    cv.create_line((lastx, lasty, x, y), width=20,capstyle=ROUND, smooth=TRUE, splinesteps=36)
    #  --- PIL
    draw.line((lastx, lasty,x,y), fill='white', width=20)
    draw.rectangle((lastx,lasty,x,y),fill='white')
    lastx, lasty = x, y

def clear():
    draw.rectangle(((0,0),(280,280)),fill = 'black')
    cv.delete("all")

image1 = PIL.Image.new('RGB', (280, 280), 'black')
draw = ImageDraw.Draw(image1)

root = Tk()

lastx, lasty = None, None


cv = Canvas(root, width=280, height=280, bg='white')
# --- PIL

cv.bind('<1>', activate_paint)
cv.pack(expand=YES, fill=BOTH)

btn_save = Button(text="predict", command=predict)
btn_save.pack()

btn_c = Button(text="clear", command=clear)
btn_c.pack()

root.mainloop()
