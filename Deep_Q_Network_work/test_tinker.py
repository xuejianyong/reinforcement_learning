from tkinter import *
import tkinter as tk
from  PIL import Image, ImageTk, ImageGrab
import itertools as it
import numpy as np
import random
import time
import os


interaction_images = {
    'agent':'agent.png',
    'orange':'orange.png',
    'wall':'wall.png',
    'pac_1':'pac_1.png',
    'pac_2':'pac_2.png'
}
currentPath = os.path.join(os.getcwd(), 'images')


UNIT = 70
MAZE_W = 7
MAZE_H = 7

window_width = MAZE_W*UNIT
window_height = MAZE_H*UNIT
x_range = range(0, MAZE_H)
y_range = range(0, MAZE_W)
all_locations = np.array(list(it.product(x_range,y_range)))*UNIT
all_locations = all_locations.tolist()


class Env(tk.Tk, object):
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = ['north', 'south', 'east', 'west', 'stay']
        self.n_actions = len(self.action_space)
        self.prey_move_directions = ['ne', 'nw', 'se', 'sw']

        self.title('MMRL-Maze')
        x_cordinate = int((self.winfo_screenwidth() / 2) - (window_width / 2))
        y_cordinate = int((self.winfo_screenheight() / 2) - (window_height / 2))
        self.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white', height=MAZE_H * UNIT, width=MAZE_W * UNIT)
        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):  # create |
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):  # create --
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        """
        # create origin
        origin = np.array([20, 20])

        # hell
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        # hell
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')


        # create oval
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect -- the agent
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        """

        # create the agent and the fruit(or the prey)
        agent_image = Image.open(os.path.join(currentPath,interaction_images['agent']))
        agent_image = agent_image.resize((UNIT, UNIT), Image.ANTIALIAS)
        self.agent_img = ImageTk.PhotoImage(agent_image)
        self.agent = self.canvas.create_image(0, 0, anchor="nw", image=self.agent_img)

        # fruit_image = Image.open(interaction_images['orange'])
        fruit_image = Image.open(os.path.join(currentPath,interaction_images['orange']))
        fruit_image = fruit_image.resize((UNIT, UNIT), Image.ANTIALIAS)
        self.fruit_img = ImageTk.PhotoImage(fruit_image)
        all_locations.remove([0, 0])  # 这里猎物的位置要随机出现， 同时猎物的移动方向也确定了下来
        x_prey, y_prey = all_locations[random.randint(0, len(all_locations) - 1)]
        self.fruit = self.canvas.create_image(x_prey, y_prey, anchor="nw", image=self.fruit_img)
        self.prey_direction = self.prey_move_directions[random.randint(0, 3)]
        #self.prey_direction = 'sw'
        #self.canvas.create_rectangle(100, 100, UNIT, UNIT, fill='red')

        wall_image = Image.open(os.path.join(currentPath,interaction_images['wall']))
        wall_image = wall_image.resize((UNIT - 4, UNIT - 4), Image.ANTIALIAS)
        self.wall_img = ImageTk.PhotoImage(wall_image)

        pac_2_image = Image.open(os.path.join(currentPath,interaction_images['pac_2']))
        pac_2_image = pac_2_image.resize((UNIT - 10, UNIT - 10), Image.ANTIALIAS)
        self.pac_2_img = ImageTk.PhotoImage(pac_2_image)
        all_locations.remove(list(map(int, self.canvas.coords(self.fruit))))
        x_prey, y_prey = all_locations[random.randint(0, len(all_locations) - 1)]
        self.pac2 = self.canvas.create_image(x_prey+5, y_prey+5, anchor="nw", image=self.pac_2_img)

        pac_1_image = Image.open(os.path.join(currentPath, interaction_images['pac_1']))
        pac_1_image = pac_1_image.resize((UNIT - 10, UNIT - 10), Image.ANTIALIAS)
        self.pac_1_img = ImageTk.PhotoImage(pac_1_image)
        all_locations.remove(list(map(int, list(np.array(self.canvas.coords(self.pac2)) - 5))))
        x_prey, y_prey = all_locations[random.randint(0, len(all_locations) - 1)]
        self.canvas.create_image(x_prey + 5, y_prey + 5, anchor="nw", image=self.pac_1_img)

        # pack all
        self.canvas.bind("<Button-1>", lambda event: self.drawRect(event))
        self.canvas.pack()

    def drawRect(self, event):
        #print('click at %d, %d' % (event.x, event.y))
        click_x = (event.x // UNIT) * UNIT
        click_y = (event.y // UNIT) * UNIT
        click_x_color = click_x + UNIT//2
        click_y_color = click_y + UNIT // 2
        color = ImageGrab.grab().getpixel((event.x_root-event.x+click_x_color,event.y_root-event.y+click_y_color))
        #print(color)
        if color[0] == 51:
            self.canvas.create_rectangle(click_x+1, click_y+1, click_x+UNIT-1, click_y+UNIT-1, fill='white', outline='white')
        else:
            self.canvas.create_image(click_x + 2, click_y + 2, anchor="nw", image=self.wall_img)
        #print('the coordinates of the rect is start %d, %d, %d, %d' % (click_x*UNIT, click_y*UNIT,UNIT,UNIT))
        #self.canvas.create_rectangle(click_x+1, click_y+1, click_x+UNIT-1, click_y+UNIT-1, fill='red', outline='red')


if __name__ == '__main__':
    env = Env()
    #env.after(100, update)
    env.mainloop()

"""
def callback(event):
    print('click at %d, %d' % (event.x,event.y))
    click_x = event.x//UNIT
    click_y = event.y//UNIT
    cv.create_rectangle(10, 10, 110, 110, fill='red')

window.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

canvas = tk.Canvas(window,bg="white",height=window_height,width=window_width)

# create grids
for c in range(0, window_width, UNIT):
    x0, y0, x1, y1 = c, 0, c, window_height
    canvas.create_line(x0, y0, x1, y1)
for r in range(0, window_height, UNIT):
    x0, y0, x1, y1 = 0, r, window_width, r
    canvas.create_line(x0, y0, x1, y1)

# create the agent and the fruits
agent_image = Image.open(interaction_images['agent'])
agent_image = agent_image.resize((50, 50), Image.ANTIALIAS)
agent_img = ImageTk.PhotoImage(agent_image)
agent = canvas.create_image(50,50,anchor="nw", image=agent_img)

fruit_image = Image.open(interaction_images['orange'])
fruit_image = fruit_image.resize((50, 50), Image.ANTIALIAS)
fruit_img = ImageTk.PhotoImage(fruit_image)
fruit = canvas.create_image(200,250,anchor="nw", image=fruit_img)
print(canvas.coords(fruit))
fruit_coords = list(map(int, canvas.coords(fruit)))

canvas.bind("<Button-1>", callback)

canvas.pack()



window.mainloop()
"""

