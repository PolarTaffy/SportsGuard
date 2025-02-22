from taipy import Gui
from taipy.gui import invoke_long_callback
import taipy.gui.builder as tgb
import numpy as np
import pandas as pd
import time


messageFeed = list()

#Starting Messages
messageFeed.append("Welcome to the Live Sports Injury Detector! \n")
messageFeed.append("Please wait for the live feed to start... \n")
messageFeed.append("The system is currently monitoring the game... \n")

# Function to read the latest result_frame
def get_latest_frame():
    try:
        with open('result_frame.txt', 'r') as f:
            return f.read()
    except FileNotFoundError:
        return ""

page = """
# Live Sports Injury Detector

<|{messageFeed}|table|>

<|img src="data:image/jpeg;base64,{latest_frame}" width="640" height="480"|>
"""



def injury_msg(playerName, time, action):
    string = "At " + time + ", " + playerName + " has " + action + ". \n Alerting referees..."
    messageFeed.append(string)
    #Gui.update("messageFeed", messageFeed)
    Gui.update()

def update_frame():
    while True:
        latest_frame = get_latest_frame()
        Gui.update("latest_frame", latest_frame)
        time.sleep(1)

if __name__ == "__main__":
    Gui(page).run(title="Injury Detector", use_reloader=True)