from os import *
from tkinter import *
from tkinter import Menu
from tkinter import filedialog
from tkinter import messagebox as mbox
from sklearn.metrics import accuracy_score
from statistics import stdev
from profileManager import *
import tkinter as tk
import pyaudio
import Recording
import emotionProcessor
import scikit_network
import numpy as np
#import FileWav

# Hardcoded Variables 
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
wave_output_filename = "user_recording.wav"

# Code to create the window for the GUI.
class Application(Frame):
    def __init__(self,master):
        Frame.__init__(self,master)
        self.grid(column = 4, row = 10)
        self.create_widgets()
        self.recorder=Recording.Recording(wave_output_filename, CHANNELS, RATE, CHUNK)
        self.recordingtest = False
        #self.filewav=FileWav.FileWav(CHANNELS, RATE, CHUNK) # Add Kaew
        #self.processor=emotionProcessor.EmotionProcessor(wave_output_filename) ย้ายไปself.endAudio      
#-------------------------------------------------------------------------------------------------------------------------------------------------------
    # class widgets.. this is where the code for each off six buttons two lable's and two text boxes are held.
    # Grid not pack is used. grid= where is that button on label located in the frame.
    # each button has a command atribute that connects the button with a function that controls what the button does.  
    def create_widgets(self):        
        self.label = Label (self, text = "- - -  - - - - - - - - - - - - - - - - - - - - -",font=("Comic Sans MS", 10), width = 30)
        self.label.grid(column = 1, row = 1)
        self.label = Label (self, text = "- - - - - - - - - SELECT MODEL - - - - - - - - -",font=("Comic Sans MS", 10), width = 30)
        self.label.grid(column = 2, row = 1)
        self.label = Label (self, text = "- - -  - - - - - - - - - - - - - - - - - - - - -",font=("Comic Sans MS", 10), width = 30)
        self.label.grid(column = 3, row = 1)
        
        self.label = Label(self, text = " Data From (Open Folder .wav) :",font=("Comic Sans MS", 10), width = 30)
        self.label.grid(column = 1, row = 2, pady = 5)
        
        self.textData = StringVar()
        self.text = Entry(self, textvariable = self.textData,font=("Comic Sans MS", 10), width = 30)
        self.text.grid(column = 2, row = 2, pady = 5)
        
        self.openButton = Button(self, text = " Browser " , justify = "center", command = self.foldernameTrain,font=("Comic Sans MS", 10,"bold"),fg="white",bg="orange", width = 15)
        self.openButton.grid(column = 3, row = 2, pady = 5)
        
        self.label = Label(self, text = " Save To Data (.csv) :",font=("Comic Sans MS", 10), width = 30)
        self.label.grid(column = 1, row = 3, pady = 5)
        
        self.textSave = StringVar()
        self.text = Entry(self, textvariable = self.textSave,font=("Comic Sans MS", 10), width = 30)
        self.text.grid(column = 2, row = 3, pady = 5)
        
        self.openButton = Button(self, text = " Browser " , justify = "center", command = self.filenameTrain,font=("Comic Sans MS", 10,"bold"),fg="white",bg="green", width = 15)
        self.openButton.grid(column = 3, row = 3, pady = 5)
        
        self.openButton = Button(self, text = " Train Performance " , justify = "center", command = self.Train,font=("Comic Sans MS", 10,"bold"),fg="white",bg="red", width = 15)
        self.openButton.grid(column = 2, row = 4, pady = 5)
        
        self.label = Label (self, text = "- - -  - - - - - - - - - - - - - - - - - - - - -",font=("Comic Sans MS", 10), width = 30)
        self.label.grid(column = 1, row = 5)
        self.label = Label (self, text = "- - - - - - - - - - RESULT - - - - - - - - - -",font=("Comic Sans MS", 10), width = 30)
        self.label.grid(column = 2, row = 5)
        self.label = Label (self, text = "- - -  - - - - - - - - - - - - - - - - - - - - -",font=("Comic Sans MS", 10), width = 30)
        self.label.grid(column = 3, row = 5)
            
        self.label = Label (self, text = " Best Parameter :",font=("Comic Sans MS", 10), width = 30)
        self.label.grid(column = 1, row = 6, pady = 5)
        
        self.textParameter = StringVar()
        self.text = Entry(self, textvariable = self.textParameter,font=("Comic Sans MS", 10), width = 20)
        self.text.grid(column = 2, row = 6, pady = 5)
                
        self.label = Label (self, text = " Best Performance :",font=("Comic Sans MS", 10), width = 30)
        self.label.grid(column = 1, row = 7, pady = 5)
        
        self.textPerformance = StringVar()
        self.text = Entry(self, textvariable = self.textPerformance,font=("Comic Sans MS", 10), width = 20)
        self.text.grid(column = 2, row = 7, pady = 5)
        
#-------------------------------------------------------------------------------------------------------------------------------------------------------    
    # Add From Kaew
    def Train(self):   
        self.userName = ""
        path = self.foldername        
        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            for file in f:
                if '.wav' in file:
                    files.append(os.path.join(r, file))
                    #music_audio = os.path.basename(self.filename)
                    for f in files:
                        self.processor=emotionProcessor.EmotionProcessor(f)
                        self.recordingtest= True
                        if(self.recordingtest == True):
                            # Call the method to get the audio metrics
                            self.audio_metrics = self.processor.collectMetrics()
                            # Create a user profile object using the entered user name
                            self.user_profile = profileManager(self.userName, self.model)
                            # Access the profile for the given user
                            self.user_profile.accessProfile()
                            #Get the prediction from the scikit network
                            self.predicted = scikit_network.compare_train(self.audio_metrics, self.user_profile)
                            print(self.predicted[0])
        self.textParameter.set(self.predicted[1])
        self.textPerformance.set(self.predicted[2])
#-------------------------------------------------------------------------------------------------------------------------------------------------------    
    # Add From Kaew    (initialdir="C:/Users/K/Project/sound/ ที่อยู่เสียงเครื่องตัวเอง) ยังใช้งานไม่ได้
    def filenameTrain(self):
        self.filename = filedialog.askopenfilename(initialdir="C:/Users/K/Desktop/", title="Choose Select file",filetypes=(("csv files", "*.csv"), ("all files", "*.*")))
        self.model = os.path.basename(self.filename)
        self.textSave.set(self.model)
#-------------------------------------------------------------------------------------------------------------------------------------------------------   
    # Add From Kaew 
    def foldernameTrain(self):
        self.foldername = filedialog.askdirectory(initialdir="C:/Users/K/Desktop/.../", title="Choose Select file")
        self.textData.set(self.foldername)
#-------------------------------------------------------------------------------------------------------------------------------------------------------		
# Modify root window.
root = Tk()
root.title("Emotional classification from conversation voices")

# The size of the whole frame.
root.geometry ("730x265")

# Background for the whole GUI
app = Application(root)

#kick off the event loop
root.mainloop()



