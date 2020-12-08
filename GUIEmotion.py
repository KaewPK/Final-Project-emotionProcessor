from os import *
from tkinter import *
from tkinter import Menu
from tkinter import filedialog
from tkinter import messagebox as mbox
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
        self.grid(column = 3, row = 11)
        self.create_widgets()
        self.recorder=Recording.Recording(wave_output_filename, CHANNELS, RATE, CHUNK)
        self.recordingtest = False   
#-------------------------------------------------------------------------------------------------------------------------------------------------------
    # class widgets.. this is where the code for each off six buttons two lable's and two text boxes are held.
    # Grid not pack is used. grid= where is that button on label located in the frame.
    # each button has a command atribute that connects the button with a function that controls what the button does.  
    def create_widgets(self):    
        self.label = Label (self, text = "- - -  - - - - - - - - - - - - - - - - - - SELECT",font=("Comic Sans MS", 10), width = 30)
        self.label.grid(column = 1, row = 1)
        self.label = Label (self, text = "MODEL FROM - - - - - - - - - - - - - - - - -",font=("Comic Sans MS", 10), width = 30)
        self.label.grid(column = 2, row = 1)
        
        self.label = Label(self, text = " MODEL FROM :",font=("Comic Sans MS", 10), width = 30)
        self.label.grid(column = 1, row = 2, pady = 5)
        
        self.textDatamodel = StringVar()
        self.text = Entry(self, textvariable = self.textDatamodel,font=("Comic Sans MS", 10), width = 20)
        self.text.grid(column = 2, row = 2, pady = 5)
        
        self.openButton = Button(self, text = " Browser " , justify = "center", command = self.selectModel,font=("Comic Sans MS", 10,"bold"),fg="white",bg="orange", width = 15)
        self.openButton.grid(column = 2, row = 3, pady = 5)
        
        self.label = Label (self, text = "- - - - - - - - - - - - - - - - - - - - - - AUDIO",font=("Comic Sans MS", 10), width = 30)
        self.label.grid(column = 1, row = 4)
        self.label = Label (self, text = "RECORDER - - - - - - - - - - - - - - - - - - -",font=("Comic Sans MS", 10), width = 30)
        self.label.grid(column = 2, row = 4)
        
        # Label for the output below "user Id or name".
        self.label = Label(self, text = " Record User Name :",font=("Comic Sans MS", 10), width = 30)
        self.label.grid(column = 1, row = 5, pady = 5)
    
        #output for the User id
        self.user = StringVar()
        self.txt = Entry(self, textvariable = self.user,font=("Comic Sans MS", 10), width = 20)
        self.txt.grid(column = 2, row = 5, pady = 5)
        
        #Start button, which begins the recording process.
        self.startButton = Button(self, text = " Start : recording" , justify = "center", command = self.recordAudio,font=("Comic Sans MS", 10,"bold"),fg="white",bg="green", width = 15)
        self.startButton.grid(column = 1,row = 6, pady = 5)
        
        # Stop button, which ends the recording process and begins the process of metric extraction/comparison.
        self.stopButton = Button(self, text = " Stop : recording" , justify = "center", command = self.endAudio,font=("Comic Sans MS", 10,"bold"),fg="white",bg="red", width = 15)
        self.stopButton.grid(column = 2, row = 6, pady = 5)
        
        self.label = Label (self, text = "- - - - - - - - - - - - - - - - - - - - - - READ",font=("Comic Sans MS", 10), width = 30)
        self.label.grid(column = 1, row = 7)
        self.label = Label (self, text = "OPENFILE .wav - - - - - - - - - - - - - - - - -",font=("Comic Sans MS", 10), width = 30)
        self.label.grid(column = 2, row = 7)
        
        self.label = Label(self, text = " Open File .wav ",font=("Comic Sans MS", 10), width = 30)
        self.label.grid(column = 1, row = 8, pady = 5)
        
        self.label = Label(self, text = " From My Computer ",font=("Comic Sans MS", 10), width = 30)
        self.label.grid(column = 1, row = 9, pady = 5)
        
        self.textfile = StringVar()
        self.text = Entry(self, textvariable = self.textfile,font=("Comic Sans MS", 10), width = 20)
        self.text.grid(column = 2, row = 8, pady = 5)
        
        self.openButton = Button(self, text = " Browser & Analysis " , justify = "center", command = self.filenameAudio,font=("Comic Sans MS", 10,"bold"),fg="white",bg="blue", width = 15)
        self.openButton.grid(column = 2, row = 9, pady = 5)
        
        self.label = Label (self, text = "- - - - - - - - - - - - - - - - - - - - - - - - - - -",font=("Comic Sans MS", 10), width = 30)
        self.label.grid(column = 1, row = 10)
        self.label = Label (self, text = "RESULT - - - - - - - - - - - - - - - - - - - - - -",font=("Comic Sans MS", 10), width = 30)
        self.label.grid(column = 2, row = 10)
        
        # Label for the GUI that Says " predicted emotion".        
        self.label = Label (self, text = " Predicted Emotion :",font=("Comic Sans MS", 10), width = 30)
        self.label.grid(column = 1, row = 11, pady = 5)
        
        # Output field for the label where the predicted emotion will be added when the knn processed the audio matrics hopefully correctly. 
        self.emotionalPrediction = StringVar()
        self.text = Entry(self, textvariable = self.emotionalPrediction,font=("Comic Sans MS", 10), width = 20)
        self.text.grid(column = 2, row = 11, pady = 5)
#-------------------------------------------------------------------------------------------------------------------------------------------------------    
    # Add From Kaew    (initialdir="C:/Users/K/Project/sound/ ที่อยู่เสียงเครื่องตัวเอง) ยังใช้งานไม่ได้
    def selectModel(self):
        self.modelname = filedialog.askopenfilename(initialdir="C:/Users/K/Desktop/", title="Choose Select file",filetypes=(("csv files", "*.csv"), ("all files", "*.*")))
        self.model = os.path.basename(self.modelname)
        self.textDatamodel.set(self.model)
#-------------------------------------------------------------------------------------------------------------------------------------------------------    
    # Add From Kaew    (initialdir="C:/Users/K/Project/sound/ ที่อยู่เสียงเครื่องตัวเอง) ยังใช้งานไม่ได้
    def filenameAudio(self):
        self.filename = filedialog.askopenfilename(initialdir="C:/Users/K/Desktop/", title="Choose Select file",filetypes=(("wav files", "*.wav"), ("all files", "*.*")))
        self.file = os.path.basename(self.filename)
        self.textfile.set(self.file)
        #music_audio = os.path.basename(self.filename)
        self.processor=emotionProcessor.EmotionProcessor(self.filename)
        #self.recorder=Recording.Recording(self.filename, CHANNELS, RATE, CHUNK)
        #self.recorder.startAudio()
        self.emotionalPrediction.set("Analysis..")
        self.recordingtest= True
        if(self.recordingtest == True):
            #Stop recording audio
            #self.recorder.stopAudio()
            #Get the entered user name from the entry box
            self.userName = self.user.get()
            # Call the method to get the audio metrics
            self.audio_metrics = self.processor.collectMetrics()
            # Create a user profile object using the entered user name
            self.user_profile = profileManager(self.userName, self.model)
            # Access the profile for the given user
            self.user_profile.accessProfile()
            #Get the prediction from the scikit network
            self.predicted = scikit_network.compare_new(self.audio_metrics, self.user_profile)
            self.emotionalPrediction.set(self.predicted[0])
            
            #yes no box asking if returned emotion was correct
            question = ("Was predicted emotion " + self.predicted[0] + " correct?")
            if mbox.askyesno("Emotion Prediction Assessment" , question):
                self.user_profile.addtoProfile(self.audio_metrics, self.predicted[0])
                self.recordingtest = False
            else:
                newtab = Tk()
                newtab.title("Wrong Emotion Correction")
                newtab.geometry("300x158")


                self.correction = StringVar(newtab)
                self.correction.set("Normal")

                emotions = OptionMenu(newtab, self.correction, "Angry", "Normal")
                emotions.grid(row = 0, column = 0)

                submitButton = Button(newtab, text = "Submit Emotion" , justify = "center", command = lambda:[self.submit(), newtab.destroy()], bg = "lightgray")
                submitButton.grid(row = 1, column = 0)

                newtab.mainloop()
        else:
            mbox.showerror("Incorrect button press!", "You must be recording to stop. Please start/restart recording.")
        return self
#-------------------------------------------------------------------------------------------------------------------------------------------------------   
    def recordAudio(self):
        if(self.recordingtest == True):
	        print("Already Recording!")
        else:
	        self.recorder=Recording.Recording(wave_output_filename, CHANNELS, RATE,CHUNK)
	        self.recorder.startAudio()
	        self.emotionalPrediction.set("Analysis..")
	        self.recordingtest= True
        return self
    # End audio also needs a popup button.
#-------------------------------------------------------------------------------------------------------------------------------------------------------    
    def endAudio(self):
        self.processor=emotionProcessor.EmotionProcessor(wave_output_filename)
        if(self.recordingtest == True):
            #Stop recording audio
            self.recorder.stopAudio()
            #Set the box containing the emotional prediction to be blank
            self.emotionalPrediction.set("Done Recording.")
            #Get the entered user name from the entry box
            self.userName = self.user.get()
            # Call the method to get the audio metrics
            self.audio_metrics = self.processor.collectMetrics()
            # Create a user profile object using the entered user name
            self.user_profile = profileManager(self.userName, self.model)
            # Access the profile for the given user
            self.user_profile.accessProfile()
            #Get the prediction from the scikit network
            self.predicted = scikit_network.compare_new(self.audio_metrics, self.user_profile)
            self.emotionalPrediction.set(self.predicted[0])
            #os.remove("user_recording.wav")

            #yes no box asking if returned emotion was correct
            question = ("Was predicted emotion " + self.predicted[0] + " correct?")
            if mbox.askyesno("Emotion Prediction Assessment" , question):
                self.user_profile.addtoProfile(self.audio_metrics, self.predicted[0])
                self.recordingtest = False
            else:
                newtab = Tk()
                newtab.title("Wrong Emotion Correction")
                newtab.geometry("300x158")


                self.correction = StringVar(newtab)
                self.correction.set("Normal")

                emotions = OptionMenu(newtab, self.correction, "Angry", "Normal")
                emotions.grid(row = 0, column = 0)

                submitButton = Button(newtab, text = "Submit Emotion" , justify = "center", command = lambda:[self.submit(), newtab.destroy()], bg = "lightgray")
                submitButton.grid(row = 1, column = 0)

                newtab.mainloop()
                
        else:
            mbox.showerror("Incorrect button press!", "You must be recording to stop. Please start/restart recording.")
        return self
#-------------------------------------------------------------------------------------------------------------------------------------------------------        
    def submit(self):
        self.predicted = self.correction.get()
        self.user_profile.addtoProfile(self.audio_metrics, self.predicted)
        self.recordingtest = False
        # Add Kaew
        self.emotionalPrediction.set(self.predicted)
#-------------------------------------------------------------------------------------------------------------------------------------------------------		
# Modify root window.
root = Tk()
root.title("Emotional classification from conversation voices")

# The size of the whole frame.
root.geometry ("490x370")

# Background for the whole GUI
app = Application(root)

#kick off the event loop
root.mainloop()



