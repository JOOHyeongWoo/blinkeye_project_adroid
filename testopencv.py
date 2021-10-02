
from kivy.app import App
from kivy.uix.camera import Camera
import cv2
from kivy.uix.label import Label
import mediapipe as mp
import numpy as np
from keras.models import load_model
import os
from scipy.spatial import distance
import time


class MainApp(App):
    def build(self):
        
        #cam =Camera(play=True, resolution=(640,480))
        #return cam
        return Label(text='hello world')

if __name__ == '__main__':
    MainApp().run()


