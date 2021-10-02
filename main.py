#from re import M
import kivy
#kivy.require('1.0.6') # replace with your current kivy version !

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.slider import Slider
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import os
from scipy.spatial import distance
import time
from testmedia import Eye_check

class Screen(GridLayout):
    def __init__(self, **kwargs):
        super(Screen, self).__init__(**kwargs)
        self.rows =5

        
        self.check_blink_num= BoxLayout(orientation='horizontal')
        self.add_widget(self.check_blink_num)
        self.check_blink_num.add_widget(Label(text='check blink num'))
        self.check_blink_num.add_widget(Label(text='20'))
        
        self.slider= Slider(orientation= 'horizontal', value_track=True, value_track_color=(1,0,0,1))
        self.slider.bind(value= self.on_slider_changed)
        self.add_widget(self.slider)

        self.min_checkbox= BoxLayout(orientation='horizontal')
        self.add_widget(self.min_checkbox)

        self.min_checkbox.one_min = CheckBox()
        self.min_checkbox.one_min.bind(active= self.on_checkbox)
        self.min_checkbox.add_widget(self.min_checkbox.one_min)

        self.min_checkbox.three_min = CheckBox()
        self.min_checkbox.three_min.bind(active= self.on_checkbox)
        self.min_checkbox.add_widget(self.min_checkbox.three_min)

        
        

        self.button= Button(text="Start",font_size=40 , padding=[90,40] )
        self.button.bind(on_press= self.on_pressed)
        self.add_widget(self.button)


    def on_pressed(self, instance):
        print("pressed button")

    def on_checkbox(self, instance, value):
        if value:
            print("checked")
        else:
            print("unchecke")

    def on_slider_changed(self,instance,value):
        print(value)


class Upper_bar(BoxLayout):
    pass

class slider_bar(BoxLayout):
    pass

class Check_minute(BoxLayout):
    pass

class Notification_bar(BoxLayout):
    pass
       
class Button_bar(BoxLayout):
    pass


class MainApp(App):
    pass


class MyApp(App):

    def build(self):
        Eye_check.checkblink()
        return  Screen()


if __name__ == '__main__':
    #MyApp().run()
    MainApp().run()