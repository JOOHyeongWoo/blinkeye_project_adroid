from kivy.app import App
from kivy.uix.button import Button

fontname= 'NanumGothic.ttf'
class TestApp(App):
    def build(self):
        return Button(text='안녕', font_name=fontname)

TestApp().run()