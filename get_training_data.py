import numpy as np
from PIL import ImageGrab
import cv2
import win32api as wapi
import time
import timeit
import os
import sys

        
        

def key_output():
   #output one-hot vector [up,down,left,right]
   output = [#1 if wapi.GetAsyncKeyState(0x57) else 0,
             #1 if wapi.GetAsyncKeyState(0x53) else 0,
             1 if wapi.GetAsyncKeyState(0x41) else 0,
             1 if wapi.GetAsyncKeyState(0x44) else 0]  
   return output
def get_training_data():
   time.sleep(3)
   training_data = []
   files = 1
   
   while(True):
     #get screen at the up left corner  
     printscreen =  np.array(ImageGrab.grab(bbox=(0,0,800,600)))
     printscreen = cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB)
     #resize to 160x120
     printscreen = cv2.resize(printscreen, (400,300))
     output = key_output()
     with open('D:\speed.txt') as f:
        speed = f.readline()
      
     if speed == '':
        #if training_data isn't empty
        
        if training_data:
           speed = training_data[-1][1][-1]
          
        else:
           speed = 0
           
      
     output.append(float(speed))         
     training_data.append([printscreen,output])
     print '{0}\r'.format('recording...' + str(len(training_data))),
     #press p to pause
     if wapi.GetAsyncKeyState(0x4C):
          sys.stdout.write("\033[K")
          print '{0}\r'.format('pausing...' + str(len(training_data))),
          time.sleep(1)
          while(True):
         
            if wapi.GetAsyncKeyState(0x4C):
               time.sleep(1)
               break
            if wapi.GetAsyncKeyState(0x2E):
               training_data = []
               
     if wapi.GetAsyncKeyState(0x1B):
           break
             
     #save as np file in every 500 frames    
     if len(training_data) == 500:
        print 
        print files
        filename = 'data'+str(files)
        np.save(filename,training_data)  
        training_data = []   
        files = files + 1
     cv2.waitKey(50)

os.system('cls')
get_training_data()
