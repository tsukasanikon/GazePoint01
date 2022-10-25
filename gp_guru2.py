import pyautogui as pag
import socket
import time 
import pandas as pd
import pygame
from pygame.locals import *
import sys
import math
import datetime
import numpy as np


# Host machine IP
HOST = '127.0.0.1'
# Gazepoint Port
PORT = 4242
ADDRESS = (HOST, PORT)

# gaze & target data save
gaze_data_list = []
target_data_list = []
time_list = []
dt_now = datetime.datetime.now()
dt= dt_now.isoformat()
name = dt[0:16]
name = name.replace(':','_')
name = name.replace('T','_')
gaze_file_name = 'guru_gaze_data_'+ name + '.csv'
target_file_name = 'guru_ball_data_'+ name + '.csv'
time_file_name = 'guru_time_'+ name + '.csv'


# set full screen size
width =pag.size().width
height = pag.size().height
SCREEN = Rect((0, 0,  width, height))
CENTER = [int(width/2), int(height/2)]

# ball size and circle 
BALL_SUF = 50
BALL_CNT= int(BALL_SUF/2)
INIT_BALL_RAD = 10
RADIUS = int(height/2) - 50
angle = 0
ang_ver = 0.4

class GazePointData():
    def __init__(self, ADDRESS, gaze_data_list, gaze_file_name, time_list,time_file_name):
        self.address= ADDRESS
        self.gaze_data_list = gaze_data_list
        self.gaze_file_name = gaze_file_name
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(self.address)
        self.flag_record  = 'OFF'
        self.time_list = time_list
        self.time_file_name= time_file_name

    def start_calib(self):
        #　Caliblation  start 
        config_list = [\
            'CALIBRATE_START',
            'CALIBRATE_SHOW',
        ]
        
        for c in config_list:
            print(f'first: {c}')
            msg = f'<SET ID="{c}" STATE="1" />\r\n'
            time.sleep(1)
            self.sock.send(msg.encode())
            message = self.sock.recv(4096).decode('utf-8')
            print(f'message: {message}')
        
                
    def prep_get_data(self):
        
        self.sock.send(str.encode('<GET ID="TIME_TICK_FREQUENCY" />\r\n'))
        rxdat = self.sock.recv(1024)    
        print(bytes.decode(rxdat))
        
        
        config_list = [\
            'ENABLE_SEND_COUNTER',
            'ENABLE_SEND_TIME',
            'ENABLE_SEND_TIME_TICK',
            'ENABLE_SEND_PUPILMM',
            'ENABLE_SEND_POG_FIX',
            'ENABLE_SEND_EYE_RIGHT',
            'ENABLE_SEND_EYE_LEFT',
            'ENABLE_SEND_BLINK',
            'ENABLE_SEND_DATA',
            ]
        
        for c in config_list:
            print(f'first: {c}')
            msg = f'<SET ID="{c}" STATE="1" />\r\n'
            #time.sleep(1)
            self.sock.send(msg.encode())
            message = self.sock.recv(4096).decode('utf-8')
            print(f'message: {message}')

    def get_data(self):
        if self.flag_record == 'ON':

            time_a= time.time() 
            rxdat = self.sock.recv(1024)    
            x =bytes.decode(rxdat)
            print(f'time: {time_a}')
            print(x)
            self.gaze_data_list.append(x)
            self.time_list.append(time_a)
        elif self.flag_record == 'OFF':
            pass    
        
    def save_data(self):
          
        df = pd.DataFrame(self.gaze_data_list)
        df.to_csv(self.gaze_file_name)
        print('Save_data')
        
        df_time = pd.DataFrame(self.time_list)
        df_time.to_csv(self.time_file_name)
        print('Save_time')
        

    def disconnect(sock):
        sock.close()
        print('END')
    
    def record_data(self):
        self.flag_record = 'ON'    
        
class Ball:

    def __init__(self, gp,  target_data_list, target_file_name, angle=angle, ang_ver=ang_ver):
        self.gp = gp
        self.target_file_name = target_file_name
        self.target_data_list = target_data_list
        self.status = "INIT"
        self.angle = angle
        self.ang_ver = ang_ver
        self.image = pygame.Surface((BALL_SUF, BALL_SUF))
        pygame.draw.circle(self.image, (255, 255, 255), (BALL_CNT, BALL_CNT), INIT_BALL_RAD)
        self.rect = self.image.get_rect()
        ### 丸の位置
        self.rect.centerx = round(math.cos(math.radians(angle))*RADIUS+CENTER[0]-INIT_BALL_RAD/2)
        self.rect.centery  = round(math.sin(math.radians(angle))*RADIUS+CENTER[1]-INIT_BALL_RAD/2)

        
    def start(self):
        self.status = "RUNNING"


    def update(self):
        if self.status == 'RUNNING': 
            self.rect.centerx = round(math.cos(math.radians(self.angle))*RADIUS+CENTER[0]-INIT_BALL_RAD/2)
            self.rect.centery  = round(math.sin(math.radians(self.angle))*RADIUS+CENTER[1]-INIT_BALL_RAD/2)
            self.angle += ang_ver
            if self.angle >= 360:
                self.angle = 0
            self.target_data_list.append((self.rect.centerx, self.rect.centery))
        
    def draw(self, screen):
        screen.blit(self.image, self.rect)
    
    
    def save_data(self):
        df = pd.DataFrame(self.target_data_list)
        df.to_csv(self.target_file_name)
        print('Save_data')



def main():

    '''初期設定'''
    pygame.init()
    screen = pygame.display.set_mode(SCREEN.size)
    pygame.display.set_caption('GURU')    
    clock = pygame.time.Clock()


    
    


    '''登場する人/物/背景の作成'''
    gp =  GazePointData(ADDRESS, gaze_data_list, gaze_file_name,time_list,time_file_name)      
    gp.start_calib()
    gp.prep_get_data()
        
    ball = Ball(gp,  target_data_list, target_file_name)
    
    while True:
        
        #'''画面(screen)をクリア'''
        screen.fill((0, 0, 0))
        

        #'''ゲームに登場する人/物/背景の位置Update'''

        ball.update()
        gp.get_data()
            
        #'''画面(screen)上に登場する人/物/背景を描画'''
        ball.draw(screen)
        

        #'''画面(screen)の実表示'''
        pygame.display.update()
        

        #'''イベント処理'''
        for event in pygame.event.get():
            if event.type == QUIT:
                gp.save_data()
                ball.save_data()
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:  # キーを押したとき
            # ESCキーならスクリプトを終了
                if event.key == K_ESCAPE:
                    gp.save_data()
                    ball.save_data()
                    pygame.quit()
                    sys.exit()
            if event.type == MOUSEBUTTONDOWN:
                gp.record_data()
                ball.start()



                
                
            
        #'''描画スピードの調整（FPS)'''
        clock.tick(60)

if __name__ == "__main__":
    main()