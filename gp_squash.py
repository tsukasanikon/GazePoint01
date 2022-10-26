import pyautogui as pag
import socket
import time 
import pandas as pd
import pygame
from pygame.locals import *
import sys
import math
import datetime

# Host machine IP
HOST = '127.0.0.1'
# Gazepoint Port
PORT = 4242
ADDRESS = (HOST, PORT)

# gaze & target data 
gaze_data_list = []
target_data_list = []
time_list = []
dt_now = datetime.datetime.now()
dt= dt_now.isoformat()
name = dt[0:16]
name = name.replace(':','_')
name = name.replace('T','_')
gaze_file_name = 'squash_gaze_data_'+ name + '.csv'
target_file_name = 'squash_target_data_'+ name + '.csv'
time_file_name = 'squash_time_'+ name + '.csv'


# # gaze data 
# data_list = []
# csv_file_name = 'data_test_2.csv'
# get screen size
width =pag.size().width
height = pag.size().height
SCREEN = Rect((0, 0,  width, height))

class GazePointData():
    def __init__(self, ADDRESS, gaze_data_list, gaze_file_name, time_list,
                 time_file_name):
        self.address= ADDRESS
        self.data_list = gaze_data_list
        self.csv_file_name = gaze_file_name
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(self.address)
        self.flag_record = 'OFF'
        self.time_list = time_list
        self.time_file_name= time_file_name

    # テスト前のカリブレーション実施する
    def start_calib(self):
        #　Caliblation  start 
        config_list = [\
            'CALIBRATE_DELAY',
            'CALIBRATE_START',
            'CALIBRATE_SHOW',
        ]
        
        STATE_list = [\
            5,
            1,
            1,
        ]

        
        for c , s in zip(config_list, STATE_list):
            msg = f'<SET ID="{c}" STATE="{s}" />\r\n'
            time.sleep(1)
            self.sock.send(msg.encode())
            message = self.sock.recv(4096).decode('utf-8')
            print(f'message: {message}')
    
    # 実験における測定データを指定（OPEN GAZE API から選ぶ）            
    def prep_get_data(self):
        config_list = [\
            'ENABLE_SEND_COUNTER',
            'ENABLE_SEND_TIME',
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

    #パケットでのデータ抽出を実施、リストデータに追加
    def get_data(self):
        if self.flag_record == 'ON':
            time_a= time.time() 
            rxdat = self.sock.recv(1024)    
            x =bytes.decode(rxdat)
            print(f'time: {time_a}')
            print(x)
            self.data_list.append(x)
            self.time_list.append(time_a)
        else:
            pass    
    # リスト型測定データをPandasのDataFrameに変換し、csvファイルにて保存    
    def save_data(self):
        df = pd.DataFrame(self.data_list)
        df.to_csv(self.csv_file_name)
        print('Save_data')
        
        df_time = pd.DataFrame(self.time_list)
        df_time.to_csv(self.time_file_name)
        print('Save_time: Success!')
        
    # ソケットス通信をdisconnectする
    def disconnect(self):
        self.sock.close()
        print('END')
    
    # 測定データを記録するか否かのフラグ    
    def record_data(self):
        self.flag_record = 'ON'
            




class Paddle:
    
    def __init__(self):
        self.image = pygame.Surface((100, 10))
        self.image.fill((255, 255, 255))
        self.rect = self.image.get_rect()
        self.rect.center = (SCREEN.centerx, SCREEN.bottom - 50)
        
    def update(self):
        self.rect.centerx = pygame.mouse.get_pos()[0]
        self.rect.clamp_ip(SCREEN)  
        
    def draw(self, screen):
        screen.blit(self.image, self.rect)
        
class Ball:

    def __init__(self,pad, target_data_list, target_file_name):
        self.image = pygame.Surface((20, 20))
        pygame.draw.circle(self.image, (255, 0, 0), (10, 10), 10)
        self.rect = self.image.get_rect()
        self.pad = pad
        self.rect.centerx = self.pad.rect.centerx
        self.rect.bottom = self.pad.rect.top
        self.dx, self.dy = 3, -4
        self.status = "INIT"
        self.data_list = target_data_list
        self.csv_file_name = target_file_name
        
        
    def start(self):
        self.status = "RUNNING"
    
        
    def update(self):
        if self.status == "INIT":
            self.rect.centerx = self.pad.rect.centerx
            self.rect.bottom = self.pad.rect.top
            return
        old_rect = self.rect.copy()
        self.rect.move_ip(self.dx, self.dy)
        if self.rect.colliderect(self.pad.rect):
            if self.pad.rect.left >= old_rect.right:
                self.rect.right = self.pad.rect.left
                self.dx = - self.dx
            
            elif self.pad.rect.right <= old_rect.left:
                self.rect.left = self.pad.rect.right
                self.dx = - self.dx
            elif self.pad.rect.top >= old_rect.bottom:
                self.rect.bottom = self.pad.rect.top
                x = self.rect.centerx - self.pad.rect.left
                y = - 100 * x / self.pad.rect.width + 145
                self.dx = 5 * math.cos(math.radians(y))
                self.dy = - 5 * math.sin(math.radians(y))
            else:
                self.rect.top = self.pad.rect.bottom
                self.dy = - self.dy
        if self.rect.left < SCREEN.left or self.rect.right > SCREEN.right:
            self.dx = -self.dx
        if self.rect.top < SCREEN.top:
            self.dy = -self.dy
        if self.rect.bottom > SCREEN.bottom:
            self.status = "INIT"
        self.rect.clamp_ip(SCREEN)
        cntx =  self.rect.centerx
        cnty = self.rect.centery
        cntxy = (cntx, cnty)
        self.data_list.append(cntxy)
        
    def draw(self, screen):
        screen.blit(self.image, self.rect)
        
        
    def save_data(self):
        df = pd.DataFrame(self.data_list)
        df.to_csv(self.csv_file_name)
        print('Save_data: Success!')


def main():
    
    gp =  GazePointData(ADDRESS, gaze_data_list, gaze_file_name,time_list,
                 time_file_name)
    gp.start_calib()
    gp.prep_get_data()
    
    '''初期設定'''
    pygame.init()
    screen = pygame.display.set_mode(SCREEN.size)
    pygame.display.set_caption('SQUASH GAME')

    clock = pygame.time.Clock()
    


    '''登場する人/物/背景の作成'''       
        
    pad = Paddle()
    ball = Ball(pad, target_data_list, target_file_name)
    
    while True:
        
        #'''画面(screen)をクリア'''
        screen.fill((0, 0, 0))
        

        #'''ゲームに登場する人/物/背景の位置Update'''
        pad.update()
        ball.update()
        gp.get_data()

            
        #'''画面(screen)上に登場する人/物/背景を描画'''
        pad.draw(screen)
        ball.draw(screen)
       

        #'''画面(screen)の実表示'''
        pygame.display.update()
        pygame.display.flip()
        

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
                ball.start()
                gp.record_data()

                
                
            
        #'''描画スピードの調整（FPS)'''
        clock.tick(200)

if __name__ == "__main__":
    main()