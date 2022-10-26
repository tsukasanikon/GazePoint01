#必要なライブラリーをインポート
#%matplotlib inline
import japanize_matplotlib
import numpy as np
import cv2
import math
import pandas as pd
import copy 
import os
import glob
import math
import time
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import pywt 
import scipy.interpolate, scipy.optimize


#パラメータの設定
#単位ピクセル当たりの実際の長さ（カメラレンズからの値で算出したが
px = 35000/1246
py =28000/1008
print("x‗ピクセルの長さ: {:.3f}".format(px))
print("y‗ピクセルの長さ: {:.3f}".format(py))

#npz データをDataFrameに変換する関数
#npzファイルを指定する
#file_dir = './micro_test_10_8_1.npz'

def load_data(file_dir):
    new_array= np.load(file_dir)
    time=new_array['time']
    diameter =new_array['diameter']
    center = new_array['center']
    df= pd.DataFrame([time, diameter, center])
    df= df.T
    df=df.rename(columns={0: 'Time', 1: 'Diameter', 2: 'Center'})
    #df= df.dropna(how='any')
    #df= pd.DataFrame([diameter, center])
    #df= df.T
    #df=df.rename(columns={0: 'Diameter', 1: 'Center'})
    #df= df.dropna(how='any')
    return df


#DataFrame データからxy座標データに変換
def make_xy_data(df):
    t= df['Time'].tolist()
    x=df['Center'].tolist()
    y=np.squeeze(x)
    x_a=y[:,0]
    y_a = y[:,1]
    xy_df = pd.DataFrame([t, x_a, y_a]).T
    xy_df=xy_df.rename(columns = {0:'time', 1:'x', 2:'y'})
    return xy_df

#xy_dfデータから速度Vxytを求める関数
def calc_xy_deg(xy_df):
    x_v = xy_df['x'].values
    y_v = xy_df['y'].values 
    mode_x = statistics.mode(x_v)
    mode_y = statistics.mode(y_v)
    deg_xts = []    
    deg_yts = []
    t_xyts = []
    #dev_xyts =[]
    num_dev = 0
    for i in range(len(x_v)):
            dif_x = x_v - mode_x
            dif_y = y_v - mode_y
            deg_xt = math.degrees(math.asin(dif_x/12000))
            deg_yt = math.degrees(math.asin(dif_y/12000))
            deg_xts.append(deg_xt)            
            deg_yts.append(deg_yt)
            t_xyts.append(i)
            #df = pd.DataFrame([t_xyts, v_xyts]).T
            #df = df.rename(columns={0: 'time', 1: 'v_xy_deg'})
    return deg_xts, deg_yts






#xy_dfデータから速度Vxytを求める関数
def calc_velocity(xy_df):
    x_v = xy_df['x'].values
    y_v = xy_df['y'].values 
    t= 1
    v_xyts = []
    for i in range(len(x_v)):
        if i > 1 and i < (len(x_v) - 2 ):
            v_xt = (x_v[i+2] + x_v[i +1] - x_v[i-1] - x_v[i -2])*px/ 6*t
            v_yt = (y_v[i+2] + y_v[i +1] - y_v[i-1] - y_v[i -2])*py/ 6*t
            v_xyt = math.sqrt(v_xt**2 + v_yt**2)
            v_xyts.append(v_xyt)
    return v_xyts

#xy_dfデータから速度Vxytを求める関数
def calc_velocity_2(xy_df):
    x_v = xy_df['x'].values
    y_v = xy_df['y'].values 
    t_v = xy_df['time'].values
    v_xyts = []
    t_xyts = []
    dev_xyts =[]
    num_dev = 0
    for i in range(len(x_v)):
        if i > 1 and i < (len(x_v) - 2 ):
            v_tt = t_v[i+2] + t_v[i+1] - t_v[i-1] - t_v[i-2]
            if v_tt != 6: x += 1 
            v_xt = (x_v[i+2] + x_v[i +1] - x_v[i-1] - x_v[i -2])*px/ v_tt

            v_yt = (y_v[i+2] + y_v[i +1] - y_v[i-1] - y_v[i -2])*py/ v_tt

            v_xyt = math.sqrt(v_xt**2 + v_yt**2)
            v_xyts.append(v_xyt)
            t_xyts.append(i-2)
    df = pd.DataFrame([t_xyts, v_xyts]).T
    df = df.rename(columns={0: 'time', 1: 'v_xy'})
     
    return df

#xy_dfデータから速度Vx とVyを求める関数
def calc_velocity_x_y_deg(xy_df):
    x_v = xy_df['x'].values
    y_v = xy_df['y'].values 
    t_v = xy_df['time'].values
    deg_xts = [0,0,]
    deg_yts = [0,0,]
    t_xyts = [0,1,]
    #dev_xyts =[]
    num_dev = 0
    for i in range(len(x_v)):
        if i > 1 and i < (len(x_v) - 2 ):  
            v_tt = t_v[i+2] + t_v[i+1] - t_v[i-1] - t_v[i-2]
            v_xt = abs(x_v[i+2] + x_v[i +1] - x_v[i-1] - x_v[i -2])*px/ (v_tt)
            v_yt = abs(y_v[i+2] + y_v[i +1] - y_v[i-1] - y_v[i -2])*py/ (v_tt)
            deg_xt = math.degrees(math.asin(v_xt/12000))
            deg_yt = math.degrees(math.asin(v_yt/12000))            
            deg_xts.append(deg_xt)
            deg_yts.append(deg_yt)            
            t_xyts.append(i)
            #df = pd.DataFrame([t_xyts, v_xyts]).T
            #df = df.rename(columns={0: 'time', 1: 'v_xy_deg'})
    for j in range(2):
        deg_xts.append(0)
        deg_yts.append(0)        
        t_xyts.append(len(x_v)-2 + j) 
    return deg_xts, deg_yts, t_xyts


# #xy_dfデータから速度Vxytを求める関数
# def calc_velocity_deg(xy_df):
#     x_v = xy_df['x'].values
#     y_v = xy_df['y'].values 
#     t_v = xy_df['time'].values
#     deg_xyts = [0,0,]
#     t_xyts = [0,1,]
#     #dev_xyts =[]
#     num_dev = 0
#     for i in range(len(x_v)):
#         if i > 1 and i < (len(x_v) - 2 ):
            
#             v_tt = t_v[i+2] + t_v[i+1] - t_v[i-1] - t_v[i-2]
#             v_xt = (x_v[i+2] + x_v[i +1] - x_v[i-1] - x_v[i -2])*px/ (v_tt)
#             v_yt = (y_v[i+2] + y_v[i +1] - y_v[i-1] - y_v[i -2])*py/ (v_tt)
#             v_xyt = math.sqrt(v_xt**2 + v_yt**2)
#             print(f'angle1: {v_xyt/12000}')
#             deg_xyt = math.degrees(math.asin(v_xyt/12000))
#             deg_xyts.append(deg_xyt)
#             t_xyts.append(i)
#             #df = pd.DataFrame([t_xyts, v_xyts]).T
#             #df = df.rename(columns={0: 'time', 1: 'v_xy_deg'})
#     for j in range(2):
#         deg_xyts.append(0)
#         t_xyts.append(len(x_v)-2 + j) 
#     return deg_xyts, t_xyts

def to_micron(xy_df):
    global px 
    global py
    x_px = xy_df['deno_x'].values
    y_px = xy_df['deno_y'].values
    x_micr = x_px *px
    y_micr = y_px * py
    xy_df['x_micr'] = x_micr
    xy_df['y_micr'] = y_micr
    return xy_df

def to_velo(xy_df):
    int_t = 6
    x_micr = xy_df['x_micr'].values
    y_micr =xy_df['y_micr'].values
    t_v =xy_df['time'].values
    list_v_x = [0,0,0,0,]
    list_v_y = [0,0,0,0,]
    list_v_xy = [0,0,0,0,]
    for i in range(len(x_micr)):
        if i > 1 and i < (len(x_micr) - 2 ):
            
            v_tt = (t_v[i+2] + t_v[i+1] - t_v[i-1] - t_v[i-2])
            v_x = (x_micr[i+2] + x_micr[i+1] - x_micr[i-1] - x_micr[i-2])/ v_tt
            v_y = (y_micr[i+2] + y_micr[i +1] - y_micr[i-1] - y_micr[i -2])/ v_tt
            list_v_x.append(v_x)
            list_v_y.append(v_y)
            v_xy = math.sqrt(v_x**2 + v_y**2)
            list_v_xy.append(v_xy)
    print(f'v_y: {len(list_v_y)}')        
    print(f'v_x: {len(list_v_x)}')
    print(f'v_xy: {len(list_v_xy)}')
    xy_df['v_x'] = list_v_x
    xy_df['v_y'] = list_v_y
    xy_df['v_xy'] =list_v_xy   
    return xy_df

def to_deg(xy_df):
    v_xy = xy_df['v_xy'].values
    deg_xyt = math.degrees(math.asin(v_xyt/12000))
    print(f'deg_xyt: {deg_xyt}')
    deg_xyts.append(deg_xyt)
    t_xyts.append(i)

#xy_dfデータから速度Vxytを求める関数
def calc_velocity(xy_df):
    x_v = xy_df['x'].values
    y_v = xy_df['y'].values 
    t= 1
    v_xyts = []
    for i in range(len(x_v)):
        if i > 1 and i < (len(x_v) - 2 ):
            v_xt = (x_v[i+2] + x_v[i +1] - x_v[i-1] - x_v[i -2])*px/ 6*t
            v_yt = (y_v[i+2] + y_v[i +1] - y_v[i-1] - y_v[i -2])*py/ 6*t
            v_xyt = math.sqrt(v_xt**2 + v_yt**2)
            v_xyts.append(v_xyt)
    return v_xyts    
    

#deno_x と　deno_yデータから速度Vxytを求める関数
def calc_velocity_deg(xy_df):
    x_v = xy_df['deno_x'].values
    y_v = xy_df['deno_y'].values 
    t_v = xy_df['time'].values
    deg_xyts = [0,0,]
    t_xyts = [0,1,]
    #dev_xyts =[]
    num_dev = 0
    for i in range(len(x_v)):
        if i > 1 and i < (len(x_v) - 2 ):
            
            v_tt = (t_v[i+2] + t_v[i+1] - t_v[i-1] - t_v[i-2])
            v_xt = (x_v[i+2] + x_v[i +1] - x_v[i-1] - x_v[i -2])*px/ (v_tt)
            print(f'v_xt:{v_xt}')
            v_yt = (y_v[i+2] + y_v[i +1] - y_v[i-1] - y_v[i -2])*py/ (v_tt)
            v_xyt = math.sqrt(v_xt**2 + v_yt**2)
            print(f'v_xyt: {v_xyt}')
            print(f'angle: {v_xyt/12000}')
            deg_xyt = math.degrees(math.asin(v_xyt/12000))
            print(f'deg_xyt: {deg_xyt}')
            deg_xyts.append(deg_xyt)
            t_xyts.append(i)
            #df = pd.DataFrame([t_xyts, v_xyts]).T
            #df = df.rename(columns={0: 'time', 1: 'v_xy_deg'})
    for j in range(2):
        deg_xyts.append(0)
        t_xyts.append(len(x_v)-2 + j) 
    return deg_xyts, t_xyts



#標準化したでーたから標準偏差の5倍のデータを抽出する
def to_std(v_xyts):
    np_v_xyts = np.array(v_xyts)
    mean_xyts = np.mean(np_v_xyts)
    std_xyts = np.std(np_v_xyts )
    standarized_xyts = (np_v_xyts - mean_xyts)/ std_xyts
    
    return standarized_xyts

#標準化したでーたから標準偏差の5倍のデータを抽出する
def to_std_2(v_xyts_df):
    mean_xyts = v_xy_df['v_xy'].mean()
    std_xyts =  v_xy_df['v_xy'].std()
    v_xy_df['stand_xy'] = (v_xy_df['v_xy']- mean_xyts)/ std_xyts
    
    return v_xy_df

def find_candidate(standarized_xyts):
    thres= 5 
    micros = []    
    for i in range(len(standarized_xyts)):
        if standarized_xyts[i] > thres:
            micros.append([i, standarized_xyts[i] ])
    return micros     


def find_cand_2(standarized_xyts):
    thres= 4 
    micros = []    
    for i in range(len(stand_xy_df)):
        if stand_xy_df['stand_xy'][i] > thres:
            micros.append([stand_xy_df['time'][i], stand_xy_df['stand_xy'][i]])
    return micros             

def miceos(cands):
    np_cands = np.array(cands)
    x, _ = np.split(np_cands, [1], 1)
    result = []
    tmp = [x[0].tolist()]
    for i in range(len(x)-1):
        if x[i+1] - x[i] == 1:
            tmp.append(x[i+1].tolist())
        else:
            if len(tmp) > 0:
                result.append(tmp)
            tmp = []
            tmp.append(x[i+1].tolist())
    return result        

#リストを引数で、連続した数値を出す
def freq_lst(lst):
    x = np.array(lst)
    x = np.append(x, 0)
    tmp = [x[0]]
    result = []
    for i in (range(len(x)-1)):
        if (x[i+1] - x[i]) == 1:
            tmp.append(x[i+1])
        else:
            if len(tmp) > 0:
                result.append(tmp)
            tmp = []
            tmp.append(x[i+1])
    return result     

#ウェブレット変換、逆変換waveletの種類の選択、level の選択


def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise(x, wavelet='db10', level=2):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    return pywt.waverec(coeff, wavelet, mode='per')   


       

#マイクロサッカードのピークのグラフを作成
def draw_peaks(df_v, max_v, s, e):
    max_v = max_v
    s = s
    e = e
    df_v = df_v

    fig = plt.figure(figsize=(4,8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(df_v.iloc[s:e,:]['std_v'], marker='o',mec='none', ms=4, lw=1, label='Ratio')
    ax1.hlines(5, s, e, colors='red', linestyles= 'dashed', label='Thre5')
    ax1.set_xticks(np.arange(s, e), minor=False)
    ax1.set_xlabel("Time [microsec]")
    ax1.set_ylabel("Ratio of std verocity")
    ax1.set_title("標準化した速度（Vxyt） の標準偏差との倍率,  Time: {} -{}microsec".format(s,e))

    ax1.grid()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(df_v.iloc[s:e, :]['v_deg'],marker='o', mec='none', ms=4, lw=1, label='Acce')
    ax2.hlines(100, s, e, colors='red', linestyles= 'dashed', label='Velo100')

    ax2.set_xticks(np.arange(s, e), minor=False)
    ax2.grid()
    ax2.set_xlabel("Time [microsec]")
    ax2.set_ylabel("Verocity[deg/sec]")
    v_max = df_v.iloc[s:e,:]['v_deg'].max()
    ax2.set_title("最大速度, 及び半値全幅の区間, Time: {} -{}microsec".format(s,e))

    idx_max = df_v.iloc[s:e,:]['v_deg'].idxmax()
    half_v =v_max/2
    ax2.hlines(half_v, s, e, colors='red', linestyles= 'solid', label='Velo100')
    # show plots
    fig.tight_layout()


    x, y1 = df_v.iloc[s:e, :]['v_deg'].index.values , df_v.iloc[s:e, :]['v_deg'].values

    y2 = np.full_like(x, half_v)
    interp1 = scipy.interpolate.InterpolatedUnivariateSpline(x, y1)
    interp2 = scipy.interpolate.InterpolatedUnivariateSpline(x, y2)
    plt.plot(x, y1, marker='', mec='none', ms=4, lw=1, label='y1')
    plt.plot(x, y2, marker='', mec='none', ms=4, lw=1, label='y2')

    idx = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)

    #plt.plot(x[idx], y1[idx], 'ms', ms=7, label='Nearest data-point method')


    new_x = np.linspace(x.min(), x.max(), 100)
    new_y1 = interp1(new_x)
    new_y2 = interp2(new_x)
    idx = np.argwhere(np.diff(np.sign(new_y1 - new_y2)) != 0)
    #print(idx)
    plt.plot(new_x[idx], new_y1[idx], 'ro', ms=7, label='Nearest data-point method, with re-interpolated data')
    if new_x[idx] != []:
        pass #print(f'Delta: {new_x[idx][1] - new_x[idx][0]}')
    if v_max < max_v:
        pass #print('True')
    else: 
        pass # print('NG')       

# def difference(x):
#     return np.abs(interp1(x) - interp2(x))

# x_at_crossing = scipy.optimize.fsolve(difference, x0=3.0)
# #plt.plot(x_at_crossing, interp1(x_at_crossing), 'cd', ms=7, label='fsolve method')
# ax2.plot(idx_max, v_max, 'ro', markersize=5)

#マイクロサッカードのアニメーション
def draw_plot_ms(df_v, file_name, start, end, ms_s, ms_e):
    import matplotlib.animation as animation
    from matplotlib.animation import FFMpegWriter

    #def anime_ms(df ,file_name, start, end, ms_s, ms_e):
    fig = plt.figure(figsize=(5, 5))

    ax = fig.add_subplot(111,  title=f'{file_name}_{ms_s}_{ms_e}sec')
    ax.set_aspect('equal')    
    ax.set_xlabel("X-axis(mm)")
    ax.set_ylabel("Y-axis(mm)")
    ax.grid()
    #ax.legend()
    frames = []
    #     lines= ["dashdot","solid" ]
    #     line = 0
    #     colors =["blue", "red"]

    for i in range(df[start:end].shape[0]):

        t = start + i


        if t >= start and t < ms_s:
            image_1= ax.plot(df_v['x_micr'][start:t], df_v['y_micr'][start:t], color='blue', linestyle ='dashdot')
            frames.append(image_1)    
        elif ms_s<=t and t<=ms_e :
            image_1= ax.plot(df_v['x_micr'][start:ms_s], df_v['y_micr'][start:ms_s], color='blue', linestyle ='dashdot' )
            image_2= ax.plot(df_v['x_micr'][ms_s:t], df_v['y_micr'][ms_s:t], color='red',linestyle ='solid') 
            frames.append(image_1+image_2)     
        elif ms_e<t and t<=end:
            image_1= ax.plot(df_v['x_micr'][start:ms_s], df_v['y_micr'][start:ms_s], color='blue', linestyle ='dashdot')
            image_2= ax.plot(df_v['x_micr'][ms_s:ms_e], df_v['y_micr'][ms_s:ms_e], color='red',linestyle ='solid') 
            image_3= ax.plot(df_v['x_micr'][ms_e:t], df_v['y_micr'][ms_e:t], color='blue', linestyle ='dashdot')
            frames.append(image_1+image_2+image_3)  






    anime = animation.ArtistAnimation(fig, frames, interval=10 , blit=True, repeat_delay=0)

    saved_file = file_name + '.gif' 
    anime.save(saved_file, writer="pillow")
    #plt.show()
    html = display.HTML(anime.to_jshtml())
    display.display(html)
    plt.close()