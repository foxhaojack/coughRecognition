
# coding: utf-8

# In[43]:


from pyaudio import PyAudio, paInt16 
import numpy as np 
from datetime import datetime 
import wave
import os
import wave
import contextlib
import MySQLdb
import speech_recognition as sr
import time
import os
import pyaudio
import wave 
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
from keras.models import load_model
from preprocess import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from pydub import AudioSegment
import MySQLdb
feature_dim_2 = 11
feature_dim_1 = 20
channel = 1
epochs = 50
batch_size = 100
verbose = 1
num_classes = 2
class recoder:
    NUM_SAMPLES = 2000      #pyaudio内置缓冲大小
    SAMPLING_RATE = 8000    #取样频率
    LEVEL = 500         #声音保存的阈值
    COUNT_NUM = 20      #NUM_SAMPLES个取样之内出现COUNT_NUM个大于LEVEL的取样则记录声音
    SAVE_LENGTH = 8         #声音记录的最小长度：SAVE_LENGTH * NUM_SAMPLES 个取样
    TIME_COUNT = 10     #录音时间，单位s

    Voice_String = []
    
    def savewav(self,filename):
        wf = wave.open(filename, 'wb') 
        wf.setnchannels(1) 
        wf.setsampwidth(2) 
        wf.setframerate(self.SAMPLING_RATE) 
        wf.writeframes(np.array(self.Voice_String).tostring()) 
        # wf.writeframes(self.Voice_String.decode())
        wf.close() 

    def recoder(self):
        pa = PyAudio() 
        stream = pa.open(format=paInt16, channels=1, rate=self.SAMPLING_RATE, input=True, 
            frames_per_buffer=self.NUM_SAMPLES) 
        save_count = 0 
        save_buffer = [] 
        time_count = self.TIME_COUNT
        a = datetime.now()
        prevTime = a.second

        while True:
            #time_count -= 1
            
            # print time_count
            # 读入NUM_SAMPLES个取样
            string_audio_data = stream.read(self.NUM_SAMPLES) 
            # 将读入的数据转换为数组
            audio_data = np.fromstring(string_audio_data, dtype=np.short)
            # 计算大于LEVEL的取样的个数
            large_sample_count = np.sum( audio_data > self.LEVEL )
            #print(np.max(audio_data))
            # 如果个数大于COUNT_NUM，则至少保存SAVE_LENGTH个块
            if large_sample_count > self.COUNT_NUM:
                save_count = self.SAVE_LENGTH 
            else: 
                save_count -= 1

            if save_count < 0:
                save_count = 0 

            if save_count > 0 : 
            # 将要保存的数据存放到save_buffer中
                #print  save_count > 0 and time_count >0
                save_buffer.append( string_audio_data ) 
            else: 
            #print save_buffer
            # 将save_buffer中的数据写入WAV文件，WAV文件的文件名是保存的时刻
                #print "debug"
                if len(save_buffer) > 0 : 
                    self.Voice_String = save_buffer
                    save_buffer = [] 
                    print("Recode a piece of  voice successfully!")
                    return True
            a = datetime.now()
            nowTime = a.second
            #if time_count==0:  
            N = (nowTime%10)-(prevTime%10)
            if N>=2 or N==-8: 
                if len(save_buffer)>1:
                    self.Voice_String = save_buffer
                    save_buffer = [] 
                    print("Recode a piece of  voice successfully!")
                    prevTime = a.second
                    return True
                else:
                    return False

def get_wav_time(wav_path):
    '''
    获取音频文件是时长
    :param wav_path: 音频路径
    :return: 音频时长 (单位秒)
    '''
    with contextlib.closing(wave.open('C:\\Users\\Kelly\\recordTest\\'+ wav_path, 'r')) as f:
        frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)
    return duration

def predict(filepath, model):
    sample = wav2mfcc(filepath)
    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
    return get_labels()[0][
            np.argmax(model.predict(sample_reshaped))
    ]
def AudioToText(file_name):
    r = speech_recognition.Recognizer()
    with speech_recognition.AudioFile(file_name) as source:  
        audio = r.record(source)
    try:
        Text = r.recognize_google(audio, language="zh-TW")   
    except sr.UnknownValueError:
        Text = "無法翻譯"
    except sr.RequestError as e:
        Text = "無法翻譯{0}".format(e)
    return Text
    
if __name__ == "__main__":
    model = load_model('my_model.h5')
    list= []
    list2= []
    ret= []
    rm = []
    rm2=[]
    fi =[]
    LIMITCOUNT = 10
    filepath = 'C:\\Users\\Kelly\\recordTest'
    while (1):
        r = recoder()
        r.recoder()
        x = datetime.now()
        strx = str(x.year)+"-"+str(x.month)+"-"+str(x.day)+"-"+str(x.hour)+"-"+str(x.minute)+"-"+str(x.second)
        print(strx)
        r.savewav("C:\\Users\\Kelly\\recordTest\\"+strx+".wav")
        LIMITCOUNT -= 1
        for root, dirs, files in os.walk(filepath):
            for f in files:
                list.append(f)
                for i in list:
                    if not i in list2:
                        ret.append(str(get_wav_time(i)))
                        list2.append(i)
        for i in list2:
            for k,v in enumerate(ret):
                if (v=="0.0" or v=="0.25"):
                    rm.append(k)
                    for i in rm:
                        if not i in rm2:
                            rm2.append(i)
                            os.remove("C:\\Users\\Kelly\\recordTest\\"+list2[i]) 
        if len([lists for lists in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, lists))]) == 10:
            for root, dirs, files in os.walk(filepath):
                for f in files:
                    Text = AudioToText("C:/Users/Kelly/recordtest/"+f)
                    if(Text !='無法翻譯'):
                        os.remove("C:\\Users\\Kelly\\recordTest\\"+f)
            for root, dirs, files in os.walk(filepath):
                for f in files:
                    sound = AudioSegment.from_file("C:/Users/Kelly/recordtest/"+f)
                    loudness = sound.dBFS
                    loudness1 = sound.rms
                    if(loudness<-45 and loudness1<200):
                        os.remove("C:\\Users\\Kelly\\recordTest\\"+f)
            for root, dirs, files in os.walk(filepath):
                for f in files:
                    predict2=predict(("C:\\Users\\Kelly\\recordTest\\"+f),model=model)
                    if(predict2!='cough'):
                        print(predict2)
                        os.remove("C:\\Users\\Kelly\\recordTest\\"+f)
                    else:
                        fi.append(f.split('.')[0])
                        conn=MySQLdb.connect(host="localhost",user="root", passwd="abc0963470417", db="cough", charset="utf8")
                        cursor=conn.cursor()  
                        for i in range(len(fi)):
                            SQL = "INSERT INTO cough_time(time)VALUES('"+fi[i]+"')"
                            cursor.execute(SQL)
                            conn.commit()
                        conn.close()
                        os.remove("C:\\Users\\Kelly\\recordTest\\"+f)
                        del fi[:]


# In[42]:


print(fi)


# In[ ]:




