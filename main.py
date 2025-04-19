import os
import sys
import time
from threading import Thread

import joblib
from sklearn import svm

import KeyValue
import signals
import suggestions
import windows
import serial
import _thread

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from sklearn.model_selection import GridSearchCV, train_test_split
from PyQt5.QtWidgets import QApplication, QMainWindow,QComboBox

loop_read = False
is_recording = False
AUTOCORRECT = True

PredictOrLearn = 0
# Recording parameters
target_sign = "a"
current_batch = "0"
target_directory = "data"


current_test_index = 0

# Resets the output file
output_file = open("output.txt", "w")
output_file.write("")
output_file.close()
clf = joblib.load('model.pkl')
classes = joblib.load('classes.pkl')
output = []


hinter = suggestions.Hinter.load_english_dict()
ser = serial.Serial()


"""serial_read_thread 负责监听串口，接收传感器传输的原始数据"""
def serial_read_thread(args):
    current_sample = 0
    while (True):  # 无限循环，不断从串口读取数据
        if ser.isOpen():
            # 从单片机读取的数据存放到line中
            line = str(ser.readline(), encoding="utf-8").replace("\r\n", "")
            if line == "STARTING BATCH":  # 数据传输开始
                is_recording = True
                output = []  # 初始化一个空列表 output，用于存储采集到的数据行
                print("RECORDING...")
            elif line == "CLOSING BATCH":  # 数据传输结束
                is_recording = False

                """output列表里的临时数据会根据不同的模式，以不同的文件名，存入不同文件夹"""
                if len(output) > 1:  # 如果output里存入了数据
                    print("DONE, SAVING..."),
                    # 生成文件名和文件保存路径
                    filename = "{sign}_sample_{batch}_{number}.txt".format(sign=KeyValue.target_sign, batch=KeyValue.current_batch,number=KeyValue.current_sample)
                    path = target_directory + os.sep + filename

                    # 预测模式下，修改 output 里的数据的存放路径为“tmp.txt”文件中
                    if PredictOrLearn == 0:
                        path = "tmp.txt"
                        filename = "tmp.txt"
                    else:
                        # 记录模式下，设置记录结束的标志为True
                        KeyValue.sample_record_end = True

                    f = open(path, "w")  # 打开数据文件，以写入模式
                    f.write('\n'.join(output))  # 将 output 列表中的数据写入文件，每行数据之间用换行符分隔。
                    f.close()
                    print("SAVED IN {filename}".format(filename=filename))

                    if PredictOrLearn == 0:
                        print("PREDICTING...")

                        # 从文件中加载传感器数据，处理成一维数组
                        # 这里加载的path就是tmp.txt文件
                        sample_test = signals.Sample.load_from_file(path)
                        linearized_sample = sample_test.get_linearized(reshape=True)

                        # 使用训练好的模型 clf 进行预测
                        number = clf.predict(linearized_sample)  # predict() 方法的返回值，是一个包含预测结果的数组，在这里只包含一个元素
                        char = chr(ord('a') + number[0])

                        KeyValue.predictchar = char
                        KeyValue.predic_Information_str = "预测字符: {0}".format(char)
                        KeyValue.predict_end = True
                        print(char) # 在控制台输出预测结果
###############################################################################################################
                        """将预测字符通过串口发送给单片机"""
                        try:
                            ser.write(KeyValue.predictchar.encode())  # 发送预测字符
                            print(f"Sent predict char '{KeyValue.predictchar}' to serial port.")  # 输出发送信息到控制台
                        except serial.SerialException as e:
                            print(f"Error sending data to serial port: {e}")
#################################################################################################################
                else:
                    print("ERROR...")

            else:
                output.append(line)



thread1 = Thread(target=serial_read_thread, args=('Curry',))
thread1.daemon = True

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = windows.Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
