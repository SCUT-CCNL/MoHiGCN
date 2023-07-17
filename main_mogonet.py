""" Example for MOGONET classification
"""
from train_test import train_test
import numpy as np
from utils import save_model_dict
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'
import sys
import time

if __name__ == "__main__":
    class Logger(object):#for recording
        def __init__(self, filename="Default.log"):
            self.terminal = sys.stdout
            self.log = open(filename, "a")
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            pass
    print("Start recording")
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    data_folder_list = ['OurDBs/BRCA', 'OurDBs/LGG', 'OurDBs/HNSC', 'OurDBs/CESC', 'OurDBs/STAD']
    for data_folder in data_folder_list:

        print(data_folder)
        view_list = [1,2,3]
        num_epoch_pretrain=0
        lr_e_pretrain=0.001
        num_epoch_list = [1500] #1500
        if data_folder=='OurDBs/BRCA':
            lr_e_list, lr_c_list = [0.0001], [0.001]
            weight_decay_list = [0.00005]
        else:
            lr_e_list, lr_c_list = [0.00001], [0.001]
            weight_decay_list = [0]


        if data_folder == 'OurDBs/BRCA':
            num_class = 5
        if data_folder == 'OurDBs/LGG':
            num_class = 2
        if data_folder == 'OurDBs/HNSC':
            num_class = 2
        if data_folder == 'OurDBs/CESC':
            num_class = 2
        if data_folder == 'OurDBs/STAD':
            num_class = 2

        acc_m_all, acc_std_all = [],[]
        m1_m_all, m1_std_all = [],[]
        m2_m_all, m2_std_all = [],[]
        for num_epoch in num_epoch_list:
            for lr_e in lr_e_list:
                for lr_c in lr_c_list:
                    for weight_decay in weight_decay_list:
                        for c in range(10):
                            print('----------- c:',c, '-----------')
                            ACC = [0]
                            m1 = [0]
                            m2 = [0]
                            model_dict = [0]
                            ACC,m1,m2,model_dict=train_test(data_folder+'/DB', view_list, num_class,
                                                          lr_e_pretrain, lr_e, lr_c, weight_decay, num_epoch_pretrain, num_epoch)
                            acc_m_all.append(ACC)
                            m1_m_all.append(m1)
                            m2_m_all.append(m2)
                        acc_m_all=np.array(acc_m_all)
                        m1_m_all=np.array(m1_m_all)
                        m2_m_all=np.array(m2_m_all)
                        print(data_folder)
                        print(str(round(np.mean(acc_m_all[:]),3))+"±("+str(round(np.std(acc_m_all[:]),3))+")",
                          str(round(np.mean(m1_m_all[:]),3))+"±("+str(round(np.std(m1_m_all[:]),3))+")",
                          str(round(np.mean(m2_m_all[:]),3))+"±("+str(round(np.std(m2_m_all[:]),3))+")")
                        print("\n")






    
