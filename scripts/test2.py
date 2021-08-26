#! /usr/bin/env python
# import rospy
# from std_msgs.msg import String

import numpy as np
import matplotlib.pyplot as plt

def main():
    data_file_path = "../data/data-v2.1.txt"
    confidence_file_path = "../data/confidence-v1.2.txt" 
    with open(data_file_path, "r") as f:
        point_x = []
        for line in f.readlines():
            line = line.strip('\n')
            find = line.find(' x: ')
            if (find >= 0):
                point_x.append(float(line[(find+4):]))
    with open(data_file_path, "r") as f:
        point_y = []
        for line in f.readlines():
            line = line.strip('\n')
            find = line.find(' y: ')
            if (find >= 0):
                point_y.append(float(line[(find+4):]))
    with open(data_file_path, "r") as f:
        point_z = []
        for line in f.readlines():
            line = line.strip('\n')
            find = line.find(' z: ')
            if (find >= 0):
                point_z.append(float(line[(find+4):]))
    with open(confidence_file_path, "r") as f:
        confidence = []
        for line in f.readlines():
            line = line.strip('\n')
            find = line.find('depth_confidence')
            if (find >= 0):
                confidence.append(float(line[find+18:]))


    data_file_path = "../data/mix.txt"
    with open(data_file_path, "r") as f:
        image_px = []
        image_py = []
        uwb_px = []
        uwb_py = []
        mix_px = []
        mix_py = []
        count = 0
        for line in f.readlines():
            line = line.strip('\n')
            find_x = line.find('[[')
            find_xed = line.find('] ')
            find_y = line.find(' [')
            find_yed = line.find(']]')
            if count%3==0:
                if (find_x >= 0):
                    image_px.append(float(line[2:find_xed]))
                elif (find_y >= 0):
                    image_py.append(float(line[2:find_yed]))
                    count = count + 1
            elif count%3==1:
                if (find_x >= 0):
                    uwb_px.append(float(line[2:find_xed]))
                elif (find_y >= 0):
                    uwb_py.append(float(line[2:find_yed]))
                    count = count + 1
            elif count%3==2:
                if (find_x >= 0):
                    mix_px.append(float(line[2:find_xed]))
                elif (find_y >= 0):
                    mix_py.append(float(line[2:find_yed]))
                    count = count + 1

    data_file_path = "../data/image.txt"
    with open(data_file_path, "r") as f:
        image_x = []
        image_y = []
        for line in f.readlines():
            line = line.strip('\n')
            find_x = line.find(' x: ')
            find_y = line.find(' y: ')
            if (find_x >= 0):
                image_x.append(float(line[find_x+4:]))
            if (find_y >= 0):
                image_y.append(float(line[find_y+4:]))
    data_file_path = "../data/uwb.txt"
    with open(data_file_path, "r") as f:
        uwb_x = []
        uwb_y = []
        for line in f.readlines():
            line = line.strip('\n')
            find_x = line.find(' x: ')
            find_y = line.find(' y: ')
            if (find_x >= 0):
                uwb_x.append(float(line[find_x+4:]))
            if (find_y >= 0):
                uwb_y.append(float(line[find_y+4:]))
    

    plt.figure(figsize=(15,9))
    # t1 = np.linspace(0, 10, len(image))
    # t2 = np.linspace(0, 10, len(image_px))
    plt.plot(image_px, image_py, ls='-', lw=2, color='red')
    plt.plot(uwb_px, uwb_py, ls='-', lw=2, color='blue')
    plt.plot(mix_px, mix_py, ls='-', lw=2, color='purple')
    plt.plot(image_x, image_y)
    plt.plot(uwb_x, uwb_y)
    plt.legend(loc='upper right')
    plt.xlabel('x/m')
    plt.ylabel('y/m')

    
    # uwb_x = []
    # uwb_y = []
    # uwb_z = []
    # image_x = []
    # image_y = []
    # image_z = []
    # for i,point in enumerate(point_x):
    #     if i%2==0:
    #         uwb_x.append(point)
    #     else:
    #         image_x.append(point)
    # for i,point in enumerate(point_y):
    #     if i%2==0:
    #         uwb_y.append(point)
    #     else:
    #         image_y.append(point)
    # for i,point in enumerate(point_z):
    #     if i%2==0:
    #         uwb_z.append(point)
    #     else:
    #         image_z.append(point)

    # plt.figure(figsize=(15,9))
    # t1 = np.linspace(0, 10, len(uwb_x))
    # # kalman_filter
    # # N = len(point_x)
    # # K = np.zeros((N,1))
    # # X = np.zeros((N,1))
    # # P = np.zeros((N,1))
    # # X[0] = point_x[0]
    # # P[0] = 10
    # # R = 0.001
    # # Q = 0.01
    # # for i in range(1,len(point_x)):
    # #     K[i] = P[i-1]/(P[i-1]+R)
    # #     X[i] = X[i-1]+K[i]*(point_x[i]-X[i-1])
    # #     P[i] = (1-K[i])*(P[i-1]+Q)
    # plt.plot(t1, uwb_x, ls='-', lw=2, label='point_x', color='purple')
    # # plt.plot(t1, point_x, ls='-', lw=2, label='point_x', color='purple')
    # plt.legend(loc='upper right')
    # plt.xlabel('t(not real time)')
    # plt.ylabel('x/m')
    # # plt.xlim((0,4))
    # # plt.ylim((0.5,4))

    
    # fig2 = plt.figure(figsize=(15,9))
    # ax3 = fig2.add_subplot(111)
    # ax4 = ax3.twinx()
    # t3 = np.linspace(0, 10, len(uwb_y))
    # t4 = np.linspace(0, 10, len(uwb_x))
    # ax3.plot(t3, uwb_y, ls='-', lw=2, label='point_y', color='purple')
    # ax4.plot(t4, uwb_x, ls='-', lw=2, label='point_x', color='red')
    # ax3.legend(loc='upper left')
    # ax4.legend(loc='upper right')
    # ax3.set_xlabel('t(not real time)')
    # ax3.set_ylabel('y/m')
    # ax4.set_ylabel('x/m')
    # # ax3.set_xlim((0,4))
    # # ax3.set_ylim((-0.2,0.2))

    # fig3 = plt.figure(figsize=(15,9))
    # ax5 = fig3.add_subplot(111)
    # ax6 = ax5.twinx()
    # t5 = np.linspace(0, 10, len(uwb_z))
    # t6 = np.linspace(0, 10, len(uwb_x))
    # ax5.plot(t5, uwb_z, ls='-', lw=2, label='point_z', color='purple')
    # ax6.plot(t6, uwb_x, ls='-', lw=2, label='point_x', color='red')
    # ax5.legend(loc='upper left')
    # ax6.legend(loc='upper right')
    # ax5.set_xlabel('t(not real time)')
    # ax5.set_ylabel('z/m')
    # ax6.set_ylabel('x/m')
    # # ax5.set_xlim((0,4))
    # # ax5.set_ylim((0,1.2))

    plt.show()

if __name__ == '__main__':
    main()