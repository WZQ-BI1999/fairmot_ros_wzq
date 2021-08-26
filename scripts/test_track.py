#! /usr/bin/env python
from os import read
import numpy as np
from matplotlib import pyplot as plt
from numpy import matrixlib as mt
from numpy.core.fromnumeric import mean
from numpy.lib.polynomial import polyfit
import rospy
from nlink_example.msg import UwbFilter
from fairmot_ros_wzq.msg import people_list
import message_filters
import time

count = 0


class kalman_filter():
    def __init__(self, A, B, H, R, Q, x0, P0):
        self.A = mt.mat(np.array(A, dtype=np.float32))
        self.B = mt.mat(np.array(B, dtype=np.float32))
        self.H = mt.mat(np.array(H, dtype=np.float32))
        self.R = mt.mat(np.array(R, dtype=np.float32))
        self.Q = mt.mat(np.array(Q, dtype=np.float32))
        self.x0 = mt.mat(np.array(x0, dtype=np.float32))
        self.P0 = mt.mat(np.array(P0, dtype=np.float32))
        self.x_pre = self.x0
        self.P_pre = self.P0

        self.K = (self.P0*self.H.T)*((self.H*self.P0*self.H.T+self.R).I)
        # print(self.x_pre)
        self.I = np.eye((self.K*self.H).shape[0],(self.K*self.H).shape[1])

        # 无外加控制
        self.u = 0.
    def run(self, data):
        z = mt.mat(np.array([[data[0]],[data[1]]], dtype=np.float32))
        self.predict()
        self.update(z)

        return self.x_pre, self.P_pre 

    def predict(self):
        self.x_pre = self.A*self.x_pre + self.B*self.u
        self.P_pre = self.A*self.P_pre*self.A.T + self.Q
    
    def update(self, z):
        self.K = (self.P_pre*self.H.T)*((self.H*self.P_pre*self.H.T+self.R).I)
        self.x_pre = self.x_pre + self.K*(z - self.H*self.x_pre)
        self.P_pre = (self.I - self.K*self.H)*self.P_pre
        

class trajectory(object):
    def __init__(self, image_points, uwb_points, A,B,H,R1,Q1,R2,Q2,x0_i,x0_u,P0):
        self.image_track = []
        self.uwb_track = []
        self.image_track.append(image_points)
        self.uwb_track.append(uwb_points)
        
        self.ign = 0
        self.max_ign = 5

        self.max_points = 50
        self.count = len(self.image_track)

        self.kalman_filter = []
        filter1=kalman_filter(A,B,H,R1,Q1,x0_i,P0)
        filter2=kalman_filter(A,B,H,R2,Q2,x0_u,P0)
        self.kalman_filter.append(filter1)
        self.kalman_filter.append(filter2)

    def update(self, image_point, uwb_point): # 更新轨迹，利用uwb点对image点插值
        x_uwb, P_uwb = self.kalman_filter[1].run(uwb_point)
        if len(self.uwb_track)>=self.max_points:
            self.uwb_track.pop(0)
            self.uwb_track.append(x_uwb)
        else:
            self.uwb_track.append(x_uwb)

        if image_point == []:
            if len(self.image_track)<=4:
                image_point = self.diff(self.image_track)
            else:
                # 曲线拟合
                # x,y = self.gather(self.image_track, 2)
                # print(self.image_track)
                # print(x)
                # print(y)
                # p = np.polyfit(x,y,2)
                # x_pre = uwb_point[0]
                # y_pre = np.polyval(p,x_pre)
                # image_point = [x_pre, y_pre, self.image_track[self.count-1][2]]
                
                # 插值
                # points = []
                # for i in range(4):
                #     points.append(self.image_track[len(self.image_track)-4+i])
                # image_point = self.diff(points)
                image_point = [self.image_track[self.count-1][0,0],
                               self.image_track[self.count-1][1,0]]
        
        x_image, P_image = self.kalman_filter[0].run(image_point)

        if len(self.image_track)>=self.max_points:
            self.image_track.pop(0)
            self.image_track.append(x_image)
        else:
            self.image_track.append(x_image)
            self.count = self.count + 1
        
        P_mix = (P_uwb.I+P_image.I).I
        x_mix = P_mix*(P_uwb.I*x_uwb + P_image.I*x_image)
        
        return x_image, x_uwb, x_mix

    def diff(self, points):
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        if len(points)==1:
            x_pre = x[0]
            y_pre = y[0]
        else:
            x1 = np.mean([x[i] for i in range(len(points)//2)])
            x2 = np.mean([x[i] for i in range(len(points)//2,len(points))])
            y1 = np.mean([y[i] for i in range(len(points)//2)])
            y2 = np.mean([y[i] for i in range(len(points)//2,len(points))])
            x_pre = x2+(x2-x1)*(1/2+1/(len(points)//2))
            y_pre = y2+(y2-y1)*(1/2+1/(len(points)//2))
        point_pre = [x_pre,y_pre]

        return point_pre
    
    def gather(self,points,n):
        x = [point[0] for point in self.image_track]
        y = [point[1] for point in self.image_track]
        i = 0
        count = 0
        sum_x = 0
        sum_y = 0
        gather_x = []
        gather_y = []
        while i<len(points):
            count = count + 1
            if count<=n:
                sum_x = sum_x+x[i]
                sum_y = sum_y+y[i]
            else:
                gather_x.append(sum_x/n)
                gather_y.append(sum_y/n)
                sum_x = 0
                sum_y = 0
                count = 0
            i = i + 1
        if count!=0:
            gather_x.append(sum_x/count)
            gather_y.append(sum_y/count)
        
        return gather_x, gather_y

track = trajectory([2.47,-0.16,0.89], [2.49,-0.4,1.3],
                    A=[[1,0],[0,1]],
                    B=0,H=[[1,0],[0,1]],
                    R1=[[0.01,0],[0,0.01]],
                    Q1=[[0.0001,0],[0,0.0001]],
                    R2=[[0.01,0],[0,0.01]],
                    Q2=[[0.0001,0],[0,0.0001]],
                    x0_i=[[2.47],[-0.16]],
                    x0_u=[[2.49],[-0.4]],
                    P0=[[1,0],[0,1]])
uwb_points = []
image_points = []
mix_points = []

def callback(people, uwb_filter):
    global count
    global track
    if (people.people_items):
        image_point = [people.people_items[0].points.x,
                        people.people_items[0].points.y]
    else:
        image_point = []
    uwb_point = [uwb_filter.point.x,
                  uwb_filter.point.y]
        
    image, uwb, mix = track.update(image_point, uwb_point)
    file = open(r'../data/mix.txt', mode='a')
    file.write(str(image))
    file.write('\n')
    file.write(str(uwb))
    file.write('\n')
    file.write(str(mix))
    file.write('\n')
    file.close()
    print(mix[0])




def listener():
    rospy.init_node("test", anonymous=True)
    people = message_filters.Subscriber("people_list", people_list)
    # rslidar_points = message_filters.Subscriber("rslidar_points", PointCloud2)
    uwb_filter = message_filters.Subscriber("uwb_filter", UwbFilter)
    # sub = message_filters.ApproximateTimeSynchronizer([people_centriod, rslidar_points, uwb_filter], 10, 1, allow_headerless=True)
    sub = message_filters.ApproximateTimeSynchronizer([people, uwb_filter], 10, 1, allow_headerless=False)
    sub.registerCallback(callback)
    
    rospy.spin()


if __name__ == '__main__':
    file = open('../data/mix.txt','w').close()
    listener()