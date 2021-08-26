#! /usr/bin/env python
from logging import Logger
import logging
from os import cpu_count
import numpy as np
import rospy
from fairmot_ros_wzq.msg import people_item ,people_list
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import PoseStamped
from nlink_example.msg import UwbFilter
import message_filters
from numpy import matrixlib as mt

goal_ID = 0         # 追踪ID
ID_list = []        # ID追踪列表
list_max = 100      # ID追踪最大存储个数
count = 0           # 判断初始化
stored_points = []  # 存储轨迹点
ign = 0
max_ign = 5
lost = False

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
        self.mix_track = []
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
        self.mix_track.append(x_mix)
        
        # 计算轨迹行进方向
        if len(self.mix_track)<6:
            orientation = np.array([[1],[0]], dtype=np.float32)
        else:
            l = len(self.mix_track)
            s1 = np.array(np.zeros((2,1)), dtype=np.float32)
            s2 = np.array(np.zeros((2,1)), dtype=np.float32)
            for i in range(l-4,l-2):
                s1 = s1 + self.mix_track[i].getA()/2
            for i in range(l-2,l):
                s2 = s2 + self.mix_track[i].getA()/2
            orientation = s2 - s1
        
        yaw = np.math.atan(orientation[1]/orientation[0])*360/(2*np.math.pi) 
        roll = 0
        pitch = 0
        cy = np.math.cos(yaw*np.math.pi/360)
        sy = np.math.sin(yaw*np.math.pi/360)
        cp = np.math.cos(pitch*np.math.pi/360)
        sp = np.math.sin(pitch*np.math.pi/360)
        cr = np.math.cos(roll*np.math.pi/360)
        sr = np.math.sin(roll*np.math.pi/360)
        qw = cy*cp*cr+sy*sp*sr
        qx = cy*cp*sr-sy*sp*cr
        qy = sy*cp*sr+cy*sp*cr
        qz = sy*cp*cr-cy*sp*sr
        q = [qw,qx,qy,qz]
        
        return x_image, x_uwb, x_mix, q

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

# 初始化ID
def init_ID(people_list_data):
    global ID_list
    global goal_ID
    max_frame_x = 1920
    max_frame_y = 1080
    center = [max_frame_x/2, max_frame_y/2]
    people = []
    for people_item in people_list_data.people_items:
        people.append(people_item)
        if len(ID_list)<=list_max:
            ID_list.append(people_item.people_ID)
        else:
            Logger.warn('ID_list overflow in init!!!')
    dist = []
    for man in people:
        origin_x = man.origin_x
        origin_y = man.origin_y
        width = man.width
        height = man.height
        centriod = [(origin_x+width)/2, (origin_y+height)/2]
        cen_dist = np.linalg.norm(np.array(centriod)-np.array(center))
        dist.append(cen_dist)
    goal_ID = people[dist.index(min(dist))].people_ID

def update_ID(people_list_data, uwb_data):
    global ID_list, goal_ID, ign, lost

    ID_list_cur = []
    people = []
    uwb = [uwb_data.point.x, uwb_data.point.y, uwb_data.point.z]
    for item in people_list_data.people_items:
        people.append(item)
        ID_list_cur.append(item.people_ID)
    if len(ID_list_cur):    # 相机捕捉到目标
        for ID_c in ID_list:
            if ID_c in ID_list_cur:
                ign = 0
                goal_ID = ID_c
            elif lost:
                diff = []
                for i,id in enumerate(ID_list_cur):
                    image_point = [people[i].points.x, people[i].points.y]
                    uwb_point = [uwb[0], uwb[1]]
                    dist = np.linalg.norm(np.array(image_point)-np.array(uwb_point))
                    diff.append(dist)
                goal_ID = ID_list_cur[diff.index(min(diff))]
                ID_list.append(goal_ID)
        if ign:
            ign = ign + 1
    else:   # 相机捕捉不到目标时，如何确定goal
        ign = ign + 1

    if ign>max_ign: # 判断是否应该丢失目标
        lost = True
    else:
        lost = False
    


        
def callback(people_list_data, uwb_data):
    global count, track

    image_point = []
    uwb_point = [uwb_data.point.x, uwb_data.point.y, uwb_data.point.z]
    if (count == 0):    # 初始化目标ID
        if people_list_data.people_items:
            init_ID(people_list_data)
            for item in people_list_data.people_items:
                if item.people_ID == goal_ID:
                    image_point = [item.points.x,item.points.y,item.points.z]
            track = trajectory(image_point, uwb_point,
                                A=[[1,0],[0,1]],
                                B=0,H=[[1,0],[0,1]],
                                R1=[[0.01,0],[0,0.01]],
                                Q1=[[0.0001,0],[0,0.0001]],
                                R2=[[0.01,0],[0,0.01]],
                                Q2=[[0.0001,0],[0,0.0001]],
                                x0_i=[[image_point[0]],[image_point[1]]],
                                x0_u=[[uwb_point[0]],[uwb_point[1]]],
                                P0=[[1,0],[0,1]])
            count = count + 1
        else:
            Logger.info('At the beginning, can not find person!')
    elif (count>0):                                     # 
        if people_list_data.people_items:
            for item in people_list_data.people_items:
                if item.people_ID == goal_ID:
                    image_point = [item.points.x,item.points.y,item.points.z]
        update_ID(people_list_data, uwb_data)
        # 存储更新image、uwb轨迹点
        x_img, x_uwb, x_mix, orientation = track.update(image_point, uwb_point)
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.pose.position.x = x_mix[0][0]
        goal_msg.pose.position.y = x_mix[1][0]
        goal_msg.pose.orientation.w = orientation[0]
        goal_msg.pose.orientation.x = orientation[1]
        goal_msg.pose.orientation.y = orientation[2]
        goal_msg.pose.orientation.z = orientation[3]
        goal_pub.publish(goal_msg)
        count = count + 1

    else:
        Logger.info('Can not find person, please put yourself forward the camera!')

    


def goal():
    rospy.init_node("goal", anonymous=True)
    people_list_data = message_filters.Subscriber("people_list", people_list)
    # rslidar_points = message_filters.Subscriber("rslidar_points", PointCloud2)
    uwb_data = message_filters.Subscriber("uwb_filter", UwbFilter)
    # sub = message_filters.ApproximateTimeSynchronizer([people_centriod, rslidar_points, uwb_filter], 10, 1, allow_headerless=True)
    sub = message_filters.ApproximateTimeSynchronizer([people_list_data, uwb_data], 10, 1, allow_headerless=False)
    sub.registerCallback(callback)
    
    rospy.spin()


if __name__=='__main__':
    goal_pub = rospy.Publisher('goal', PoseStamped, queue_size=10)
    goal()