#!/usr/bin/env python
#   /home/ww/catkin_ws/src/fairmot_ros_wzq/msg
import os
from re import S
import cv2
from matplotlib import colors
import numpy as np
from numpy.core.fromnumeric import std
from numpy.core.numeric import indices
from numpy.lib.function_base import vectorize
import _init_paths
from opts import opts
from tracking_utils.utils import mkdir_if_missing
import datasets.dataset.jde as datasets
from track import eval_seq
import pyzed.sl as sl # zed

import torch
from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer

import rospy
import pcl
import argparse
import struct
import ctypes
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud, PointCloud2, PointField
from geometry_msgs.msg import Point32
from fairmot_ros_wzq.msg import people_item, people_list, pose

# parser = argparse.ArgumentParser(
#     description='Fairmot_ros')
# parser.add_argument('--Mode', '-m', choices=('ZED', 'SVO'), default='',
#                     help='data from ZED/data from SVO_FILE')
# args = parser.parse_args()

def pcl_to_ros(pcl_array, frame_id, stamp):
    """ Converts a ROS PointCloud2 message to a pcl PointXYZRGB
    
        Args:
            pcl_array (PointCloud_PointXYZRGB): A PCL XYZRGB point cloud
            
        Returns:
            PointCloud2: A ROS point cloud
    """
    pcl_array = np.array(pcl_array, dtype=np.float32)
    # print(pcl_array)
    pcl_array = pcl_array.reshape(-1, 4)

    ros_msg = PointCloud2()

    ros_msg.header.stamp = stamp
    ros_msg.header.frame_id = frame_id

    ros_msg.height = 1
    ros_msg.width = pcl_array.size

    ros_msg.fields.append(PointField(
                            name="x",
                            offset=0,
                            datatype=PointField.FLOAT32, count=1))
    ros_msg.fields.append(PointField(
                            name="y",
                            offset=4,
                            datatype=PointField.FLOAT32, count=1))
    ros_msg.fields.append(PointField(
                            name="z",
                            offset=8,
                            datatype=PointField.FLOAT32, count=1))
    ros_msg.fields.append(PointField(
                            name="rgb",
                            offset=16,
                            datatype=PointField.FLOAT32, count=1))

    ros_msg.is_bigendian = False
    ros_msg.point_step = 32
    ros_msg.row_step = ros_msg.point_step * ros_msg.width
    ros_msg.is_dense = False
    buffer = []

    for data in pcl_array:
        # color = [0,0,255,1.0]
        # s = struct.pack('>f', color)
        s = struct.pack('>f', data[3])
        i = struct.unpack('>l', s)[0]
        pack = ctypes.c_uint32(i).value

        r = (pack & 0x00FF0000) >> 16
        g = (pack & 0x0000FF00) >> 8
        b = (pack & 0x000000FF)

        buffer.append(struct.pack('ffffBBBBIII', data[0], data[1], data[2], 1.0, b, g, r, 0, 0, 0, 0))
        # print(len(bytes(struct.pack('ffffBBBBIII', data[0], data[1], data[2], 1.0, b, g, r, 0, 0, 0, 0))))
    #     buffer = buffer + (struct.pack('ffffBBBBIII', data[0], data[1], data[2], 1.0, b, g, r, 0, 0, 0, 0))
    #     print(len(bytes(struct.pack('ffffBBBBIII', data[0], data[1], data[2], 1.0, b, g, r, 0, 0, 0, 0))))
    # ros_msg.data = bytes(buffer.encode('utf-8'))
    ros_msg.data = int(bytes(buffer))
    # print(str(buffer)[0])

    return ros_msg

def K_means(data, K):
    """
    程序说明：
    本函数实现二维和三维数据的K_means聚类算法
    data:输入的数据，维度(m, 2)或者(m, 3)
    K:表示希望分出来的类数
    """
    num = np.shape(data)[0]

    cls = np.zeros([num], np.int)
  
    random_array = np.random.random(size = K)
    random_array = np.floor(random_array*num)
    rarray = random_array.astype(int)
    center_point = data[rarray]

    change = True  #change表示簇中心是否有过改变，又改变了就需要继续循环程序，没改变则终止程序
    while change:
        for i in range(num):
            temp = data[i] - center_point   #此句执行之后得到的是两个数或三个数：x-x_0,y-y_0或x-x_0, y-y_0, z-z_0
            temp = np.square(temp)          #得到(x-x_0)^2等
            distance = np.sum(temp,axis=1)  #按行相加，得到第i个样本与所有center point的距离
            cls[i] = np.argmin(distance)    #取得与该样本距离最近的center point的下标

        change = False
        for i in range(K):
            # 找到属于该类的所有样本
            club = data[cls==i]
            newcenter = np.mean(club, axis=0)  #按列求和，计算出新的中心点
            ss = np.abs(center_point[i]-newcenter) # 如果新旧center的差距很小，看做他们相等，否则更新之。run置true，再来一次循环
            if np.sum(ss, axis=0) > 1e-4:
                center_point[i] = newcenter
                change = True
    # 找到拥有最多点的类
    cls_num = []
    for i in range(K):
        club = data[cls==i]
        cls_num.append(len(club))
    max_num = cls_num.index(max(cls_num))
    claster = data[cls==max_num]

    return claster


# 数据预处理
def Data_preprocess(pointcloud_value, confidence_value):
    # 数据降维
    pointcloud_value = pointcloud_value.reshape(-1,3)
    confidence_value = confidence_value.reshape(-1,1)
    # 剔除nan、inf数据以及depth_confidence过低的数据
    pd_delete = np.union1d(np.where(np.isnan(pointcloud_value))[0], np.where(np.isinf(pointcloud_value))[0])
    con_delete = np.where(confidence_value < 10)[0]
    delete_data = np.union1d(pd_delete, con_delete)
    pointcloud_value_pro = np.delete(pointcloud_value, delete_data, axis=0)
    confidence_value_pro = np.delete(confidence_value, delete_data, axis=0)
    if (confidence_value_pro==[]):
        confidence_value_pro = [0]
    # 体素滤波-数据下采样处理
    #pointcloud_pcl = pcl.PointCloud_PointXYZRGB(pointcloud_value_pro)
    #pointcloud_pcl_filtered = pointcloud_pcl.make_voxel_grid_filter()
    #pointcloud_pcl_filtered.set_leaf_size(0.05,0.05,0.05)
    #pointcloud_pcl = pointcloud_pcl_filtered.filter()
    #pointcloud_pcl = pointcloud_pcl.to_array()
    # # 聚类（两类）
    # pointcloud_pcl = K_means(pointcloud_pcl, 2)

    # 统计学滤波-离群点剔除
    #print(pointcloud_value_pro.shape)
    pointcloud_pcl = pcl.PointCloud()
    pointcloud_pcl.from_array(np.array(pointcloud_value_pro, dtype=np.float32))
    sor = pointcloud_pcl.make_statistical_outlier_filter()
    # sor.set_InputCloud(pointcloud_pcl)
    # sor = pcl.StatisticalOutlierRemovalFilter_PointXYZRGBA(pointcloud_pcl)
    sor.set_mean_k(50)
    sor.set_std_dev_mul_thresh(0.2)
    sor.set_negative(False)
    pointcloud_pcl_radius = sor.filter()
    flag = 0
    # print(pointcloud_pcl_radius.shape)
    # sor.set_negative(True)
    # print(sor.filter().to_array())
    

    # pointcloud_pcl_radius = pcl.PointCloud()
    # pointcloud_pcl_radius.from_array(np.array(pointcloud_value_pro, dtype=np.float32))
    
    # 欧式距离聚类
    tree = pointcloud_pcl_radius.make_kdtree()
    ec = pointcloud_pcl_radius.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.05)
    ec.set_MinClusterSize(100)
    # ec.set_MaxClusterSize(25000)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()
    # 找到最大类
    max_num = 0
    find_cluster = 0
    if (cluster_indices):
        for i in range(len(cluster_indices)):
            if len(cluster_indices[i])>max_num:
                max_num = len(cluster_indices[i])
                find_cluster = i
    # 根据索引筛选点
    print(len(cluster_indices))
    print(find_cluster)
    if (find_cluster < len(cluster_indices)):
        ece_data = np.array(np.zeros((len(cluster_indices[find_cluster]),3)),dtype=np.float32) 
        pointcloud_pcl_radius = pointcloud_pcl_radius.to_array()
        for i,num in enumerate(cluster_indices[find_cluster]):
            ece_data[i,:] = pointcloud_pcl_radius[num,:]
        pointcloud_pcl_ece = pcl.PointCloud()
        pointcloud_pcl_ece.from_array(np.array(ece_data, dtype=np.float32))
        flag = 1
    
    # 半径滤波
    #print(pointcloud_value_pro)
    #pointcloud_pcl = pcl.PointCloud()
    #pointcloud_pcl.from_array(pointcloud_value_pro)
    #print(pointcloud_pcl)
    #ror = pointcloud_pcl.make_RadiusOutlierRemoval()
    #print(ror)
    #ror.set_radius_search(0.08)
    #print(ror.get_radius_search())
    #ror.set_MinNeighborsInRadius(15)
    #print(ror.get_MinNeighborsInRadius())
    #pointcloud_pcl_radius = ror.filter()
    #print(pointcloud_pcl_radius)
    #pointcloud_pcl_radius = pointcloud_pcl_radius.to_array()
    #print(pointcloud_pcl_radius.shape)

    ## 聚类（两类）
    #pointcloud_pcl_radius = K_means(pointcloud_pcl_radius, 3)

    # 体素滤波
    #pointcloud_pcl_vox = pcl.PointCloud()
    #pointcloud_pcl_vox.from_array(pointcloud_pcl_radius)
    #vox = pointcloud_pcl_vox.make_voxel_grid_filter()
    #vox.set_leaf_size(0.05,0.05,0.05)
    #pointcloud_pcl_vox_filtered = vox.filter().to_array()

    # 半径滤波
    # #print(pointcloud_value_pro)
    # pointcloud_pcl_r = pcl.PointCloud()
    # pointcloud_pcl_r.from_array(pointcloud_pcl_radius)
    # #print(pointcloud_pcl)
    # ror = pointcloud_pcl_r.make_RadiusOutlierRemoval()
    # #print(ror)
    # ror.set_radius_search(20)
    # #print(ror.get_radius_search())
    # ror.set_MinNeighborsInRadius(3)
    # #print(ror.get_MinNeighborsInRadius())
    # pointcloud_pcl_radius_2 = ror.filter()
    # #print(pointcloud_pcl_radius)
    # pointcloud_pcl_radius_2 = pointcloud_pcl_radius_2.to_array()
    # print(pointcloud_pcl_radius_2.shape)
    pointcloud_fin = pcl.PointCloud()
    if (flag == 0):
        pointcloud_fin = pointcloud_pcl_radius.to_array()
    elif (flag == 1):
        pointcloud_fin = pointcloud_pcl_ece.to_array()
    else:
        logger.info("error: There is no pointcloud output in Data preprocessing!!!")
    
    return pointcloud_fin, confidence_value_pro

# 框选点云数据
def get_pointcloud_value(mat_input, x1, y1, x2, y2):
    mat_np = mat_input.get_data()
    data = mat_np[y1:y2,x1:x2,:3]

    return data

# 框选深度置信度数据
def get_confidence_value(mat_input, x1, y1, x2, y2):
    mat_np = mat_input.get_data()
    mat_out = mat_np[y1:y2,x1:x2]
    mat_out = 100 - mat_out

    return mat_out

# 计算三维坐标质心与深度置信度
def calculate_centriod(pointcloud, confidence_map, x, y, w, h):
    # 1920*1080 --> 1280*720
    x = x/1.5
    y = y/1.5
    w = w/1.5
    h = h/1.5
    x = x + w/6
    w = 2*w/3
    y = y + h/6
    h = h/3
    # 防止框选范围越界
    x_max = pointcloud.get_width()
    y_max = pointcloud.get_height()
    if x<0:x1 = 0
    else:x1 = x
    if (x+w)>x_max:x2 = x_max
    else:x2 = x+w
    if y<0:y1 = 0
    else:y1 = y
    if (y+h)>y_max:y2 = y_max
    else:y2 = y+h
    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)
    # 框选数据
    pointcloud_value = get_pointcloud_value(pointcloud, x1, y1, x2, y2)
    confidence_value = get_confidence_value(confidence_map, x1, y1, x2, y2)
    # 数据预处理
    [pointcloud_value_pro, confidence_value_pro] = Data_preprocess(pointcloud_value, confidence_value)
    # 计算质心与深度置信度
    pd_x = pointcloud_value_pro[:,0]
    pd_y = pointcloud_value_pro[:,1]
    pd_z = pointcloud_value_pro[:,2]
    centriod = [pd_x.mean(), pd_y.mean(), pd_z.mean()]
    depth_confidence = confidence_value_pro.mean()
    
    # print(confidence_value_pro)

    return centriod, depth_confidence, pointcloud_value_pro

def imageToROSmsg(img, frameId, t):
    img_msg = Image()
    img_msg.header.stamp.secs = t.get_seconds()
    img_msg.header.stamp.nsecs = t.get_nanoseconds() % pow(10,9)
    img_msg.header.frame_id = frameId
    img_msg.height = img.shape[0]
    img_msg.width = img.shape[1]

    #img_msg.step = img.bytes()

    #size = img_msg.step * img_msg.height

    #dataType = img.get_data_type()

    #if (dataType == sl.MAT_TYPE.F32_C1):
    #    img_msg.encoding = '32FC1'
    #elif (dataType == sl.MAT_TYPE.F32_C2):
    #    img_msg.encoding = '32FC2'
    #elif (dataType == sl.MAT_TYPE.F32_C3):
    #    img_msg.encoding = '32FC3'
    #elif (dataType == sl.MAT_TYPE.F32_C4):
    #    img_msg.encoding = '32FC4'
    #elif (dataType == sl.MAT_TYPE.U8_C1):
    #    img_msg.encoding = 'mono8'
    #elif (dataType == sl.MAT_TYPE.U8_C2):
    #    img_msg.encoding = '8UC2'
    #elif (dataType == sl.MAT_TYPE.U8_C3):
    #    img_msg.encoding = 'bgr8'
    #elif (dataType == sl.MAT_TYPE.U8_C4):
    #    img_msg.encoding = 'bgra8'
    #elif (dataType == sl.MAT_TYPE.U16_C1):
    #    img_msg.encoding = 'mono16'
    #else:
    #    print('data_type not found!!!')
    img_msg.encoding='rgb8'
    img_msg.data = np.array(img).tostring()
    return img_msg

# 行人三维坐标（相机坐标系）--> 图像二维坐标（图像坐标系）
def trans_point(point_c, pose_camera, K_in):
    #disto = sl.CameraParameters.disto
    #x = pose_camera.pose_with_cov.pose.orientation.x
    #y = pose_camera.pose_with_cov.pose.orientation.y
    #z = pose_camera.pose_with_cov.pose.orientation.z
    #w = pose_camera.pose_with_cov.pose.orientation.w
    #R_ext = [[1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
    #         [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
    #         [2*x*y-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]]
    #t_x = pose_camera.pose_with_cov.pose.position.x
    #t_y = pose_camera.pose_with_cov.pose.position.y
    #t_z = pose_camera.pose_with_cov.pose.position.z
    #s = point_c[2]
    #T_wc = [[R_ext[0][0], R_ext[0][1], R_ext[0][2], t_x],
    #        [R_ext[1][0], R_ext[1][1], R_ext[1][2], t_y], 
    #        [R_ext[2][0], R_ext[2][1], R_ext[2][2], t_z]]
    #point_c = [[point_c[0]], [point_c[1]], [point_c[2]], [1]]
    #point_w = np.matmul(T_wc, point_c)
    #point_2D = np.matmul(K_in, point_w/s)
    #point_2D = point_2D[0:2]
    #point_c[0] = point_c[0] * 1.5
    #point_c[1] = point_c[1] * 1.5
    #point_c[2] = point_c[2] * 1.5
    point_2D = np.matmul(K_in, point_c/point_c[2])
    return point_2D[0:2]

# 测试
def test_pub(pointcloud_s, point_centriod_s, point_uwb, point_laser, frame_ID, t):
    pointcloud_msg = PointCloud()
    point_centriod_msg = PointCloud()
    point_uwb_msg = PointCloud()
    point_laser_msg = PointCloud()
    pointcloud_msg.header.frame_id = frame_ID
    pointcloud_msg.header.stamp.secs = t.get_seconds()
    pointcloud_msg.header.stamp.nsecs = t.get_nanoseconds() % pow(10,9)
    # 框内点云
    for pointcloud_no in range(len(pointcloud_s)):
        for point_no in range(len(pointcloud_s[pointcloud_no])):
            point = pointcloud_s[pointcloud_no][point_no]
            #channels = ChannelFloat32()
            #channels.name = "rgb"
            #channels.values = []
            point_a = Point32()
            point_a.x = point[0]
            point_a.y = point[1]
            point_a.z = point[2]
            #channels.values.append(255.0)
            #channels.values.append(0.0)
            #channels.values.append(0.0)
            pointcloud_msg.points.append(point_a)
            #pointcloud_msg.channels.append(channels)
            
    # 质心点
    point_centriod_msg.header.frame_id = frame_ID
    point_centriod_msg.header.stamp.secs = t.get_seconds()
    point_centriod_msg.header.stamp.nsecs = t.get_nanoseconds() % pow(10,9)
    for centriod in point_centriod_s:
        #channels = ChannelFloat32()
        #channels.name = "rgb"
        #channels.values = []
        point_a = Point32()
        point_a.x = centriod[0]
        point_a.y = centriod[1]
        point_a.z = centriod[2]
        #channels.values.append(255.0)
        #channels.values.append(255.0)
        #channels.values.append(255.0)
        point_centriod_msg.points.append(point_a)
        #point_centriod_msg.channels.append(channels)
    # uwb点
    # 激光点
    pub_test1.publish(pointcloud_msg)
    pub_test2.publish(point_centriod_msg)
    return 0

# def test_pub_fast(pointcloud_s, point_centriod_s, point_uwb, point_laser, frame_ID, t):
#     pointcloud_msg = PointCloud2()
#     point_centriod_msg = PointCloud2()
#     # print(pointcloud_s)
#     #print(point_centriod_s)
#     pointcloud_msg = pcl_to_ros(pointcloud_s, frame_id=frame_ID,stamp=t)
#     point_centriod_msg = pcl_to_ros(point_centriod_s, frame_id=frame_ID, stamp=t)
#     pub_test1.publish(pointcloud_msg)
#     pub_test2.publish(point_centriod_msg)
#     return 0

# 保存数据
def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))

# 目标追踪
def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30, use_cuda=True):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    #for path, img, img0 in dataloader:
    while not rospy.is_shutdown():
        for i, (path, img, img0, pointcloud, confidence_map, zed_timestamp, zed_pose_with_cov, K_in) in enumerate(dataloader):
            #if i % 8 != 0:
                #continue
            #print(pointcloud)
            #print(img0.shape)
            # if frame_id % 20 == 0:
            #     logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

            # run tracking
            timer.tic()
            if use_cuda:
                blob = torch.from_numpy(img).cuda().unsqueeze(0)
            else:
                blob = torch.from_numpy(img).unsqueeze(0)
            online_targets = tracker.update(blob, img0)
            online_tlwhs = []
            online_ids = []
            online_dists = []
            #online_scores = []
            people_IDs = []
            origin_xs = []
            origin_ys = []
            widths = []
            heights = []
            centriod_s = [] 
            pointcloud_pro_s = []
            people_list_msg = people_list()
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                tdist = t.dist
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_dists.append(tdist)
                    #online_scores.append(t.score)
                    people_IDs.append(tid)
                    origin_xs.append(tlwh[0])
                    origin_ys.append(tlwh[1])
                    widths.append(tlwh[2])
                    heights.append(tlwh[3]) 
                    [centriod, depth_confidence, pointcloud_pro] = calculate_centriod(pointcloud, confidence_map, tlwh[0], tlwh[1], tlwh[2], tlwh[3])
                    pointcloud_pro_s.append(pointcloud_pro)
                    centriod_s.append(centriod)
                    #publish
                    #obj = people(tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3], centriod_s, tdist)
                    people_msg = people_item()
                    people_msg.people_ID = tid 
                    people_msg.width =  int(tlwh[2])
                    people_msg.height = int(tlwh[3])
                    people_msg.origin_x = int(tlwh[0])
                    people_msg.origin_y = int(tlwh[1])
                    people_msg.points.x = float(centriod[0])
                    people_msg.points.y = float(centriod[1])
                    people_msg.points.z = float(centriod[2])
                    if not(np.isnan(depth_confidence)):
                        people_msg.depth_confidence = int(depth_confidence)
                    people_list_msg.people_items.append(people_msg)
            

            timer.toc()
            # save results
            results.append((frame_id + 1, online_tlwhs, online_ids))
            #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
            #publish
            people_list_msg.header.frame_id = 'map'
            people_list_msg.header.stamp.secs = zed_timestamp.get_seconds()
            people_list_msg.header.stamp.nsecs = zed_timestamp.get_nanoseconds() % pow(10,9)
            people_list_msg.identify_time = rospy.Time.now()
            pose_msg = pose()
            pose_msg.pose_with_cov.pose.position.x = zed_pose_with_cov[0][0]
            pose_msg.pose_with_cov.pose.position.y = zed_pose_with_cov[0][1]
            pose_msg.pose_with_cov.pose.position.z = zed_pose_with_cov[0][2]
            pose_msg.pose_with_cov.pose.orientation.x = zed_pose_with_cov[1][0]
            pose_msg.pose_with_cov.pose.orientation.y = zed_pose_with_cov[1][1]
            pose_msg.pose_with_cov.pose.orientation.z = zed_pose_with_cov[1][2]
            pose_msg.pose_with_cov.pose.orientation.w = zed_pose_with_cov[1][3]
            # people_list_msg.pose_with_cov.covariance = [float('{:.4f}'.format(i)) for i in zed_pose_with_cov[2]]
            pose_msg.pose_with_cov.covariance = zed_pose_with_cov[2]
            

            if show_image or save_dir is not None:
                online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                                fps=1. / timer.average_time, dists=online_dists)
            if show_image:
                cv2.imshow('online_im', online_im)
                #cv2.imshow('img', img)
                cv2.waitKey(1)        
	

            if save_dir is not None:
                cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
            frame_id += 1
            
            img_msg = imageToROSmsg(online_im, 'map', zed_timestamp)

            #test_pub(pointcloud_s=pointcloud_pro_s, point_centriod_s=centriod_s, point_uwb=[], point_laser=[], frame_ID='map', t=zed_timestamp)
            pub_pose.publish(pose_msg)
            pub_people_list.publish(people_list_msg)
            pub_img.publish(img_msg)
            rate.sleep()
            if rospy.is_shutdown():
                break
    # save results
    write_results(result_filename, results, data_type)
    #write_results_score(result_filename, results, data_type)
    

    return frame_id, people_IDs, origin_xs, origin_ys, widths, heights, timer.average_time, timer.calls

def eval_seq_svo(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30, use_cuda=True):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    #for path, img, img0 in dataloader:
    while not rospy.is_shutdown():
        for i, (path, img, img0, pointcloud, confidence_map, zed_timestamp, K_in) in enumerate(dataloader):
            
            timer.tic()
            if use_cuda:
                blob = torch.from_numpy(img).cuda().unsqueeze(0)
            else:
                blob = torch.from_numpy(img).unsqueeze(0)
            online_targets = tracker.update(blob, img0)
            online_tlwhs = []
            online_ids = []
            online_dists = []
            #online_scores = []
            people_IDs = []
            centriod_s = [] 
            pointcloud_pro_s = []
            people_list_msg = people_list()
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                tdist = t.dist
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_dists.append(tdist)
                    #online_scores.append(t.score)
                    people_IDs.append(tid)
                    [centriod, depth_confidence, pointcloud_pro] = calculate_centriod(pointcloud, confidence_map, tlwh[0], tlwh[1], tlwh[2], tlwh[3])
                    pointcloud_pro_s.append(pointcloud_pro)
                    centriod_s.append(centriod)
                    #publish
                    #obj = people(tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3], centriod_s, tdist)
                    people_msg = people_item()
                    people_msg.people_ID = tid 
                    people_msg.width =  int(tlwh[2])
                    people_msg.height = int(tlwh[3])
                    people_msg.origin_x = int(tlwh[0])
                    people_msg.origin_y = int(tlwh[1])
                    people_msg.points.x = float(centriod[0])
                    people_msg.points.y = float(centriod[1])
                    people_msg.points.z = float(centriod[2])
                    if not(np.isnan(depth_confidence)):
                        people_msg.depth_confidence = int(depth_confidence)
                    people_list_msg.people_items.append(people_msg)
            
            timer.toc()
            # save results
            results.append((frame_id + 1, online_tlwhs, online_ids))
            #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
            #publish
            people_list_msg.header.frame_id = 'map'
            people_list_msg.header.stamp.secs = zed_timestamp.get_seconds()
            people_list_msg.header.stamp.nsecs = zed_timestamp.get_nanoseconds() % pow(10,9)
            people_list_msg.identify_time = rospy.Time.now()
            pose_msg = pose()
            
            if show_image or save_dir is not None:
                online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                                fps=1. / timer.average_time, dists=online_dists)
            if show_image:
                cv2.imshow('online_im', online_im)
                #cv2.imshow('img', img)
                cv2.waitKey(1)        
	

            if save_dir is not None:
                cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
            frame_id += 1
            
            img_msg = imageToROSmsg(online_im, 'map', zed_timestamp)

            test_pub(pointcloud_s=pointcloud_pro_s, point_centriod_s=centriod_s, point_uwb=[], point_laser=[], frame_ID='map', t=zed_timestamp)
            pub_pose.publish(pose_msg)
            pub_people_list.publish(people_list_msg)
            pub_img.publish(img_msg)
            rate.sleep()
            if rospy.is_shutdown():
                break
    # save results
    write_results(result_filename, results, data_type)
    #write_results_score(result_filename, results, data_type)
    

    return frame_id, people_IDs, timer.average_time, timer.calls

# zed目标追踪
def recogniton():
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)
    print("start tracking")
    if opt.Mode == 'ZED':
        dataloader = datasets.LoadVideo(0, opt.img_size)
    elif opt.Mode == 'SVO':
        dataloader = datasets.LoadSVO(0, opt.img_size)
    else:
        print("please specify command line arg paramter 'ZED' or 'FILES', default is ZED")
        dataloader = datasets.LoadVideo(0, opt.img_size)
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate
    frame_dir = None if opt.output_format == 'text' else os.path.join(result_root, 'frame')
    
    if opt.Mode == 'ZED':
        eval_seq(opt, dataloader, 'mot', result_filename,
                save_dir=frame_dir, show_image=True, frame_rate=frame_rate)
    elif opt.Mode == 'SVO':
        eval_seq_svo(opt, dataloader, 'mot', result_filename,
                save_dir=frame_dir, show_image=True, frame_rate=frame_rate)
    else:
        print("please specify command line arg paramter 'ZED' or 'FILES', default is ZED")

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    # demo(opt)
    pub_people_list = rospy.Publisher('people_list', people_list, queue_size=10)
    pub_pose = rospy.Publisher('pose', pose, queue_size=10)
    pub_img = rospy.Publisher('img', Image, queue_size=10)
    # 可视化
    pub_test1 = rospy.Publisher('people_pointcloud', PointCloud, queue_size=10)
    pub_test2 = rospy.Publisher('people_centriod', PointCloud, queue_size=10)

    rospy.init_node('camera', anonymous=True)
    rate = rospy.Rate(20)
    recogniton()
       
