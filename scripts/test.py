#!/usr/bin/env python
#   /home/ww/catkin_ws/src/fairmot_ros_wzq/msg
from re import U
import numpy as np
import rospy
#import pcl
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud, PointCloud2
from sensor_msgs import point_cloud2
from nlink_example.msg import UwbFilter
from fairmot_ros_wzq.msg import people_list
import message_filters



# def callback(people_centriod, rslidar_points, uwb_filter):
def callback(people, uwb_filter):
    # points = point_cloud2.read_points_list(rslidar_points)
    # points = np.array(points, dtype=np.float32)
    
    #uwb_point = [uwb_filter.point.x,uwb_filter.point.y, uwb_filter.point.z]
    
    # print(points[0])
    # print(people.points)
    # print(uwb_point)
    file = open(r'../data/data-v2.1.txt', mode='a')
    file.write(str(uwb_filter))
    file.write('\n')
    # file.write(str(people)[str(people).find('people_items'):])
    file.write(str(people))
    file.write('\n')
    file.close()

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
    listener()
