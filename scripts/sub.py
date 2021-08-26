#!/usr/bin/env python
import rospy
from fairmot_ros_wzq.msg import people_item

def callback(people_item):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", people_item)

def listener():
    rospy.init_node('sub',anonymous=True)
    rospy.Subscriber('people_item', people_item, callback)

    rospy.spin()

if __name__ == '__main__':
    listener()