#!/usr/bin/env python
# import rospy
# from std_msgs.msg import String
# import pcl

# def test_radius(pointcloud_pcl_radius):
#     pointcloud_pcl_r = pcl.PointCloud()
#     pointcloud_pcl_r.from_array(pointcloud_pcl_radius)
#     #print(pointcloud_pcl)
#     ror = pointcloud_pcl_r.make_RadiusOutlierRemoval()
#     #print(ror)
#     ror.set_radius_search(20)
#     #print(ror.get_radius_search())
#     ror.set_MinNeighborsInRadius(3)
#     #print(ror.get_MinNeighborsInRadius())
#     pointcloud_pcl_radius_2 = ror.filter()
#     #print(pointcloud_pcl_radius)
#     pointcloud_pcl_radius_2 = pointcloud_pcl_radius_2.to_array()
#     print(pointcloud_pcl_radius_2.shape)

# def talker():
#     pub = rospy.Publisher('chatter1', String, queue_size=10)
#     rospy.init_node('talker1', anonymous=True)
#     rate = rospy.Rate(3) # 10hz
#     while not rospy.is_shutdown():
#         hello_str = "hello world 1 %s" % rospy.get_time()
#         rospy.loginfo(hello_str)
#         pub.publish(hello_str)
#         rate.sleep()
# if __name__ == '__main__':
#     try:
#         talker()
#     except rospy.ROSInterruptException:
#         pass
#     # test_radius()

# eu cluster
import pcl
import numpy as np


import rospy
from sensor_msgs.msg import PointCloud



def callback(cloud_arr):
    cloud_data = np.zeros((len(cloud_arr.points), 3), dtype=np.float32)
    for i, point in enumerate(cloud_arr.points):
        cloud_data[i][0] = point.x
        cloud_data[i][1] = point.y
        cloud_data[i][2] = point.z
    cloud = pcl.PointCloud()
    cloud.from_list(cloud_data)

    tree = cloud.make_kdtree()
    # tree = cloud_filtered.make_kdtree_flann()

    # std::vector<pcl::PointIndices> cluster_indices;
    # pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    # ec.setClusterTolerance (0.02); // 2cm
    # ec.setMinClusterSize (100);
    # ec.setMaxClusterSize (25000);
    # ec.setSearchMethod (tree);
    # ec.setInputCloud (cloud_filtered);
    # ec.extract (cluster_indices);
    ec = cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.1)
    ec.set_MinClusterSize(10000)
    # ec.set_MaxClusterSize(25000)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()

    print('Cloud before filtering: ')
    print(cloud.size)
    print(len(cluster_indices[0]))

def listener():
    rospy.init_node("test1", anonymous=True)
    rospy.Subscriber("people_pointcloud", PointCloud, callback)

    rospy.spin()

if __name__ == "__main__":
    listener()


# -*- coding: utf-8 -*-
# port of
# http://pointclouds.org/documentation/tutorials/cylinder_segmentation.php
# you need to download
# http://svn.pointclouds.org/data/tutorials/table_scene_mug_stereo_textured.pcd

# import pcl
# import numpy as np


# import rospy
# from sensor_msgs.msg import PointCloud

# def callback(cloud_arr):
    
#     cloud_data = np.zeros((len(cloud_arr.points), 3), dtype=np.float32)
#     for i, point in enumerate(cloud_arr.points):
#         cloud_data[i][0] = point.x
#         cloud_data[i][1] = point.y
#         cloud_data[i][2] = point.z
#     cloud = pcl.PointCloud()
#     cloud_filtered = pcl.PointCloud()
#     cloud.from_list(cloud_data)
#     print(cloud.size)

#     fil = cloud.make_passthrough_filter()
#     fil.set_filter_field_name("z")
#     fil.set_filter_limits(0, 1.5)
#     cloud_filtered = fil.filter()

#     print(cloud_filtered.size)

#     seg = cloud_filtered.make_segmenter_normals(ksearch=50)
#     seg.set_optimize_coefficients(True)
#     seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
#     seg.set_normal_distance_weight(0.1)
#     seg.set_method_type(pcl.SAC_RANSAC)
#     seg.set_max_iterations(100)
#     seg.set_distance_threshold(0.03)
#     indices, model = seg.segment()

#     print(model)

#     # cloud_plane = cloud_filtered.extract(indices, negative=False)
#     # NG : const char* not str
#     # cloud_plane.to_file('table_scene_mug_stereo_textured_plane.pcd')
#     # pcl.save(cloud_plane, 'table_scene_mug_stereo_textured_plane.pcd')

#     cloud_cyl = cloud_filtered.extract(indices, negative=True)

#     seg = cloud_cyl.make_segmenter_normals(ksearch=50)
#     seg.set_optimize_coefficients(True)
#     seg.set_model_type(pcl.SACMODEL_CYLINDER)
#     seg.set_normal_distance_weight(0.1)
#     seg.set_method_type(pcl.SAC_RANSAC)
#     seg.set_max_iterations(10000)
#     seg.set_distance_threshold(0.05)
#     seg.set_radius_limits(0, 0.1)
#     indices, model = seg.segment()

#     print(model)

#     # cloud_cylinder = cloud_cyl.extract(indices, negative=False)
#     # NG : const char* not str
#     # cloud_cylinder.to_file("table_scene_mug_stereo_textured_cylinder.pcd")
#     # pcl.save(cloud_cylinder, 'table_scene_mug_stereo_textured_cylinder.pcd')


# def listener():
#     rospy.init_node("test1", anonymous=True)
#     rospy.Subscriber("people_pointcloud", PointCloud, callback)

#     rospy.spin()

# if __name__ == "__main__":
#     listener()



# import pcl
# import numpy as np
# import random

# import argparse



# parser = argparse.ArgumentParser(
#     description='PointCloudLibrary example: Remove outliers')
# parser.add_argument('--Removal', '-r', choices=('Radius', 'Condition'), default='',
#                     help='RadiusOutlier/Condition Removal')
# args = parser.parse_args()


# def main():
#     # pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
#     # pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
#     cloud = pcl.PointCloud()
#     cloud_filtered = pcl.PointCloud()

#     # // Fill in the cloud data
#     # cloud->width  = 5;
#     # cloud->height = 1;
#     # cloud->points.resize (cloud->width * cloud->height);
#     #
#     # for (size_t i = 0; i < cloud->points.size (); ++i)
#     # {
#     # cloud->points[i].x = 1024 * rand () / (RAND_MAX + 1.0f);
#     # cloud->points[i].y = 1024 * rand () / (RAND_MAX + 1.0f);
#     # cloud->points[i].z = 1024 * rand () / (RAND_MAX + 1.0f);
#     # }
#     #
#     # x,y,z
#     points = np.zeros((5, 3), dtype=np.float32)
#     RAND_MAX = 1024.0
#     for i in range(0, 5):
#         points[i][0] = 1024 * random.random() / RAND_MAX
#         points[i][1] = 1024 * random.random() / RAND_MAX
#         points[i][2] = 1024 * random.random() / RAND_MAX

#     cloud.from_array(points)

#     # if (strcmp(argv[1], "-r") == 0)
#     # {
#     # pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
#     # // build the filter
#     # outrem.setInputCloud(cloud);
#     # outrem.setRadiusSearch(0.8);
#     # outrem.setMinNeighborsInRadius (2);
#     # // apply filter
#     # outrem.filter (*cloud_filtered);
#     # }
#     # else if (strcmp(argv[1], "-c") == 0)
#     # {
#     #   // build the condition
#     #   pcl::ConditionAnd<pcl::PointXYZ>::Ptr range_cond (new pcl::ConditionAnd<pcl::PointXYZ> ());
#     #   range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr (
#     #       new pcl::FieldComparison<pcl::PointXYZ> ("z", pcl::ComparisonOps::GT, 0.0)));
#     #
#     #   range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr (
#     #       new pcl::FieldComparison<pcl::PointXYZ> ("z", pcl::ComparisonOps::LT, 0.8)));
#     #
#     #   // build the filter
#     #   pcl::ConditionalRemoval<pcl::PointXYZ> condrem (range_cond);
#     #   condrem.setInputCloud (cloud);
#     #   condrem.setKeepOrganized(true);
#     #   // apply filter
#     #   condrem.filter (*cloud_filtered);
#     # }
#     # else
#     # {
#     #   std::cerr << "please specify command line arg '-r' or '-c'" << std::endl;
#     #   exit(0);
#     # }
#     if args.Removal == 'Radius':
#         outrem = cloud.make_RadiusOutlierRemoval()
#         outrem.set_radius_search(0.8)
#         outrem.set_MinNeighborsInRadius(1)
#         cloud_filtered = outrem.filter()
#     elif args.Removal == 'Condition':
#         range_cond = cloud.make_ConditionAnd()

#         range_cond.add_Comparison2('z', pcl.CythonCompareOp_Type.GT, 0.0)
#         range_cond.add_Comparison2('z', pcl.CythonCompareOp_Type.LT, 0.8)

#         # build the filter
#         condrem = cloud.make_ConditionalRemoval(range_cond)
#         condrem.set_KeepOrganized(True)
#         # apply filter
#         cloud_filtered = condrem.filter()

#         # Test
#         # cloud_filtered = cloud
#     else:
#         print("please specify command line arg paramter 'Radius' or 'Condition'")

#     # std::cerr << "Cloud before filtering: " << std::endl;
#     # for (size_t i = 0; i < cloud->points.size (); ++i)
#     # std::cerr << "    " << cloud->points[i].x << " "
#     #                     << cloud->points[i].y << " "
#     #                     << cloud->points[i].z << std::endl;
#     # // display pointcloud after filtering
#     print('Cloud before filtering: ')
#     for i in range(0, cloud.size):
#         print('x: ' + str(cloud[i][0]) + ', y : ' +
#               str(cloud[i][1]) + ', z : ' + str(cloud[i][2]))

#     # std::cerr << "Cloud after filtering: " << std::endl;
#     # for (size_t i = 0; i < cloud_filtered->points.size (); ++i)
#     # std::cerr << "    " << cloud_filtered->points[i].x << " "
#     #                     << cloud_filtered->points[i].y << " "
#     #                     << cloud_filtered->points[i].z << std::endl;
#     print('Cloud after filtering: ')
#     for i in range(0, cloud_filtered.size):
#         print('x: ' + str(cloud_filtered[i][0]) + ', y : ' + str(
#             cloud_filtered[i][1]) + ', z : ' + str(cloud_filtered[i][2]))


# if __name__ == "__main__":
#     # import cProfile
#     # cProfile.run('main()', sort='time')
#     main()



