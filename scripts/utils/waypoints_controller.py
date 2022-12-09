#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PointStamped
import tf
import random
import numpy as np

global waypoints
waypoints = []

class WayPointsController:
    def __init__(self):
        rospy.init_node('goal_publisher', anonymous=True)
        rospy.point_pub = rospy.Subscriber('/clicked_point', PointStamped, callback)

    def rviz_goal_publisher_callback(msg): 
        if rospy.get_param("waypointCreation", True):
            point = PointStamped()
            point.header.stamp = rospy.Time.now()
            point.header.frame_id = "/map"
            x = msg.point.x
            y = msg.point.y
            z = msg.point.z
            self.waypoints.add((x,y,z))
            rospy.loginfo("coordinates:x=%f y=%f" %(x,y))

if __name__ == '__main__':
    sup = WayPointsController()
    try:
        sup.loop()
    except rospy.ROSInterruptException:
        pass  