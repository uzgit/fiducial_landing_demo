#!/usr/bin/python3


# imports for environment
import os
import signal
import subprocess

# imports for receiving/transmitting images/data
import socket
import cv2
import numpy
import datetime

# imports for ROS
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image as _Image
from sensor_msgs.msg import CameraInfo
import yaml
from apriltag_ros.msg import AprilTagDetection, AprilTagDetectionArray

import numpy
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

import pyquaternion

def sigint_handler(signum, frame):
    print("Ctrl+C break", flush=True)
    print()
    exit(1)
signal.signal(signal.SIGINT, sigint_handler)

def get_quaternion_from_euler(roll, pitch, yaw):
  """
  Convert an Euler angle to a quaternion.
   
  Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
 
  Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
  """
  qx = numpy.sin(roll/2) * numpy.cos(pitch/2) * numpy.cos(yaw/2) - numpy.cos(roll/2) * numpy.sin(pitch/2) * numpy.sin(yaw/2)
  qy = numpy.cos(roll/2) * numpy.sin(pitch/2) * numpy.cos(yaw/2) + numpy.sin(roll/2) * numpy.cos(pitch/2) * numpy.sin(yaw/2)
  qz = numpy.cos(roll/2) * numpy.cos(pitch/2) * numpy.sin(yaw/2) - numpy.sin(roll/2) * numpy.sin(pitch/2) * numpy.cos(yaw/2)
  qw = numpy.cos(roll/2) * numpy.cos(pitch/2) * numpy.cos(yaw/2) + numpy.sin(roll/2) * numpy.sin(pitch/2) * numpy.sin(yaw/2)
 
  return numpy.array([qw, qx, qy, qz])

# instantiate filters
filter_x = KalmanFilter(dim_x=2, dim_z=1)
filter_y = KalmanFilter(dim_x=2, dim_z=1)
filter_z = KalmanFilter(dim_x=2, dim_z=1)

for filter_ in [filter_x, filter_y, filter_z]:
    filter_.x = numpy.array([0, 0])
    filter_.F = numpy.array([[1, 1], [0, 1]])
    filter_.H = numpy.array([[1,0]])
    filter_.P *= 1000
    filter_.R = numpy.array([[5]])

# instantiate variable
gimbal_imu_pitch = None
gimbal_imu_roll  = None

class ControlPolicy:

    def __init__(self):
        self.mode = "tracking"
        self.c_normalized = [None, None]
        self.position_target_e = None
        self.position_target_n = None
        self.position_target_u = None
        self.yaw_displacement  = None

    def set_detection(self,
            c_normalized,
            position_target_e,
            position_target_n,
            position_target_u,
            yaw_displacement):
        self.c_normalized = list(c_normalized).copy()
        self.position_target_e = position_target_e
        self.position_target_n = position_target_n
        self.position_target_u = position_target_u
        self.yaw_displacement  = yaw_displacement

    def __repr__(self):
        #command = f"@,{rospy.get_time():+f},{mode},{yaw:+0.5f},{gimbal_tilt:+0.5f},{pitch:+0.5f},{roll:+0.5f},{throttle:+0.5f},&"
        #return f"@,{rospy.get_time():+f},{self.mode},{yaw:+0.5f},{gimbal_tilt:+0.5f},{pitch:+0.5f},{roll:+0.5f},{throttle:+0.5f},&"
        return f"{self.c_normalized}, [{self.position_target_e}, {self.position_target_n}, {self.position_target_u}], {self.yaw_displacement}"

control_policy = ControlPolicy()

command = ""
def apriltag_callback( apriltag_detection_array ):

    global command

    global filter_x
    global filter_y
    global filter_z
    global gimbal_imu_pitch
    global gimbal_imu_roll
    global altitude

    global control_policy

    procedure = "imu_transform"

    mode = "tracking"
    target = None
    valid_detections = [detection for detection in apriltag_detection_array.detections if "landing_pad" in detection.name]
    if( len(valid_detections) > 0 ):
        target = valid_detections[0]

        if( "raw" == procedure ):
            yaw         = +1.0 * target.c_normalized[0]
            gimbal_tilt = -0.5 * target.c_normalized[1]
            pitch       = +1.0 * target.position_target_enu.y
            roll        = +1.0 * target.position_target_enu.x
            throttle    = +1.0 * target.position_target_enu.z
            
            control_policy.set_detection(
                    target.c_normalized,
                    target.position_target_enu.x,
                    target.position_target_enu.y,
                    target.position_target_enu.z,
                    yaw
                    )
            
            command = f"@,{rospy.get_time():+f},{mode},{yaw:+0.5f},{gimbal_tilt:+0.5f},{pitch:+0.5f},{roll:+0.5f},{throttle:+0.5f},&"

        elif( "filtered" == procedure ):

            yaw         = +1.0 * target.c_normalized[0]
            gimbal_tilt = -0.5 * target.c_normalized[1]         # check this later (also scaled in app)
            pitch       = +1.0 * target.position_target_enu.y
            roll        = +1.0 * target.position_target_enu.x
            throttle    = +1.0 * target.position_target_enu.z

            filter_x.predict()
            filter_y.predict()
            filter_z.predict()

            filter_x.update(pitch)
            filter_y.update(roll)
            filter_z.update(throttle)

            pitch = filter_x.x[0]
            roll  = filter_y.x[0]
            yaw   = filter_z.x[0]

            control_policy.set_detection(
                    target.c_normalized,
                    target.position_target_enu.x,
                    target.position_target_enu.y,
                    target.position_target_enu.z,
                    yaw
                    )
            
            command = f"@,{rospy.get_time():+f},{mode},{yaw:+0.5f},{gimbal_tilt:+0.5f},{pitch:+0.5f},{roll:+0.5f},{throttle:+0.5f},&"

        elif( "imu_transform" == procedure ):


            position = target.pose.pose.pose.position
            #print(target.pose.pose.pose.position)

            
            # convert IMU orientation Euler components into radians from degrees
#            if( gimbal_imu_pitch is None or gimbal_imu_roll is None ):
#                print("gimbal pitch/roll not defined! cannot do transform!")
#                return 

            #print(f"{gimbal_imu_pitch:+3.4f} {gimbal_imu_roll:+3.4f}")
            gimbal_imu_pitch_rad = gimbal_imu_pitch * numpy.pi / 180
            gimbal_imu_roll_rad  = gimbal_imu_roll * numpy.pi / 180

            # put the orientation of the gimabl IMU into a Python quaternion
            orientation = pyquaternion.Quaternion(get_quaternion_from_euler(gimbal_imu_roll_rad, gimbal_imu_pitch_rad, 0))
            #print(orientation)

            # do transform
            position_target = orientation.rotate(numpy.array([position.x, position.y, position.z]))
            
            # set commands
            yaw             = +1.0 * target.c_normalized[0]
            gimbal_tilt     = -0.5 * target.c_normalized[1]
            roll            = +1.0 * position_target[1]
            throttle        = +1.0 * position_target[0]
            pitch           = +1.0 * position_target[2]
            
            control_policy.set_detection(
                    target.c_normalized,
                    target.position_target_enu.x,
                    target.position_target_enu.y,
                    target.position_target_enu.z,
                    yaw
                    )

            command = f"@,{rospy.get_time():+f},{mode},{yaw:+0.5f},{gimbal_tilt:+0.5f},{pitch:+0.5f},{roll:+0.5f},{throttle:+0.5f},&"

#    if( target is not None ):
#        command = f"@,{rospy.get_time():+f},{mode},{yaw:+0.5f},{gimbal_tilt:+0.5f},{pitch:+0.5f},{roll:+0.5f},{throttle:+0.5f},&"

def yaml_to_CameraInfo( yaml_fname ):
    """
    Parse a yaml file containing camera calibration data (as produced by
    rosrun camera_calibration cameracalibrator.py) into a
    sensor_msgs/CameraInfo msg.

    Parameters
    ----------
    yaml_fname : str
        Path to yaml file containing camera calibration data
    Returns
    -------
    camera_info_msg : sensor_msgs.msg.CameraInfo
        A sensor_msgs.msg.CameraInfo message containing the camera calibration
        data
    """
    # Load data from file
    with open(yaml_fname, "r") as file_handle:
        calib_data = yaml.load(file_handle)
    # Parse
    camera_info_msg = CameraInfo()
    camera_info_msg.width = calib_data["image_width"]
    camera_info_msg.height = calib_data["image_height"]
    camera_info_msg.K = calib_data["camera_matrix"]["data"]
    camera_info_msg.D = calib_data["distortion_coefficients"]["data"]
    camera_info_msg.R = calib_data["rectification_matrix"]["data"]
    camera_info_msg.P = calib_data["projection_matrix"]["data"]
    camera_info_msg.distortion_model = calib_data["distortion_model"]
    return camera_info_msg

def publish_image( image, image_publisher, info_publisher=None, camera_info=None):

    bridge = CvBridge()
    image_message = bridge.cv2_to_imgmsg(image, "bgr8")

    now = rospy.Time.now()

    image_message.header.stamp = now
    
    if( info_publisher is not None and camera_info is not None ):
        camera_info.header.stamp = now
        info_publisher.publish(camera_info)
    image_publisher.publish( image_message )

def main():

    global command
    global gimbal_imu_pitch
    global gimbal_imu_roll
    global altitude
    global control_policy

    os.environ["DISPLAY"] = ":0"
    
    # initialize ROS infrastructure
    rospy.init_node("image_node", anonymous=True)
    image_publisher = rospy.Publisher('/gcs/image_raw', _Image, queue_size=10)
    info_publisher = rospy.Publisher("/gcs/camera_info", CameraInfo, queue_size=10)
    camera_info = yaml_to_CameraInfo("/home/joshua/Documents/calibrationdata/ost.yaml")
    rospy.Subscriber("/tag_detections", AprilTagDetectionArray, apriltag_callback)

    # create a UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # bind the socket to the address and port
    #server_address = ('192.168.1.100', 14555)

    print("getting own address...", end="")
    #address = socket.gethostbyname("gcs.local")
    address = subprocess.check_output(command, shell=True).decode().strip()
    print("found {address}")
    port = 14555
    print("opening socket at {address}:{port}")
    server_address = (address, port)
    sock.bind(server_address)
    print("bound!")

    while True:

        #print("waiting for data...")

        # receive data from the server
        data, client_address = sock.recvfrom(1000000)

        # if we receive the tilt of the gimbal
        if( len(data) <= 100 ):

            try:
                data = eval(data.decode())
                
                gimbal_imu_pitch = data["gimbal_pitch"]
                gimbal_imu_roll = data["gimbal_roll"]
                altitude = data["altitude"]
                print()
#                print(f"{gimbal_imu_pitch=:+3.4}, {gimbal_imu_roll=:+3.4}")
                #print(data)
                print()
            except:
                print("idk")
#            gimbal_tilt = float(data)
#            print(float(data))
#            print(len(data))

        # if we receive an image
        elif( len(data) > 100 ):
            # decode the image data and display it using OpenCV
            image = cv2.imdecode(numpy.frombuffer(data, numpy.uint8), cv2.IMREAD_UNCHANGED)
            publish_image(image, image_publisher, info_publisher, camera_info)

        print(control_policy)

        if( command == None ):
            command = "HEARTBEAT"
        print(command)
#        print(command.replace(".","").encode())
        sock.sendto( command.encode(), client_address)
#        print(command)
        command = None

#        print()
#        cv2.imshow('image', image)
#        cv2.waitKey(1)

    # Don't forget to clean up when you're done!
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
