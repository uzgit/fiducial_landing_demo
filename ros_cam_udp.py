#!/usr/bin/python3

# imports for environment
import os
import signal
import subprocess
from datetime import datetime, timedelta

# imports for receiving/transmitting images/data
import socket
import cv2
import numpy

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

def constrain(input_value, minimum=-1.0, maximum=1.0):
    
    result = input_value
    if( result < minimum ):
        result = minimum
    if( result > maximum ):
        result = maximum

    return result

class ControlPolicy:

    class Timer():
        def __init__(self, time=None):

            if( time == None ):
                self.time = datetime.now()
            else:
                self.time = time
            return

        def reset(self):
            
            self.time = datetime.now()

        def __gt__(self, seconds):

            result = datetime.now() - self.time > timedelta(seconds=seconds)
            return result
        
        def __ge__(self, seconds):

            result = datetime.now() - self.time >= timedelta(seconds=seconds)
            return result
        
        def __lt__(self, seconds):

            result = datetime.now() - self.time < timedelta(seconds=seconds)
            return result
        
        def __le__(self, seconds):

            result = datetime.now() - self.time <= timedelta(seconds=seconds)
            return result

    def __init__(self, procedure="raw"):
        self.procedure = procedure
        self.mode = "tracking"
        self.c_normalized      = [None, None]
        self.position_target_e = None
        self.position_target_n = None
        self.position_target_u = None
        self.yaw_displacement  = None
        self.command           = None

        self.gimbal_imu_pitch  = None
        self.gimbal_imu_roll   = None
        self.altitude          = None

        self.detection_timeout    = 0.5 # seconds
        self.max_yaw_displacement = 0.1 # radians
        self.operational_altitude = 3 # meters


        # instantiate timers
        now = datetime.now()
        self.at_altitude_timer      = self.Timer()
        self.detection_timer        = self.Timer()
        self.close_radius_timer     = self.Timer()
        self.yaw_aligned_timer      = self.Timer()
        self.landing_altitude_timer = self.Timer()

        # instantiate filters
        self.filter_x = KalmanFilter(dim_x=2, dim_z=1)
        self.filter_y = KalmanFilter(dim_x=2, dim_z=1)
        self.filter_z = KalmanFilter(dim_x=2, dim_z=1)

        for filter_ in [self.filter_x, self.filter_y, self.filter_z]:
            filter_.x = numpy.array([0, 0])
            filter_.F = numpy.array([[1, 1], [0, 1]])
            filter_.H = numpy.array([[1,0]])
            filter_.P *= 1000
            filter_.R = numpy.array([[5]])
   
    def update_values(self, data):

        if( "gimbal_imu_pitch" in data ):
            self.gimbal_imu_pitch = data["gimbal_imu_pitch"]
        if( "gimbal_imu_roll" in data ):
            self.gimbal_imu_roll = data["gimbal_imu_roll"]
        if( "altitude" in data ):
            self.altitude = data["altitude"]

    def set_mode(self, mode):
        self.mode = mode

    def get_mode(self):
        return self.mode

    def set_detection(self,
            c_normalized,
            position,
            position_target_e,
            position_target_n,
            position_target_u,
            yaw_displacement):
        self.c_normalized = list(c_normalized).copy()
        self.position = position
        self.position_target_e = position_target_e
        self.position_target_n = position_target_n
        self.position_target_u = position_target_u
        self.yaw_displacement  = yaw_displacement

        self.detection_timer.reset()

        self.update_command()
        
    def update_command(self):

        yaw         = None
        gimbal_tilt = None
        pitch       = None
        roll        = None
        throttle    = None

        if( "raw" == self.procedure ):
            yaw         = +1.0 * self.c_normalized[0]
            gimbal_tilt = -0.5 * self.c_normalized[1]
            pitch       = +1.0 * self.position_target_n
            roll        = +1.0 * self.position_target_e
            throttle    = +1.0 * self.position_target_u

        elif( "filtered" == self.procedure ):

            yaw         = +1.0 * self.c_normalized[0]
            gimbal_tilt = -0.5 * self.c_normalized[1]         # check this later (also scaled in app)
            pitch       = +1.0 * self.position_target_n
            roll        = +1.0 * self.position_target_e
            throttle    = +1.0 * self.position_target_u

            self.filter_x.predict()
            self.filter_y.predict()
            self.filter_z.predict()

            self.filter_x.update(pitch)
            self.filter_y.update(roll)
            self.filter_z.update(throttle)

            pitch = self.filter_x.x[0]
            roll  = self.filter_y.x[0]
            yaw   = self.filter_z.x[0]

        elif( "imu" == self.procedure ):

            #position = self.pose.pose.pose.position
            position = self.position
            
            # convert IMU orientation Euler components into radians from degrees
            gimbal_imu_pitch_rad = self.gimbal_imu_pitch * numpy.pi / 180
            gimbal_imu_roll_rad  = self.gimbal_imu_roll * numpy.pi / 180

            # put the orientation of the gimabl IMU into a Python quaternion
            orientation = pyquaternion.Quaternion(get_quaternion_from_euler(gimbal_imu_roll_rad, gimbal_imu_pitch_rad, 0))

            # do transform
            position_target = orientation.rotate(numpy.array([position.x, position.y, position.z]))
            
            # set commands
            yaw             = +1.0 * self.c_normalized[0]
            gimbal_tilt     = -0.5 * self.c_normalized[1]
            roll            = +1.0 * position_target[1]
            throttle        = +1.0 * position_target[0]
            pitch           = +1.0 * position_target[2]
        
        # update timers
        now = datetime.now()
        if( self.altitude < self.operational_altitude ):
            self.at_altitude_timer.reset()
        if( numpy.sqrt(pitch ** 2 + roll ** 2) >= 0.5 ):
            self.close_radius_timer.reset()
        if( self.yaw_displacement > self.max_yaw_displacement ):
            self.yaw_aligned_timer.reset()
        if( self.altitude > 1 ):
            self.landing_altitude_timer.reset()

            
        if( self.detection_timer < self.detection_timeout ):

            if( "descent" == self.get_mode() and now - self.landing_altitude_timer >= 2 ):
                self.set_mode("land")
            elif( "alignment" == self.get_mode() and self.close_radius_timer >= 5 ):
                self.set_mode("descent")
            elif( "approach" == self.get_mode() and self.close_radius_timer >= 5 ):
                self.set_mode("alignment")

        elif( self.detection_timer < 5 ):

            self.set_mode("tracking")

        elif( self.detection_timer >= 5 and "search" != self.get_mode() ):
            
            self.set_mode("search")
            
            self.at_altitude_timer.reset()
            self.detection_timer.reset()
            self.close_radius_timer.reset()
            self.yaw_aligned_timer.reset()
            self.landing_altitude_timer.reset()

        # constrain
        yaw         = constrain(yaw)
        gimbal_tilt = constrain(gimbal_tilt)
        roll        = constrain(roll)
        throttle    = constrain(throttle)
        pitch       = constrain(pitch)

        # output
        self.command = f"@,{now},{self.get_mode()},{yaw:+0.5f},{gimbal_tilt:+0.5f},{pitch:+0.5f},{roll:+0.5f},{throttle:+0.5f},&"

    def get_command(self):

        result = None

        if( self.detection_timer < self.detection_timeout ):
            result = self.command
        else:
            print("invalid command, sending heartbeat!")

        return result

    def get_command_reset(self):

        result = self.get_command()
        self.command = None
        return result

    def __repr__(self):
        return self.command

    def __str__(self):
        return self.__repr__()


command = ""
def apriltag_callback( apriltag_detection_array ):

    global control_policy

    valid_detections = [detection for detection in apriltag_detection_array.detections if "landing_pad" in detection.name]
    try:
        if( len(valid_detections) > 0 ):
            target = valid_detections[0]
            control_policy.set_detection(
                    target.c_normalized,
                    target.pose.pose.pose.position,
                    target.position_target_enu.x,
                    target.position_target_enu.y,
                    target.position_target_enu.z,
                    target.yaw
                    )
    except Exception as e:
        print(e)
        print("error in setting target")

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

    global gimbal_imu_pitch
    global gimbal_imu_roll
    global altitude
    global control_policy
    control_policy = ControlPolicy(procedure="imu")

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
    address = "192.168.1.100"
    port = 14555
    print(f"opening socket at {address}:{port}")
    server_address = (address, port)
    sock.bind(server_address)
    print("bound!")

    while True:

        # receive data from the server
        data, client_address = sock.recvfrom(1000000)

        # if we receive info
        if( len(data) <= 500 ):

            try:
                data = eval(data.decode())
                
                control_policy.update_values( data )

            except:
                print("error parsing info message from tablet")

        # if we receive an image
        elif( len(data) > 100 ):
            # decode the image data and display it using OpenCV
            image = cv2.imdecode(numpy.frombuffer(data, numpy.uint8), cv2.IMREAD_UNCHANGED)
            publish_image(image, image_publisher, info_publisher, camera_info)

        command = control_policy.get_command_reset()
        if( command == None ):
            command = "HEARTBEAT"
        else:
            print(command)
        sock.sendto( command.encode(), client_address)

    # Don't forget to clean up when you're done!
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
