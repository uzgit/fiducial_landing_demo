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
from geometry_msgs.msg import Point

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

def constrain(input_value, minimum=-0.5, maximum=0.5):
    
    result = input_value
    if( result < minimum ):
        result = minimum
    if( result > maximum ):
        result = maximum

    return result

def apply_deadzone( input_value, deadzone=0.025 ):

    result = input_value

    if( abs(result) < deadzone ):
        result = 0

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

        def __mod__(self, period):
            
            return (datetime.now() - self.time).total_seconds() // period % 2 == 0

        def __repr__(self):

            return str(datetime.now() - self.time)

    def __init__(self, procedure="raw"):
        self.procedure = procedure
        self.mode = "land"
        self.c_normalized      = [0, 0]
        self.position_target_e = 0
        self.position_target_n = 0
        self.position_target_u = 0
        self.position          = Point()
        self.yaw_displacement  = 0
        self.command           = None

        self.mode_change       = True

        self.gimbal_imu_pitch  = None
        self.gimbal_imu_roll   = None
        self.altitude          = 0

        self.detection_timeout    = 0.5 # seconds
        self.max_yaw_displacement = 0.1 # radians
        self.operational_altitude = 3 # meters

        # instantiate timers
        now = datetime.now()
        self.at_altitude_timer      = self.Timer()
        self.detection_timer        = self.Timer(datetime.now() - timedelta(days=1))
        self.close_radius_timer     = self.Timer()
        self.landing_radius_timer   = self.Timer()
        self.yaw_aligned_timer      = self.Timer()
        self.landing_altitude_timer = self.Timer()
        self.current_mode_timer     = self.Timer()

        self.timers = [self.at_altitude_timer, self.detection_timer, self.close_radius_timer, self.landing_radius_timer, self.yaw_aligned_timer, self.landing_altitude_timer, self.current_mode_timer]

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
   
    def reset_timers(self):

        for timer in self.timers:
            timer.reset()

    def update_values(self, data):

        if( "gimbal_imu_pitch" in data ):
            self.gimbal_imu_pitch = data["gimbal_imu_pitch"]
        if( "gimbal_imu_roll" in data ):
            self.gimbal_imu_roll = data["gimbal_imu_roll"]
        if( "altitude" in data ):
            self.altitude = data["altitude"]
        if( "takeoff" in data ):
            if( data["takeoff"] ):
                self.set_mode("takeoff")

                print("******************* DETECTED TAKEOFF ******************")
                self.reset_timers();

        #print(f"{self.gimbal_imu_pitch=}, {self.gimbal_imu_roll=}, {self.altitude=}, {self.takeoff=}")

    def set_mode(self, mode):

        if( self.mode != mode ):

            print( f"mode: {self.mode} -> {mode}")

            self.mode_change = True
            self.mode = mode

            self.current_mode_timer.reset()

    def get_mode(self):
        return self.mode
    
    def get_mode_and_changed(self):
        result = self.mode_change
        self.mode_change = False
        return self.mode, result

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

    def update_command(self):

        yaw         = 0
        gimbal_tilt = 0
        pitch       = 0
        roll        = 0
        throttle    = 0

        if( self.detection_timer < 0.5 ):
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
                gimbal_imu_roll_rad  = self.gimbal_imu_roll  * numpy.pi / 180

                # put the orientation of the gimabl IMU into a Python quaternion
                orientation = pyquaternion.Quaternion(get_quaternion_from_euler(gimbal_imu_roll_rad, gimbal_imu_pitch_rad, 0))

                # do transform
                position_target = orientation.rotate(numpy.array([position.x, position.y, position.z]))
                
                # set commands
                yaw             = +1.0 * self.c_normalized[0]
                gimbal_tilt     = -1.0 * self.c_normalized[1]
                roll            = +1.0 * position_target[1]
                throttle        = +1.0 * position_target[0]
                pitch           = +1.0 * position_target[2]
        
        print( self.altitude )
        # update timers
        now = datetime.now()
        if( self.altitude < self.operational_altitude ):
            self.at_altitude_timer.reset()
        if( numpy.sqrt(pitch ** 2 + roll ** 2) >= 2.0 ):
            self.close_radius_timer.reset()
        if( numpy.sqrt(pitch ** 2 + roll ** 2) >= 0.2 ):
            self.landing_radius_timer.reset()
        if( self.yaw_displacement > self.max_yaw_displacement ):
            self.yaw_aligned_timer.reset()
        if( self.altitude > 1 ):
            self.landing_altitude_timer.reset()

        # determine mode
        if( "land" != self.get_mode() ):
            if( self.detection_timer < self.detection_timeout ):

#                if( self.landing_radius_timer >= 5 and self.yaw_aligned_timer >= 5 and self.landing_altitude_timer >= 2 ):
#                    self.set_mode("land")
#                elif( self.close_radius_timer >= 5 and self.yaw_aligned_timer >= 5 ):
#                    self.set_mode("descent")
#                #elif( "approach" == self.get_mode() and self.close_radius_timer >= 5 ):
#                el
                
#                if( self.landing_radius_timer >= 15): # and self.landing_altitude_timer >= 2 ):
#                    self.set_mode("land")
#                if( "descent" == self.get_mode() and self.current_mode_timer() > 5 ):
#                    self.set_mode("land")
                if( self.close_radius_timer >= 10 ):
                    if( self.altitude <= 1.5 ):
                        self.set_mode("land")
                    else:
                        self.set_mode("descent")
#                elif( self.close_radius_timer >= 5):#  and self.yaw_aligned_timer < 5 ):
#                    self.set_mode("yaw_alignment")
                elif( self.close_radius_timer >= 2 ): # and self.yaw_aligned_timer < 5 ):
                    self.set_mode("alignment")
                else:
                    self.set_mode("approach")

            else:
                if( self.detection_timer < 5 ):
                    self.set_mode("tracking")

                elif( self.detection_timer >= 5 ):
                    self.set_mode("search")
                
                self.at_altitude_timer.reset()
                self.close_radius_timer.reset()
                self.yaw_aligned_timer.reset()
                self.landing_altitude_timer.reset()

        # determine control outputs based on mode
        if( "approach" == self.get_mode() ):
            throttle = 0
            #yaw *= 0.5
            yaw = 0
            gimbal_tilt *= 0.5
            pitch       *= 0.5
            roll        *= 0.5
        elif( "alignment" == self.get_mode() ):
            throttle = 0
            yaw = 0
            gimbal_tilt *= 0.5
            pitch *= 0.5
            roll *= 0.5
        elif( "yaw_alignment" == self.get_mode() ):
            throttle = 0
            gimbal_tilt *= 0.5
            yaw = 0.25 * self.yaw_displacement
            pitch *= 0.25
            roll *= 0.25
        elif( "descent" == self.get_mode() ):
            throttle = -0.5
            gimbal_tilt *= 0.5
            yaw = 0 #0.1 * self.yaw_displacement
            pitch *= 0.5
            roll  *= 0.5
        elif( "land" == self.get_mode() ):
            roll *= 0.5
            pitch *= 0.5
            yaw = 0
            throttle = -0.5
        elif( "tracking" == self.get_mode() ):
            roll = 0
            pitch = 0
            throttle = 0
        elif( "search" == self.get_mode() ):
            roll = 0
            pitch = 0
            throttle = 0
            
            yaw = 0.15
            gimbal_tilt = 0.2
            if( self.current_mode_timer % 5 ):
                gimbal_tilt *= -1

        # constrain
        yaw         = constrain(yaw)
        gimbal_tilt = constrain(gimbal_tilt)
        roll        = constrain(roll)
        throttle    = constrain(throttle)
        pitch       = constrain(pitch)

#        roll        = apply_deadzone(  roll, deadzone=0.025 )
#        pitch       = apply_deadzone( pitch, deadzone=0.025 )

        # output
        self.command = f"@,{now},{self.get_mode()},{yaw:+0.5f},{gimbal_tilt:+0.5f},{pitch:+0.5f},{roll:+0.5f},{throttle:+0.5f},&"

    def get_command(self):

        result = None
        try:
            self.update_command()


            #if( self.detection_timer < self.detection_timeout ):
            if( any([self.detection_timer < self.detection_timeout, self.mode in ["search", "tracking"] ]) ):
                result = self.command
            else:
                print("invalid command, sending heartbeat!")

        except:
            print("No gimbal data yet - cannot generate command")

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
    control_policy = ControlPolicy(procedure="filtered")

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
        if( len(data) <= 400 ):

            try:
                data = eval(data.decode())

                control_policy.update_values( data )
            except Exception as e:
                print(f"error parsing info message from tablet: {e}")

        # if we receive an image
        else:

            # decode the image data and display it using OpenCV
            image = cv2.imdecode(numpy.frombuffer(data, numpy.uint8), cv2.IMREAD_UNCHANGED)
            publish_image(image, image_publisher, info_publisher, camera_info)

        command = control_policy.get_command_reset()
        if( command == None ):
            command = "HEARTBEAT"
#        else:
#            print(command)

        mode, mode_changed = control_policy.get_mode_and_changed()
        if( mode_changed ):
            command = f"MODE: {mode}"

        print(command)
        sock.sendto( command.encode(), client_address)

    # Don't forget to clean up when you're done!
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
