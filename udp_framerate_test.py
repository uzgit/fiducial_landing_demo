#!/usr/bin/python3

import socket
import cv2
import numpy
import datetime

import os
os.environ["DISPLAY"] = ":0"

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind the socket to the address and port
server_address = ('192.168.1.100', 14555)
print('starting up on {} port {}'.format(*server_address))
sock.bind(server_address)
print("bound!")

i = 0

times = []

while True:

    before = datetime.datetime.now()
    
    # Receive data from the server
    # print("Waiting for data...")
    data, client_address = sock.recvfrom(1000000)
    # print(f"Received {len(data)} bytes")

    i += 1
    #print(i)
    response = f"ACK {i}"
    sock.sendto( response.encode(), client_address)
    
    # Decode the image data and display it using OpenCV
    image = cv2.imdecode(numpy.frombuffer(data, numpy.uint8), cv2.IMREAD_UNCHANGED)

    after = datetime.datetime.now()

    times.append( (after - before).total_seconds() )
    print( f"{1/numpy.mean(times) :0.1f} fps" )
    if( len(times) >= 50 ):
        times.pop(0)

    cv2.imshow('image', image)
    cv2.waitKey(1)

# Don't forget to clean up when you're done!
cv2.destroyAllWindows()
