##################Importing Files####################

#Libraries for Time and GPIO's of Raspberry Pi
import time
from time import sleep
import RPi.GPIO as GPIO
#Libraries for Web Server
import http.server
from http.server import BaseHTTPRequestHandler, HTTPServer
#Libraries for PiCamera
from picamera.array import PiRGBArray
from picamera import PiCamera
#Computer Vision Library
import cv2
#Computation
import numpy as np


###### PIN Config ##########

#Assigning Pins- Rover
m11=29
m12=31
m21=33
m22=35

#Assigning Pins- ARM
s_base=7
s_shoulder=12
s_elbow=13
s_gripper=15

#Assigning Pins- Ultra Sonic Sensor
TRIG = 16
ECHO = 18

#Count for alignment
global FPS
FPS = 1

########## BCM and GPIO Config ##########
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

#Rover Setup
GPIO.setup(m11, GPIO.OUT)
GPIO.setup(m12, GPIO.OUT)
GPIO.setup(m21, GPIO.OUT)
GPIO.setup(m22, GPIO.OUT)

#Ultra Sonic Sensor Setup
GPIO.setup(TRIG,GPIO.OUT)
GPIO.setup(ECHO,GPIO.IN)

#All Off- Rover
GPIO.output(m11, False)
GPIO.output(m12, False)
GPIO.output(m21, False)
GPIO.output(m22, False)

#All Off- ARM
#GPIO.output(s_base, False)
#GPIO.output(s_shoulder, False)
#GPIO.output(s_elbow, False)
#GPIO.output(s_gripper, False)

#Ultra Sonic Sensor Trigger pin Off
GPIO.output(TRIG, False)

#WebServer Variable
Request = None

#Time to Settle and Rest after Initialization
time.sleep(2)

########## Functions #############

#Function for Forward Movement of Rover
def forward():
   print("FORWARD")
   GPIO.output(m11 , True)
   GPIO.output(m12 , False)
   GPIO.output(m21 , False)
   GPIO.output(m22 , True)

#Function for BACKWARD Movement of Rover
def reverse():
   print("BACKWARD")
   GPIO.output(m11 , False)
   GPIO.output(m12 , True)
   GPIO.output(m21 , True)
   GPIO.output(m22 , False)

#Function for RIGHT(Clockwise) Movement of Rover
def right():
   print("RIGHT")
   GPIO.output(m11 , True)
   GPIO.output(m12 , False)
   GPIO.output(m21 , True)
   GPIO.output(m22 , False)

#Function for LEFT(Anti-Clockwise) Movement of Rover
def left():
   print("LEFT")
   GPIO.output(m11 , False)
   GPIO.output(m12 , True)
   GPIO.output(m21 , False)
   GPIO.output(m22 , True)

#Function for STOP the Movement of Rover
def stop():
   print("STOP")
   GPIO.output(m11 , False)
   GPIO.output(m12 , False)
   GPIO.output(m21 , False)
   GPIO.output(m22 , False)

#Function to measure Distance using Ultra Sonic Sensor
def ultra():

           global distance

           GPIO.output(TRIG, True)
           time.sleep(0.00001)
           GPIO.output(TRIG, False)

           while GPIO.input(ECHO)==0:
              pulse_start = time.time()

           while GPIO.input(ECHO)==1:
              pulse_end = time.time()

           pulse_duration = pulse_end - pulse_start
           distance = pulse_duration * 17150
           distance = round(distance, 2)

           return

#Function for Object Detection using Template Matching Technique
def objdetect():

    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))

    #Getting the Template
    if Request == 'black%20king':
       #Reading Template
       template = cv2.imread('template.jpg', 0)
       print('Black King Template Uploaded!!')
    else:
       #Reading Template
       template = cv2.imread('template2.jpg', 0)
       print('White King Template Uploaded!!')

    # allow the camera to warmup
    time.sleep(0.1)

    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    FPS=1

    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # grab the raw NumPy array representing the image
        image = frame.array
        # Saving the copy of Frame
        img_rgb = image
        # Convert to Gray
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        
        # Shape of Template
        w, h = template.shape[::-1]
        
        #Template Matching
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        
        #Cordinates
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)
        # Starting and Ending Coordinates
        (startX, startY) = (int(maxLoc[0] ), int(maxLoc[1] ))
        (endX, endY) = (int((maxLoc[0] + w) ), int((maxLoc[1] + h) ))

        # Draw framerate in corner of frame
        cv2.putText(img_rgb,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

        # Draw Rectangle around Detected Object
        cv2.rectangle(img_rgb, (startX, startY), (endX, endY), (0, 0, 255), 2)

        # Show the frame
        cv2.imshow("Frame", img_rgb)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        key = cv2.waitKey(1) & 0xFF


        #CODE FOR ALIGNING
        if FPS > 10:
           #Center of rectangle
           xcord = (startX+endX)/2
           ycord = (startY+endY)/2
	
           #Alignment
           print("Aligning the Rover")
           if xcord>330:
              right()
              print("Moving Right")
           if xcord<310:
              left()
              print("Moving Left")

           time.sleep(0.1)
           stop()
        time.sleep(1)
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        if FPS>15:
           # Exit when Centerized
           if xcord>310 and xcord<330:
               camera.close()
               print("Centerized")
               break

        FPS = FPS+1
        print ("FPS: ",FPS)
        
    cv2.destroyAllWindows()
    time.sleep(1)

    return


#Function for Picking Up the Object using Robotic ARM
def pickup():
    #Code Please

    # Set pins 11 & 12 as outputs, and define as PWM servo1 & servo2
    GPIO.setup(12,GPIO.OUT)
    servo1 = GPIO.PWM(12,50) # pin 11 for servo1
    GPIO.setup(15,GPIO.OUT)
    servo2 = GPIO.PWM(15,50) # pin 12 for servo2

    # Start PWM running on both servos, value of 0 (pulse off)
    servo2.start(0)
    # Turn servo1 to 90
    servo2.ChangeDutyCycle(7)
    time.sleep(0.5)
    servo2.ChangeDutyCycle(0)
    servo2.stop()
    # Wait for 2 seconds
    time.sleep(2)

    # Start PWM running on both servos, value of 0 (pulse off)
    servo1.start(0)
    # Turn servo1 to 90
    servo1.ChangeDutyCycle(5.5)
    time.sleep(0.5)
    servo1.ChangeDutyCycle(0)
    servo1.stop()
    # Wait for 2 seconds
    time.sleep(2)

    # Start PWM running on both servos, value of 0 (pulse off)
    servo2.start(0)
    # Turn servo1 to 90
    servo2.ChangeDutyCycle(1)
    time.sleep(1)
    #servo2.ChangeDutyCycle(0)
    #servo2.stop()
    # Wait for 2 seconds
    #time.sleep(2)

    # Start PWM running on both servos, value of 0 (pulse off)
    servo1.start(0)
    # Turn servo1 to 90
    servo1.ChangeDutyCycle(1)
    time.sleep(0.5)
    servo1.ChangeDutyCycle(0)
    servo1.stop()
    # Wait for 2 seconds
    time.sleep(2)

    #Clean things up at the end
    servo1.stop()
    servo2.stop()

    print ("The Object has been Picked")
    print ("Thankyou Staff for your time!!")
    return

########## Custom Navigation Algorithm: Specifies Minimum Distance between two points ####################
def custom_nav_algorithm(m,startp,endp):
    w,h = 10,10		# 10x10(blocks) is the dimension of the input images
    sx,sy = startp 	#Start Point
    ex,ey = endp 	#End Point
    #[parent node, x, y, g, f]
    node = [None,sx,sy,0,abs(ex-sx)+abs(ey-sy)] 
    closeList = [node]
    createdList = {}
    createdList[sy*w+sx] = node
    k=0
    while(closeList):
        node = closeList.pop(0)
        x = node[1]
        y = node[2]
        l = node[3]+1
        k+=1
        #find neighbours 
        if k!=0:
            neighbours = ((x,y+1),(x,y-1),(x+1,y),(x-1,y))
        else:
            neighbours = ((x+1,y),(x-1,y),(x,y+1),(x,y-1))
        for nx,ny in neighbours:
            if nx==ex and ny==ey:
                path = [(ex,ey)]
                while node:
                    path.append((node[1],node[2]))
                    node = node[0]
                return list(reversed(path))            
            if 0<=nx<w and 0<=ny<h and m[ny][nx]==0:
                if ny*w+nx not in createdList:
                    nn = (node,nx,ny,l,l+abs(nx-ex)+abs(ny-ey))
                    createdList[ny*w+nx] = nn
                    #adding to closelist ,using binary heap
                    nni = len(closeList)
                    closeList.append(nn)
                    while nni:
                        i = (nni-1)>>1
                        if closeList[i][4]>nn[4]:
                            closeList[i],closeList[nni] = nn,closeList[i]
                            nni = i
                        else:
                            break
    return []


############# ROVER ACTION FROM PLANNED PATH ###############
def roveraction(result):
    x=0;y=0
    rflag=0; lflag=0; dflag=0; start=0
    while True:
        if result[x]==result[len(result)-2]:####LAST BOX CHECK
            if result[x][y]>result[x+1][y]:###LEFT
                if dflag==1:
                    print("RIGHT")
                    right()
                    time.sleep(1.4)
                    stop()
                    rflag=1
                elif lflag != 1:
                    print("LEFT")
                    left()
                    time.sleep(1.4)
                    stop()
                    lflag=1
            if result[x][y]<result[x+1][y]:###RIGHT
                if dflag==1:
                    print("LEFT")
                    left()
                    time.sleep(1.4)
                    stop()
                    lflag=1
                elif rflag !=1:
                    print("RIGHT")
                    right()
                    time.sleep(1.4)
                    stop()
                    rflag=1
            break
        else:
            if result[x][y]>result[x+1][y]:###LEFT
                if lflag==1:
                    print("FRWD")
                    forward()
                    time.sleep(1.1)
                    stop()
                elif dflag==1:
                    print("RIGHT->FRWD")
                    right()
                    time.sleep(1.4)
                    stop()
                    forward()
                    time.sleep(1.1)
                    stop()
                    lflag=1
                else:
                    print("LEFT->FRWD")
                    left()
                    time.sleep(1.4)
                    stop()
                    forward()
                    time.sleep(1.1)
                    stop()
                    lflag=1
                rflag=0
            if result[x][y]<result[x+1][y]:###RIGHT
                if rflag==1:
                    print("FRWD")
                    forward()
                    time.sleep(1.1)
                    stop()
                elif dflag==1:
                    print("LEFT->FRWD")
                    left()
                    time.sleep(1.4)
                    stop()
                    forward()
                    time.sleep(1.1)
                    stop()
                    rflag=1
                else:
                    print("RIGHT->FRWD")
                    right()
                    time.sleep(1.4)
                    stop()
                    forward()
                    time.sleep(1.1)
                    stop()
                    rflag=1
                lflag=0
            if result[x][y+1]>result[x+1][y+1]:###FORWARD
                if rflag==1:
                    print("LEFT->FRWD")
                    left()
                    time.sleep(1.4)
                    stop()
                    forward()
                    time.sleep(1.1)
                    stop()
                    rflag=0
                elif lflag==1:
                    print("RIGHT->FRWD")
                    right()
                    time.sleep(1.4)
                    stop()
                    forward()
                    time.sleep(1.1)
                    stop()
                    lflag=0
                else:
                    print("FRWD")
                    forward()
                    time.sleep(1.1)
                    stop()
            if result[x][y+1]<result[x+1][y+1]:###REVERSE
                if start==0:
                    print("LEFT->LEFT->FRWD")
                    left()
                    time.sleep(1.4)
                    stop()
                    left()
                    time.sleep(1.4)
                    stop()
                    forward()
                    time.sleep(1.1)
                    stop()
                    start=1
                    dflag=1
                if rflag==1:
                    print("RIGHT->FRWD")
                    right()
                    time.sleep(1.4)
                    stop()
                    forward()
                    time.sleep(1.1)
                    stop()
                    rflag=0
                    dflag=1
                elif lflag==1:
                    print("LEFT->FRWD")
                    left()
                    time.sleep(1.4)
                    stop()
                    forward()
                    time.sleep(1.1)
                    stop()
                    lflag=0
                    dflag=1
                elif dflag==1:
                    print("FRWD")
                    forward()
                    time.sleep(1.1)
                    stop()
            x=x+1

############# Traversing each Co-ordinate Window #################

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

###########Function to match RGB to color name #############

def ColorNameFromRGB(R,G,B):
    # Calculate HSV from R,G,B - something like this
    # Make a single pixel from the parameters 
    onepx=np.reshape(np.array([R,G,B],dtype=np.uint8),(1,1,3))
    # Convert it to HSV
    onepxHSV=cv2.cvtColor(onepx,cv2.COLOR_RGB2HSV)

    H=onepxHSV[0][0][0]
    S=onepxHSV[0][0][1]
    V=onepxHSV[0][0][2]
    #print("HSV: ",H,S,V)

    if S<25:
        if V<85:
           return "Black"
        elif V<170:
           return "Grey"
        return "White"
    else:
        if (H>=175 and H<=180) or (H>=0 and H<= 10):#0-10
            return "Red"
        if H>=14 and H<= 25:#20-30
            return "Orange"
        if H>=26 and H<= 35:#26
            return "Yellow"
        if H>=36 and H<= 65:#40-60
            return "Green"
        if H>=106 and H<= 122:#105-115
            return "Blue"
        if H>=132 and H<= 142:#158 ###PINK and Purple, Not properly calibrated
            return "Purple"
        if H>=145 and H<= 155:
            return "Pink"

def getpath(image_filename):

	occupied_grids = []		 
	planned_path = {}		  	
	
        image = image_filename
	(winW, winH) = (60, 60) 

	obstacles = []
        startpoint = []
        endpoint = [] 
	index = [1,1] #starting point
	#create blank image, initialized as a matrix of 0s the width and height
	blank_image = np.zeros((60,60,3), np.uint8)
	#create an array of 100 blank images
	list_images = [[blank_image for i in xrange(10)] for i in xrange(10)] 	#array of list of images 
	#empty #matrix to represent the grids of individual cropped images
	maze = [[0 for i in xrange(10)] for i in xrange(10)] 			

	#traversal for each square
	for (x, y, window) in sliding_window(image, stepSize=60, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			continue

	#	print index, image is our iterator, it's where were at returns image matrix
		clone = image.copy()
		#format square for openCV
		cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
		crop_img = image[x:x + winW, y:y + winH] 				#crop the image
		list_images[index[0]-1][index[1]-1] = crop_img.copy()			#Add it to the array of images

                ######## GET Dominent RGB Values ###############
                data = np.reshape(crop_img, (-1,3))
                print(data.shape)
                data = np.float32(data)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                flags = cv2.KMEANS_RANDOM_CENTERS
                compactness,labels,centers = cv2.kmeans(data,1,None,criteria,10,flags)

                #print('Dominant color is: bgr({})'.format(centers[0].astype(np.int32)))

                B=centers[0][0].astype(np.int32)
                G=centers[0][1].astype(np.int32)
                R=centers[0][2].astype(np.int32)

                name=ColorNameFromRGB(R,G,B)
                #print("The Color is: ",name)
                
                if name in ('Green','Red','Blue'):
                    maze[index[1]-1][index[0]-1] = 1
                    occupied_grids.append(tuple(index))

                print("The Color is: ",name)
                if name=='Red':
                    maze[index[1]-1][index[0]-1] = 1
                    startpoint.append(tuple(index))

                print("The Color is: ",name)
                if name=='Green':
                    maze[index[1]-1][index[0]-1] = 1
                    endpoint.append(tuple(index))

                if name=='Black':
                    maze[index[1]-1][index[0]-1] = 0
                    obstacles.append(tuple(index))

		#show this iteration
		cv2.imshow("Window", clone)
		cv2.waitKey(1)
		time.sleep(0.025)
	
		#Iterate
		index[1] = index[1] + 1							
		if(index[1]>10):
			index[0] = index[0] + 1
			index[1] = 1


        result = custom_nav_algorithm(maze,(startpoint[0]),(endpoint[0]))
        #print("Result: ",result)
        planned_path = result

	return occupied_grids, planned_path, obstacles

cv2.waitKey(0)
cv2.destroyAllWindows()

#Function for Autonomous Detection and Picking of Object
def automatic():

    #Webserver Reception for Voice Command via APP
    class RequestHandler_httpd(BaseHTTPRequestHandler):
      def do_GET(self):
        global Request
        messagetosend = bytes('Project Rover',"utf")
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Content-Length', len(messagetosend))
        self.end_headers()
        self.wfile.write(messagetosend)
        Request = self.requestline
        Request = Request[5 : int(len(Request)-9)]
        print(Request)
		
		
        #Checking if the Obtined Voice Command is Valid
        if Request == 'black%20king' or Request == 'white%20king':
           print("Command Accepted")
           time.sleep(1)
           print("Searching for the Requested Object.....")
           #time.sleep(0.5)
           ultra()
           print("distance:",distance)
           #Moving the Rover to distance where Object Recognition can be Done
           if distance>130:
              while distance>130:
                 forward()
                 ultra()
           else:
              print ("distance:",distance,"cm")

           stop()
           #Aligned Distance for Recognition
           print ("Object is at a distance of :",distance,"cm")
			
	   #Code for Object Recognition and Rover Alignment
           print ("Initiating Object Recognition")
           time.sleep(0.5)
	   #Object Detection
           objdetect()
           time.sleep(0.5)
           print ("Object has been Recognised and Rover has been Aligned towards the Object")
		   
	   #Moving towards the Object
           if distance>27:
              while distance>27:
                 forward()
                 #stop()
                 ultra()
           else:
              print ("distance:",distance,"cm")

           stop()
	   #Object Detection
           objdetect()
           time.sleep(0.5)
           print ("Object has been Recognised and Rover has been Aligned towards the Object")
		   
	   #Moving towards the Object
           if distance>13:
              while distance>13:
                 forward()
                 #stop()
                 ultra()
           else:
              print ("distance:",distance,"cm")

           stop()	  
           #Printing the Distance of Object	  
           if Request== 'black%20king':
              print ("The Chess Piece- Black King is at a Distance of: ",distance,"cm")
           else:
              print ("The Chess Piece- White King is at a Distance of: ",distance,"cm")
		   
           #ARM Picking Up Action
           time.sleep(1)
           pickup()

           #Exiting the Checking Loop
           #break



    server_address_httpd = ('192.168.43.219',8081)
    httpd = HTTPServer(server_address_httpd, RequestHandler_httpd)
    print('Starting Server')
    httpd.serve_forever()



    return


########### MAIN CODE #################

#load the image
img = cv2.imread('test_images/path0.jpg', cv2.IMREAD_UNCHANGED)
#print('Original Dimensions : ',img.shape)

#set the dimensions
dim = (600, 600)
#resize the image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
#print('Resized Dimensions : ',resized.shape)

#saving image
#status = cv2.imwrite('test_images/path101.jpg',resized)

#obtain the values
occupied_grids, planned_path, obstacles = getpath(resized)
print "Occupied Grids : "
print occupied_grids
print "Planned Path :"
print planned_path
print "Obstacles :"
print obstacles

#move the rover to desired point
roveraction(planned_path)

print("Autonomous Control over Rover is Enabled")
#automatic()

GPIO.cleanup()
