import os
from exif import Image
from datetime import datetime
import cv2
import math
from picamzero import Camera


def using_camera():
    cam = Camera()
#    cam.capture_sequence("image1.jpg",num_image=1, interval = 1)
#    cam.capture_sequence("image2.jpg",num_image=1, interval = 1)
    cam.capture_sequence(filename = "sequence.jpg", num_images=41, interval =3)


def get_time(image):
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        time_str = img.get("datetime_original")
        time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
    return time
    
    
def get_time_difference(image_1, image_2):
    time_1 = get_time(image_1)
    time_2 = get_time(image_2)
    time_difference = time_2 - time_1
    return time_difference.seconds


def convert_to_cv(image_1, image_2):
    image_1_cv = cv2.imread(image_1, 0)
    image_2_cv = cv2.imread(image_2, 0)
    return image_1_cv, image_2_cv


def calculate_features(image_1, image_2, feature_number):
    orb = cv2.ORB_create(nfeatures = feature_number)
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1_cv, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2_cv, None)
    return keypoints_1, keypoints_2, descriptors_1, descriptors_2


def calculate_matches(descriptors_1, descriptors_2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches
    

def display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches):
    match_img = cv2.drawMatches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches[:100], None)
    resize = cv2.resize(match_img, (1600,600), interpolation = cv2.INTER_AREA)
    cv2.imshow('matches', resize)
    cv2.waitKey(0)
    cv2.destroyWindow('matches')
    
    
def find_matching_coordinates(keypoints_1, keypoints2, matches):
    coordinates_1 = []
    coordinates_2 = []
    for match in matches:
        image_1_idx = match.queryIdx
        image_2_idx = match.trainIdx
        (x1,y1) = keypoints_1[image_1_idx].pt
        (x2,y2) = keypoints_2[image_2_idx].pt
        coordinates_1.append((x1,y1))
        coordinates_2.append((x2,y2))
    return coordinates_1, coordinates_2


def calculate_mean_distance(coordinates_1, coordinates_2):
    all_distances = 0
    merged_coordinates = list(zip(coordinates_1, coordinates_2))
    for coordinate in merged_coordinates:
        x_difference = coordinate[0][0] - coordinate[1][0]
        y_difference = coordinate[0][1] - coordinate[1][1]
        distance = math.hypot(x_difference, y_difference)
        all_distances = all_distances + distance
    return all_distances / len(merged_coordinates)


def calculate_speed_in_kmps(feature_distance, GSD, time_difference): #including the curavture of the earth
    EARTHRADIUS = 6371
    PI = math.pi
    distance = feature_distance * GSD / 100000
    print(distance)
    Phi = math.asin(distance/EARTHRADIUS)
    distance  = 2*EARTHRADIUS*PI*Phi*(1/(2*PI))
    speed = distance / time_difference
    return speed

def checkIfSpeedIsSensible(speed):
    if speed >6 and speed < 9:
        listOfSpeed.append(speed)
    else:
        print(speed, "is not valid")
        

def calculateMeanSpeed(listOfSpeed):
    sumOfSpeed = sum(listOfSpeed)
    averageSpeed = sumOfSpeed/len(listOfSpeed)
    return averageSpeed

def exportResultAsFile(speed):
    resultFile = open("result.txt","x")
    resultFile.write(str(speed)+"km/s")
    return resultFile
    
listOfImages = [
    #list of images in the data list
    ["sequence-0.jpg","sequence-1.jpg"],
    ["sequence-2.jpg", "sequence-3.jpg"],
    ["sequence-4jpg", "sequence-5.jpg"],
    ["sequence-6.jpg", "sequence-7.jpg"],
    ["sequence-8.jpg", "sequence-9.jpg"],
    ["sequence-10.jpg", "sequence-11.jpg"],
    ["sequence-12.jpg", "sequence-13.jpg"],
    ["sequence-14.jpg", "sequence-15.jpg"],
    ["sequence-16.jpg", "sequence-17.jpg"],
    ["sequence-18.jpg", "sequence-19.jpg"],
    ["sequence-20.jpg", "sequence-21.jpg"],
    ["sequence-22.jpg", "sequence-23.jpg"],
    ["sequence-24.jpg", "sequence-25.jpg"],
    ["sequence-26.jpg", "sequence-27.jpg"],
    ["sequence-28.jpg", "sequence-29.jpg"],
    ["sequence-30.jpg", "sequence-31.jpg"],
    ["sequence-32.jpg", "sequence-33.jpg"],
    ["sequence-34.jpg", "sequence-35.jpg"],
    ["sequence-36.jpg", "sequence-37.jpg"],
    ["sequence-38.jpg", "sequence-39.jpg"],
    ["sequence-40.jpg", "sequence-41.jpg"],
    ]


numberOfImagePair = len(listOfImages)
#print(numberOfImagePair)

listOfSpeed = []

using_camera() #take the pictures now

for i in range(numberOfImagePair):
    
    time_difference = get_time_difference(listOfImages[i][0], listOfImages[i][1]) #get time difference between images
    image_1_cv, image_2_cv = convert_to_cv(listOfImages[i][0], listOfImages[i][1]) #create opencfv images objects
    keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 9999) #get keypoints and descriptors
    matches = calculate_matches(descriptors_1, descriptors_2) #match descriptors
    coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)
    average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)
    speed = calculate_speed_in_kmps(average_feature_distance, 12648, time_difference)
    print(speed)
    checkIfSpeedIsSensible(speed)
#   display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches) #display matches

print("average speed is ", calculateMeanSpeed(listOfSpeed))
exportResultAsFile(calculateMeanSpeed(listOfSpeed))
