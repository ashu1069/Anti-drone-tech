# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 14:11:30 2021

@author: ashu1069
"""
"reward function"
relative_distance_factor = (current_distance/last_distance)
if distance[i]<distance[i-1]:
    reward=100-relative_distance_factor*100
elif distance[i]==distance[i-1]:
    reward=-50
else:
    reward=-100
    
#drone dynamics:angular velocity, acceleration, trajectory, time of interception
#FOV reward iterations
FOV = n               #degrees
distance = pos(D1)-pos(D2)
#input image data
if FOV = True:
    #activate thrust command
    D2.move(pos(D1))
    #find new position of D2
    intermediate_distance = pos(D1)-pos(D2)
    if intermediate_distance<distance:
        reward = 100-relative_distance-factor*100
    elif intermediate_distance=distance:
        reward = -50
    else:
        reward = -100
else:
    #repeat image processing
    
