import random

""" All of the functions below are helper functions that will not be considered in the running time compelexity since we have a high tech box and balls already"""

def subarray_glows(n,left,right):
    for i in range(left,right+1):
        if n[i]==radioactive:
            return True
    return False

def set_random_radioactive_ball(n):
    return random.randint(0,n-1)

def generate_n_element_array(n):
    return [i for i in range(n)]

"""Helper functions ended. Below is the actual function of the algorithm"""

def find_radioactive_ball(n,left,right):
    if left==right:
        return left
    mid=(left+right)//2
    if subarray_glows(n,left,mid): 
        return find_radioactive_ball(n,left,mid)
    else:
        return find_radioactive_ball(n,mid+1,right)

"""Testing Below"""

number=int(input("Enter the number of balls:"))
radioactive=set_random_radioactive_ball(number)
balls=generate_n_element_array(number)
 
print(f"We have set the radioactive ball position/index as {radioactive}") #printing initally to check later if our algorithm is working correctly
print()
test=find_radioactive_ball(balls,0,number)
print(f"The radioactive ball is at position {test}") #printing the result of the algorithm
print()
if test==radioactive:
    print("The algorithm has found the radioactive ball correctly")
else:
    print("The algorithm has failed to find the radioactive ball correctly")



    





    