# imports and basic notebook setup
from io import StringIO
import os
import numpy as np
import scipy.ndimage as nd
import PIL.Image
import json
import caffe
import time
import datetime

def format_time(seconds):
    """Converts a duration in seconds to a string in the format 'HHh MMm SSs'."""
    delta = datetime.timedelta(seconds=int(seconds))
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)

def format_time_diff(seconds):
    """Converts a duration in seconds to a string in the format 'MM-DD-YY HH:mm:SS'."""
    end_time = datetime.datetime.now() + datetime.timedelta(seconds=seconds)
    return end_time.strftime("%m-%d-%y %H:%M:%S")
with open("settings.json","rt") as f:
    settings=json.load(f)
# If your GPU supports CUDA and Caffe was built with CUDA support,
# uncomment the following to run Caffe operations on the GPU.
if "cuda" in settings and settings["cuda"]:
    caffe.set_mode_gpu()
    # caffe.set_device(0) # select GPU device if multiple devices exist
model_path = './models/' # substitute your path here
net_fn   = model_path + 'deploy.prototxt'
param_fn = model_path + 'bvlc_googlenet.caffemodel'
useRotate="rotate" in settings["transform"]
useAffine="affine" in settings["transform"]
# setting up progress bar.
status={
    "topLevel":"",
    "octave":""
}
elapsed=time.time()
def get_current_time_seconds():
    global elapsed
    a=time.time()-elapsed
    return a
expectedTimes=[]
# Patching model to be able to compute gradients.
# Note that you can also manually add "force_backward: true" line to "deploy.prototxt".


net = caffe.Classifier(net_fn, param_fn,
                       mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])
def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
def objective_L2(dst):
    dst.diff[:] = dst.data 

def make_step(net, step_size=1.5, end='inception_4c/output', 
              jitter=32, clip=True, objective=objective_L2):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]
    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
            
    net.forward(end=end)
    objective(dst)  # specify the optimization objective
    
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
            
    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)

def fancy_status():
    octave=status["octave"]
    outerLevel=status["topLevel"]
    if octave==(-1,-1):
        octaveStr="(?/?)"
    else:
        octaveStr="({}/{})".format(*octave)
    if outerLevel==(-1,-1):
        outerLevelStr="(?/?)"
    else:
        outerLevelStr="({}/{})".format(*outerLevel)
    
    # totalIndex=int(settings["dream"]["iterations"])*int(settings['dream']['octaves'])*int(settings['dream']['iterationsPer'])
    # #TODO: find actual value given parameters
    # currentIndex= outerLevel[1]*octave[1]+octave[1]+1
    # print("Processing file {} ({}%)".format(currentIndex, 100*(totalIndex//currentIndex)), end="")
    # print()
    # curr_time=get_current_time_seconds()
    # percentage=currentIndex/totalIndex
    # left=1-percentage
    # total_time=(curr_time/currentIndex)*totalIndex
    # eltime=format_time(total_time*percentage) # time elapsed, in a friendly way, e.g. 00h 22m 15s
    # remtime=format_time(total_time*left) # time remaining, again in a friendly way.
    # tottime=format_time(total_time*percentage+total_time*left) # total time.
    # fttime=format_time_diff(total_time*percentage+total_time*left) # finish time: MM-DD-YY HH:mm:SS 
    totalIndex=int(settings["dream"]["iterations"])*int(settings['dream']['octaves'])
    currentIndex= outerLevel[0]*octave[1]+octave[0]

    curr_time=get_current_time_seconds()
    percentage=currentIndex/totalIndex
    left=1-percentage
    total_time=(curr_time/currentIndex)*totalIndex
    eltime=format_time(total_time*percentage) # time elapsed, in a friendly way, e.g. 00h 22m 15s
    remtime=format_time(total_time*left) # time remaining, again in a friendly way.
    tottime=format_time(total_time) # total time.
    fttime=format_time_diff(curr_time+total_time*left) # finish time: MM-DD-YY HH:mm:SS 
    expectedTimes.append((total_time,eltime,remtime,tottime,fttime))
    if not ("quiet" in settings and settings["quiet"]):
        print(f"Outer Level: {outerLevelStr}, Octave: {octaveStr}, {percentage*100:5.2f}%, elapsed time {eltime}, remaining time {remtime}, total time {tottime}, estimated finish time {fttime}")
#deepdream will have been called int(settings["dream"]["iterations"]) times.
def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, 
              end='inception_4c/output', clip=True, **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in range(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
    
    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        #this will be called int(settings["dream"]["iterations"])*octave_n
        status["octave"]=(octave+1,octave_n)
        fancy_status()
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail
        for i in range(iter_n):
            #this will be called int(settings["dream"]["iterations"])*octave_n*iter_n times
            make_step(net, end=end, clip=clip, **step_params)

        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])

os.system('clear')    
img = np.float32(PIL.Image.open(settings['dream']['inputPath']).resize((224,224)))
os.makedirs(os.path.split(settings["dream"]["output"]["images"]["path"])[0],exist_ok=True)
frame = img
frame_i = 0
h, w = frame.shape[:2]
s = 0.05 # scale coefficient
paths=[]

##at the end, i will be equal to int(settings["dream"]["iterations"])

for i in range(int(settings['dream']['iterations'])):
    
    status["topLevel"]=(i+1,int(settings["dream"]["iterations"]))
    # "({}/{})".format(i+1,settings['dream']['iterations'])
    frame = deepdream(net, frame,iter_n=int(settings['dream']['iterationsPer']),octave_n=int(settings['dream']['octaves']),octave_scale=float(settings['dream']['octaveScale']))
    fp=settings["dream"]["output"]["images"]["path"]%frame_i
    PIL.Image.fromarray(np.uint8(frame)).save(fp)
    paths.append(fp)
    if useAffine:
        frame = nd.affine_transform(frame,
                                [float(settings["transform"]["affine"]["x_factor"]),
                                    float(settings["transform"]["affine"]["y_factor"]),1],
                            [h*s*float(settings["transform"]["affine"]["y_center"]),
                                w*s*float(settings["transform"]["affine"]["x_center"]),0], order=1)
    if useRotate:
        frame=nd.rotate(frame,float(settings["transform"]["rotate"]["angle"]))
    frame_i += 1
    

# paths=["frames/%04d.jpg"%i for i in range(0,100)]
if "video" in settings["dream"]["output"]:
    imgArray = []
    import cv2

    print("Starting reading")
    s=[970,1250]
    for filename in paths:
        if os.path.exists(filename):
            imgArray.append(cv2.resize(cv2.imread(filename),s))
    print(s)
    print("done reading")
    print("writing video")
    videoSettings = settings['dream']["output"]["video"]
    out = cv2.VideoWriter(videoSettings['path'],cv2.VideoWriter_fourcc(*videoSettings['codec']),float(videoSettings['fps']), s)
    for i in range(len(imgArray)):
        out.write(imgArray[i])
    out.release()
    print("Done making video")
if "dataLogging" in settings and settings["dataLogging"]["log"]:
    import csv
    with open(settings["dataLogging"]["path"],"wt") as f:
        a=csv.writer(f)
        #expectedTimes.append((total_time,eltime,remtime,tottime,fttime))
        
        a.writerow(["Total Time","Elapsed Time","Remaining time","Total Time","Finish Time"])
        a.writerows(expectedTimes)
        