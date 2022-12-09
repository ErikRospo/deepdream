import cv2,os

paths=["frames/%04d.jpg"%i for i in range(0,100)]
imgArray = []
print("Starting reading")
s=[970,1250]
for filename in paths:
    if os.path.exists(filename):
        imgArray.append(cv2.resize(cv2.imread(filename),s))
print(s)
print("done reading")
print("writing video")
out = cv2.VideoWriter("res2.mp4",cv2.VideoWriter_fourcc(*'mp4v'), 15, s)
for i in range(len(imgArray)):
    out.write(imgArray[i])
out.release()
print("Done making video")