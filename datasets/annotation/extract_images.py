import os

videos = []
paths = []
for path, dirs, files in os.walk('./'):
    for f in files:
        if f.endswith('.avi'):
            videos.append(os.path.join(path, f))
            paths.append(path)

for i in range (0,len(videos)):
    filename = videos[i]
    path = paths[i]
    output = os.path.join(path,'image-%4d.png')
    print(output)
    os.system("ffmpeg -i {0} -f image2 -vf fps=25 {1}".format(filename,output))
