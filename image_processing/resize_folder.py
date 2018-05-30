from PIL import Image
import os
import glob

img_list = glob.glob('full/*.jpg')

for f in img_list:
    img = Image.open(f)
    img_resize = img.resize((128,128))
    path,title = os.path.splitext(f)
    print("resizing %s ....¥n" % title)
    img_resize.save(path + '/resized_128/' + title)
