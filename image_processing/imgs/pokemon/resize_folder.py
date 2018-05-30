from PIL import Image
import os
import glob

img_list = glob.glob('full/*.jpg')

for f in img_list:
    img = Image.open(f)
    img_resize = img.resize((128,128))
    path,form = os.path.splitext(f)
    #print("resizing %s ....\n" % title)
    print("%s%s\n" % (path,form))
    img_resize.save(path + '_128' + form)
    os.remove(path + form)
