from sketchpy import canvas

from sklearn.preprocessing import scale

pen = canvas.sketch_from_svg('C:\\Users\\saran\\Downloads\\vijay.svg',scale=70)


pen.draw()