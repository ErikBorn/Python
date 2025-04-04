import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('julia1.png')#Swap with your image name

imgplot = plt.imshow(img)

# plt.annotate("Hi", (500,-500), size="large", color="blue",zorder=100)
plt.text(500,500,"HELLO",size = 20,color='blue')
plt.show()
