######################################
#           Color Averager           #
#                                    #
#             UTeach CSP             #
#                                    #
######################################

# importing PIL.Image library and os library
from PIL import Image 
import os

# Deletes the image file if it exists
if os.path.exists("/Users/erikborn/Documents/Python/jupyter/Codio/average.jpg"):
	os.remove("/Users/erikborn/Documents/Python/jupyter/Codio//average.jpg")

# Creates two blank lines before our output
print("\n\n")

# Replaces the current pixels of Image
# with the colors sent, from left to right
# creating 256 unique color bands
def change(r1, g1, b1, r2, g2, b2):
	for i in range(len(new_pixels)):
    	# Sets the first third of image to color 1
		if i % 300 < 100:
			rgb = (r1, g1, b1)
			new_pixels[i] = rgb
    
		# Sets the middle third of image to average of colors
		if i % 300 >= 100 and i % 300 <= 200:
			ra = (r1 + r2) // 2
			ga = (g1 + g2) // 2
			ba = (b1 + b2) // 2
			rgb = (ra, ga, ba)
			new_pixels[i] = rgb

		# Sets the last third of image to color 2
		if i % 300 > 200:
			rgb = (r2, g2, b2)
			new_pixels[i] = rgb

# Creates the original image and sets all of the pixels to 
# a new array called new_pixels
image = Image.new("RGB", (300,200))
pixels = image.getdata()
new_pixels = []
for p in pixels:
	new_pixels.append(p)

# sets red, green and blue decimal values for the first color
r1 = 255
g1 = 0
b1 = 0

# sets red, green and blue decimal values for the second color
r2 = 0
g2 = 0
b2 = 255

# Sends the values for the two user input colors
change(r1, g1, b1, r2, g2, b2)

# Creates and saves the new image
newImage = Image.new("RGB", image.size)
newImage.putdata(new_pixels)
newImage.save("/Users/erikborn/Documents/Python/jupyter/Codio/average.jpg")

