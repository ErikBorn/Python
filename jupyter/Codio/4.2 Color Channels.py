######################################
#           Black and White          #
#                                    #
#             UTeach CSP             #
#                                    #
######################################

# importing PIL.Image library and os library
from PIL import Image 
import os

# Deletes the image file if it exists
if os.path.exists("/Users/erikborn/Documents/Python/jupyter/Codio/shade.jpg"):
 	os.remove("/Users/erikborn/Documents/Python/jupyter/Codio/shade.jpg")

# Adds two blank lines before our output
print("\n\n")

# Replaces the current pixels of Image
# with the colors sent, from left to right
# creating "i" unique color bands
def change(red, green, blue, i):
	for j in range(len(new_pixels)):
		if j % image.width == image.width - (256 - i):
			rgb = (red, green, blue)
			new_pixels[j] = rgb

# Controls A LOT of the program - What does "i" do?
i = 0

# Creates the original image and sets all of the pixels to a new array
# called new_pixels
image = Image.new("RGB", (256 - i,100))
pixels = image.getdata()
new_pixels = []
for p in pixels:
	new_pixels.append(p)

# Sets and sends the values for each color band 
# this is where changes should be made to the code
while i < 256:
	r = i
	g = i//3
	b = i//2
  	# Passes to the function the new red, green and blue components along with 
  	# which row of pixels will change color (i - from vertical row 0 to 255)
	change(r, g, b, i)
	i = i + 1

# Creates and saves the new image
newImage = Image.new("RGB", image.size)
newImage.putdata(new_pixels)
newImage.save("/Users/erikborn/Documents/Python/jupyter/Codio/shade.jpg")

