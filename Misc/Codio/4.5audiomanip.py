# ######################################
#       Post Audio Manipulation      #
#                                    #
#            UTeach CSP				 #
#                                    #
######################################

# Importing wave library and os library
import wave
import os
import random

# Deletes old created audio files if they exist
if os.path.exists("Misc/Codio/newAudio.wav"):
	os.remove("Misc/Codio/newAudio.wav")

# Opens original audio file in a readable format
sound = wave.open("Misc/Codio/guitarchords.wav", "rb")

# Returns number of audio channels (1 for mono, 2 for stereo)
channels = sound.getnchannels()
print("Audio Channels: ", channels)

# Returns sample width in bytes
sampwidth = sound.getsampwidth()
print("Sample Width: ", sampwidth)

# Returns sampling frequency
framerate = sound.getframerate()
print("Sampling Frequency: ", framerate)

# Returns number of audio frames
numframes = sound.getnframes()
print("Number of Audio Frames: ", numframes)

# Creates an empty array to store individual frame data
data = []

# Reads the byte of the audio frame and adds it to data[]
# until each frame has been stored to the array
# Each element is stored as a byte object
while (sound.tell() < numframes):
	frame = sound.readframes(1)
	data.append(frame)

#######################
#  Audio Manipulation #
#######################

data.reverse()


######################

# Combines the elements from the data array back into a single string of bytes
samples = b''.join(data)
#print(samples)

# Creates a new audio file that is writeable
sound2 = wave.open("Misc/Codio/newAudio.wav", "wb")

# Sets the channels to the same as original file
sound2.setnchannels(channels)

# Sets the sample width to the same as original File
# This can be an integer from 1 to 4
sound2.setsampwidth(sampwidth)

# Sets the sampling frequency to the same as original file
sound2.setframerate(framerate)

# The string of bytes created above are 
# assigned to the audio file 
sound2.writeframes(samples)

# Closes out of the two audio files
sound2.close()
sound.close()

# Lets you know when the processing has finished
print("Finished")