# This is a script to get part of the large TOA file data.
import numpy as np

# Define the start time of the TOA5 file
tStartTOA5 = np.datetime64('2016-06-14 18:20:52.5')

# Define time interval you would like to take from the TOA5 file
tStart = np.datetime64('2016-06-18T00:00:00.0')
tEnd = np.datetime64('2016-06-19T00:00:00.0')

# Print out line index of tStart
iStart = int(tStart.astype('int') - tStartTOA5.astype('int')) / 100
print iStart

# Print out line index of tEnd
iEnd = int(tEnd.astype('int') - tStartTOA5.astype('int')) / 100
print iEnd

# Open file to get lines
lines = []
fileName = r"C:\Users\pdphy\Documents\20160601\TOA5_ts3_160627.dat"

with open(fileName, 'r') as f:
    for i, line in enumerate(f):
        if i < 4:
            lines.append(line)
        elif (i > iStart) & (i < iEnd):
            lines.append(line)
            #print i
        elif i > iEnd:
            break

outputFile = r'\Users\pdphy\Documents\20160601\TOA5_ts3_160618.dat'

with open(outputFile, 'w') as f:
    for line in lines:
        f.write(line)
