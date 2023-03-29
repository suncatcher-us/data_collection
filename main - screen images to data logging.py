import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
import pandas as pd
import os
import pathlib
import datetime

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
custom_config = r'--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789wW'
dataPath = "C:\\Users\\Windows\\Desktop\\SUN\\Portable\\BatteryMonitor\\SCREENSHOTS\\"
Columns = ['Date', 'Name', 'Time', 'Input', 'Output', 'Energy']

def processImage(img, scalePercent=80):
    # converts image to black on white text and scales for recognition
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = 3
    kernel = np.ones((size, size), np.float32) / (size * size)
    alpha = 0.1  # Contrast control (1.0-3.0) #0.3
    beta = 10  # Brightness control (0-100) #80
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    img = cv2.filter2D(img, -1, kernel)  # smooth
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scale_percent = scalePercent  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img


def getOneData(file):
    #takes a screenshot file and pulls out the numbers for input and output power
    #https://towardsdatascience.com/read-text-from-image-with-one-line-of-python-code-c22ede074cac
    img = cv2.imread(file)
    #output
    imgOut = processImage(img[750:1186, 544:], 80) #[750:1180, :] #cut block with output
    # cv2.imwrite(dataPath+str(file)[-16:-4]+'.jpg', img)
    rawTextOut = pytesseract.image_to_string(imgOut, config=custom_config)
    #input
    imgIn = processImage(img[1240:1320, 196:480], 80) #[750:1180, :] #cut block with Input
    # cv2.imwrite(dataPath+str(file)[-16:-4]+'.jpg', img)
    rawTextIn = pytesseract.image_to_string(imgIn, config=custom_config)
    rawText = rawTextIn + rawTextOut
    #print(rawText)
    text = str(rawText).replace(' 1','1').replace('\n', ' ').replace('W','').replace('w','')
    textList = text.split(' ')
    numbers = [t for t in textList if t.isdigit()]
    if len(numbers) != 2:
        print(str(file) + " " + str(rawText))
        return (0,0, rawText)
    # else:
    #     print(str(numbers))
    return (int(numbers[0]), int(numbers[1]), rawText)

def getData(today):
    # scans over all the screen shot files for this day and returns
    # a data list suitable for a Pandas dataframe
    folderName = today.strftime("%Y-%m-%d")
    print(folderName)
    files = list(pathlib.Path(dataPath + folderName).iterdir())
    dataList = []
    i = 0
    errors = 0
    energy = 0
    totalEnergy = 0
    input0, output0 = (0,0)
    outputList = []
    for file in files: #[70:100]:
        if file.is_file():
            name = str(file)[-16:-4]
            dateTime = pd.to_datetime(name, format="%H%M%d%m%Y")
            day = dateTime.date()
            minutes = 60*dateTime.hour+dateTime.minute
            # print(minutes)
            (input, output, rawText) = getOneData(str(file))
            if input > 1300: #bug reading data, use last value
                input = input0
            if output > 2000:  # bug reading data, use last value
                output = output0
            outputList.append(output)
            (input0, output0) = (input, output)
            totalEnergy = totalEnergy + input/60.0
            energy = energy + (input - output)/60.0
            # if correctInput[i] != int(input):
            #     errors = errors+1
            #     print("ERROR " + name + "  " + str(correctInput[i]) + "-->" + str(input))
            i = i+1
            print(str(i-1) + "  " + str(minutes//60)+":"+str(minutes%60) + "  " + str((input, output))+ "  " + str(file)+ '\n')
            # store data for this minute
            dataList.append([day, name, minutes, input, output, energy])
            # if int(numbers[0])>1000:
            #     print(str(file) + '\n' + rawText)
            # if len(numbers) != 2:
            #     badCount+=1
    # print("Total ERRORS = %d" % errors)
    df = pd.DataFrame(dataList, columns=Columns)
    df.sort_values(by='Time')
    df.to_csv(dataPath + "data.csv")
    print("total Energy in = " + str(totalEnergy))
    return df

pathlib.Path(dataPath + "SCREENSHOTS")
# works on today = datetime.date(2023, 3, 8)
correctInput = [622,625,633,646,670,674,665,635,611,583,552,522,498,486,482,483,486,488,481,473,460,439,413,394,379,373,369,367,365,354,335,322,314,306,296,285,
273,267,256,254,248,240,230,227,223,218,216,219,218,222,224,234,248,256,256,256,262,271,279,285,292,302,303,301,295,291,288,289,298,316,337,350,354,346,334,328,
321,318,320,323,320,322,324,323,324,329,329,331,335,337,339,337,330,324,318,314,310,303,290,281,278,281,285,289,296,307,310,310,307,301,298,295,287,280,274,271,
270,272,276,279,277,275,270,268,268,268,266,268,270,272,274,274,272,269,264,260,257,251,245,241,238,237,236,231,226,219,218,221,226,230,232,236,240,244,247,254,
265,281,298,318,337,352,364,370,371,367,358,352,333,317,294,276,256,235,211,190,171,154,142,136]
#getData()

#https://towardsdatascience.com/working-with-datetime-in-pandas-dataframe-663f7af6c587
#https://www.geeksforgeeks.org/different-ways-to-create-pandas-dataframe/


# df = pd.read_csv('Popular_Baby_Names.csv')
# df = pd.read_csv('data/city_sales.csv',parse_dates=['date'])
# df.info()

# df.loc['2018-5-1']
#
# df_sorted = df.sort_values(by='Count')
# df_query = df.query('Count > 30 and Rank < 20')
#
# df.to_csv()

def display(df, today):
    timeIterations = df["Time"]
    inputPower = df["Input"]
    outputPower = df["Output"]
    energy = df["Energy"]

    # fig, ax = plt.subplots(figsize=(8, 6))
    fig, (axPower, axEnergy) = plt.subplots(2, 1)
    fig.suptitle('Solar Power and Energy ' + today.strftime("%m/%d/%Y"), fontsize=16)
    # ax.xaxis.set_major_locator(md.HourLocator(interval=1))
    # ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
    # fig.autofmt_xdate()
    axPower.set(xlabel='Time', ylabel='Power (W) in vs. out')

    axPower.set_title('Power vs. time', y=0.9, pad=0)
    axPower.plot(timeIterations, np.array(inputPower),
            alpha = 0.5,
            color = 'orange',
            drawstyle = "steps")
    inputLegend = axPower.fill_between(timeIterations, np.array(inputPower),
            step = "pre",
            alpha = 0.5,
            color = 'orange',
            antialiased = True)
    axPower.plot(timeIterations, np.array(outputPower),
            alpha = 0.5,
            color = 'grey',
            drawstyle = "steps")
    powerLegend = axPower.fill_between(timeIterations, np.array(outputPower),
            step = "pre",
            alpha = 0.5,
            color = 'grey',
            antialiased = True)
    # Create the legend, combining the yellow rectangle for the
    # uncertainty and the 'mean line'  as a single item
    axPower.legend([inputLegend, powerLegend], ["Input power", "Output power"], loc=2)
    # energy
    axEnergy.set(xlabel='Time', ylabel='Energy (Wh) in Battery')
    axEnergy.set_title('Energy vs. time', y=0.9, pad=0)
    axEnergy.plot(timeIterations, np.array(energy),
            alpha = 0.3,
            color = 'blue',
            drawstyle = "steps")
    energyLegend = axEnergy.fill_between(timeIterations, np.array(energy),
            step = "pre",
            alpha = 0.3,
            color = 'blue',
            antialiased = True)

    axEnergy.legend([energyLegend], ["Energy in battery"], loc=2)
    # print(timeIterations)
    xTicks = [timeIterations[i] for i in range(0, len(timeIterations), 120)]
    axPower.set_xticks(xTicks)
    axEnergy.set_xticks(xTicks)
    # Set ticks labels for x-axis
    axPower.set_xticklabels(["%d:00" % (xTicks[i]//60) for i in range(len(xTicks))], fontsize=12) #, rotation='vertical'
    axEnergy.set_xticklabels(["%d:00" % (xTicks[i] // 60) for i in range(len(xTicks))],
                            fontsize=12)  # , rotation='vertical'
    axPower.set_ylim(0, 1700)
    axEnergy.set_ylim(0, 2300)
    plt.show()
    filename = 'dataDay' + '_'  + str(today) + ".png"
    savepath = os.path.join(dataPath, filename)
    fig.savefig(savepath, dpi=500)
    print(savepath)

# today = datetime.date(2023, 3, 8)
# df = getData(today)
# df = pd.read_csv(pathlib.Path(dataPath + 'data.csv'))
# print(df)
# display(df)
#today = datetime.date(2023, 3, 12)
today = datetime.date(2023, 3, 19)
df = getData(today)
display(df, today)