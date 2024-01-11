import numpy as np

import tkinter as tk

from scipy.integrate import solve_ivp


import matplotlib
matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from matplotlib.figure import Figure

#List of parameter names
paramNameList = ["bd1","d2","d3","a11","a12","a21",
             "a22","a23","a32","a33","ix1","ix2","ix3"]


#Dictionary of parameter values with initial values provided
paramDictionary = {
    "bd1": 3.5,
    "d2": 0.5,
    "d3": 0.5,
    "a11": 0.5,
    "a12": 0.5,
    "a21": 0.5,
    "a22": 0.5,
    "a23": 0.5,
    "a32": 0.5,
    "a33": 0.5,
    "ix1": 5.0,
    "ix2": 3.5,
    "ix3": 2.0}

#Small function to get the parameter values from
#paramDictionary as a list
def getParamValuesAsList():
    paramList = []
    for paramName in paramNameList:
        paramList.append(paramDictionary[paramName])
    
    return paramList
        

#Empty lists of length 500 to contain simulation results (500 timesteps)
x1 = [0] * 500
x2 = [0] * 500
x3 = [0] * 500
t = [0] * 500


#Dictionary of simulation results accessed for plotting
resultsDictionary = {
    "x1": x1,
    "x2": x2,
    "x3": x3,
    "t": t}

#Variables for holding predicted steady state values
x1Pred = 0
x2Pred = 0
x3Pred = 0

#Dictionary for referencing predicted steady state values
predictionsDictionary = {
    "x1": x1Pred,
    "x2": x2Pred,
    "x3": x3Pred}

#Options for setting axis variables
AXISOPTIONS = [
        "x1",
        "x2",
        "x3",
        "t"
        ]


#The model set in the exercise
def model (t, z, bd1, d2, d3, a11, a12, a21, a22, a23, a32, a33):
    x1, x2, x3 = z
    
    dx1dt = x1*(bd1 - a11*x1 - a12*x2)
    dx2dt = x2*(- d2 + a21*x1 - a22*x2 - a23*x3)
    dx3dt = x3*(- d3 + a32*x2 - a33*x3)
    
    
    return [dx1dt, dx2dt, dx3dt]


#Predict steady states and store in the corresponding dictionary
def predict(bd1, d2, d3, a11, a12, a21, a22, a23, a32, a33):
    denominator = (a11*a22*a33) + (a12*a21*a33) + (a11*a23*a32)
    x1Numerator = (a12*a33*d2) + \
        ((bd1)*((a22*a33) + (a23*a32))) - (a12*a33*d3)
    x2Numerator = ((bd1)*a21*a33) + \
        (a11*a23*d3) - (a11*a33*d2)
    x3Numerator = ((bd1)*a21*a32) - \
        (a11*a32*d2) - (d3*((a11*a22) + (a12*a21)))
    
    predictionsDictionary["x1"] = x1Numerator/denominator
    predictionsDictionary["x2"] = x2Numerator/denominator
    predictionsDictionary["x3"] = x3Numerator/denominator
    

#Run solve_ivp to simulate the model over 500 timesteps
#Saves the results to the resultsDictionary so they can
#be accessed for plotting
def solveModel(bd1, d2, d3, a11, a12, a21, a22, a23, a32, a33, ix1, ix2, ix3):
    sol = solve_ivp(model, [0, 50], [ix1, ix2, ix3],
                args = (bd1, d2, d3, a11, a12, a21, a22, a23, a32, a33),
                t_eval = np.linspace(0,50,500),
                dense_output=True)
    resultsDictionary["x1"], resultsDictionary["x2"], resultsDictionary["x3"] = sol.y
    
    resultsDictionary["t"] = sol.t


#Makes a quiver plot if two population variables are selected
#Or a population - time plot otherwise. Also plots the expected
#steady state assuming that there is a steady state with all three
#species present
def makePlot(plottingVariables, plottingCanvas, axes):
    #Clears the matplotlib plotting object if there is already a
    #plot present
    axes.clear()
    
    timeCourse = False
    
    #Check if one the axes is "t"
    for i in range (0,2):
        if plottingVariables[i] == "t":
            timeCourse = True
    
    #Plot a time course
    if timeCourse:
        plotTimeCourse(plottingVariables, axes)
    
    #Quiver plot with two populations
    else:
        plotVariables(plottingVariables, axes)
    
    #Uncomment following lines if you wish to keep the origin at 0,0:
    #a.set_ylim(ymin=0)
    #a.set_xlim(xmin=0)
    xAxis = plottingVariables[0]
    yAxis = plottingVariables[1]
    axes.set_xlabel(xAxis)
    axes.set_ylabel(yAxis)
    plottingCanvas.draw()


#Plot population over time, with a red bar showing expected steady state
def plotTimeCourse(plottingVariables, axes):
    if plottingVariables[0] == "t":
        if plottingVariables[1] == "t":
            axes.plot(resultsDictionary[plottingVariables[0]],
                   resultsDictionary[plottingVariables[1]], color = "black")
        else:
            axes.plot(resultsDictionary[plottingVariables[0]],
                   resultsDictionary[plottingVariables[1]], color = "black")
            axes.axhline(y = predictionsDictionary[plottingVariables[1]],
                      color = 'r', linestyle = '-')
        
    else:
        axes.plot(resultsDictionary[plottingVariables[0]],
               resultsDictionary[plottingVariables[1]], color = "black")
        axes.axvline(x = predictionsDictionary[plottingVariables[0]],
                  color = 'r', linestyle = '-')
     
        
##Based on implementation of quivers from https://stackoverflow.com/questions/36607742/drawing-phase-space-trajectories-with-arrows-in-matplotlib
##StackOverflow user tfv
#Plot the trajectory of two populations over time with arrows
#Add a red dot showing the expected steady state
def plotVariables(plottingVariables, axes):
    combinedData = np.zeros((2,500))
    combinedData[0] = resultsDictionary[plottingVariables[0]]
    combinedData[1] = resultsDictionary[plottingVariables[1]]
    
    axes.quiver(combinedData[0,:-1], combinedData[1,:-1],
           combinedData[0,1:]-combinedData[0,:-1],
           combinedData[1,1:]-combinedData[1,:-1],
           scale_units='xy', angles='xy', scale=1, width=0.005)
    axes.plot(predictionsDictionary[plottingVariables[0]],
           predictionsDictionary[plottingVariables[1]],'ro')


#Get the variables from drop down menus that determine
#which variables should be plotted
def getPlottingVariables():
    varString1 = var1.get()
    varString2 = var2.get()
    bothStrings = [varString1, varString2]
    
    return bothStrings


#Triggered by rerun button. Checks for changes in parameter
#values and then reruns solving, prediction and plotting
def rerunModel(canvas, axes, plottingVariables):
    #reset values
    
    for paramName in paramNameList:
        if len(paramVarDictionary[paramName].get()) != 0:
            value = float(paramVarDictionary[paramName].get())
            if value > 0:
                paramDictionary[paramName] = value
    
    paramList = getParamValuesAsList()
    
    solveModel(*paramList)

    predict(*paramList[0:10])
    
    makePlot(plottingVariables, canvas, axes)
    
    clearCurrentValues()
    
    recordCurrentValues()
    
    for e in entryBoxes:
        e.delete(0,tk.END)


#Checks current parameter values and prints their values above the
#matplotlib plot, packing the text into the tkinter widget
def recordCurrentValues():
    global currentValues1, currentValues2
    firstLine1 = "Current values: b - d1 = " + str(round(paramDictionary["bd1"],2)) + ","
    firstLine2 = ""
    for paramName in paramNameList[1:10]:
        firstLine2 = firstLine2 + " " + paramName + " = " + str(round(paramDictionary[paramName],2)) + ","
    firstLine = firstLine1 + firstLine2
    currentValues1 = tk.Label(sixthFrame, text=firstLine)
    currentValues1.pack(padx = 5, pady = 0, side = tk.LEFT)

    currentValues2 = tk.Label(seventhFrame, text="x1 initial value = " + str(round(paramDictionary["ix1"],2)) + ", x2 initial value = " + str(round(paramDictionary["ix2"],2)) + ", x3 initial value = " + str(round(paramDictionary["ix3"],2)) + ". Do not input values of zero or less.")
    currentValues2.pack(padx = 5, pady = 0, side = tk.LEFT)
    
    sixthFrame.pack(fill = "x", expand="True")

    seventhFrame.pack(fill = "x", expand="True")
    
    
#Clears the lines of text above the plot stating the current
#parameter values
def clearCurrentValues():
    currentValues1.pack_forget()
    currentValues2.pack_forget()


#Create tkinter widget
root = tk.Tk()
root.title("Python Tkinter Embedding Matplotlib")
root.minsize(640, 400)


##Place entry boxes in frames, and add all entry boxes
#to a growing list
topFrame = tk.Frame(root)

entryBoxes = []

bd1Var = tk.StringVar()
bd1Label = tk.Label(topFrame, text = "b - d1")
bd1Entry = tk.Entry(topFrame, textvariable = bd1Var)
entryBoxes.append(bd1Entry)

d2Var = tk.StringVar()
d2Label = tk.Label(topFrame, text = "d2")
d2Entry = tk.Entry(topFrame, textvariable = d2Var)
entryBoxes.append(d2Entry)


d3Var = tk.StringVar()
d3Label = tk.Label(topFrame, text = "d3")
d3Entry = tk.Entry(topFrame, textvariable = d3Var)
entryBoxes.append(d3Entry)

secondFrame = tk.Frame(root)

a11Var = tk.StringVar()
a11Label = tk.Label(secondFrame, text = "a11")
a11Entry = tk.Entry(secondFrame, textvariable = a11Var)
entryBoxes.append(a11Entry)

a12Var = tk.StringVar()
a12Label = tk.Label(secondFrame, text = "a12")
a12Entry = tk.Entry(secondFrame, textvariable = a12Var)
entryBoxes.append(a12Entry)

a21Var = tk.StringVar()
a21Label = tk.Label(secondFrame, text = "a21")
a21Entry = tk.Entry(secondFrame, textvariable = a21Var)
entryBoxes.append(a21Entry)

thirdFrame = tk.Frame(root)

a22Var = tk.StringVar()
a22Label = tk.Label(thirdFrame, text = "a22")
a22Entry = tk.Entry(thirdFrame, textvariable = a22Var)
entryBoxes.append(a22Entry)

a23Var = tk.StringVar()
a23Label = tk.Label(thirdFrame, text = "a23")
a23Entry = tk.Entry(thirdFrame, textvariable = a23Var)
entryBoxes.append(a23Entry)

a32Var = tk.StringVar()
a32Label = tk.Label(thirdFrame, text = "a32")
a32Entry = tk.Entry(thirdFrame, textvariable = a32Var)
entryBoxes.append(a32Entry)

fourthFrame = tk.Frame(root)

a33Var = tk.StringVar()
a33Label = tk.Label(fourthFrame, text = "a33")
a33Entry = tk.Entry(fourthFrame, textvariable = a33Var)
entryBoxes.append(a33Entry)

ix1Var = tk.StringVar()
ix1Label = tk.Label(fourthFrame, text = "x1 initial value")
ix1Entry = tk.Entry(fourthFrame, textvariable = ix1Var)
entryBoxes.append(ix1Entry)

fifthFrame = tk.Frame(root)

ix2Var = tk.StringVar()
ix2Label = tk.Label(fifthFrame, text = "x2 initial value")
ix2Entry = tk.Entry(fifthFrame, textvariable = ix2Var)
entryBoxes.append(ix2Entry)

ix3Var = tk.StringVar()
ix3Label = tk.Label(fifthFrame, text = "x3 initial value")
ix3Entry = tk.Entry(fifthFrame, textvariable = ix3Var)
entryBoxes.append(ix3Entry)


#Dictionary of all tkinter StringVar objects corresponding to
#user-inputted parameter values
paramVarDictionary = {
    "bd1": bd1Var,
    "d2": d2Var,
    "d3": d3Var,
    "a11": a11Var,
    "a12": a12Var,
    "a21": a21Var,
    "a22": a22Var,
    "a23": a23Var,
    "a32": a32Var,
    "a33": a33Var,
    "ix1": ix1Var,
    "ix2": ix2Var,
    "ix3": ix3Var}



##Adapted from https://codeloop.org/how-to-embed-matplotlib-in-tkinter-window/
#Create a matplotlib figure in a canvas that can be placed into the widget
f = Figure(figsize=(5,4), dpi=100)
axesSubplot = f.add_subplot(111)

canvas = FigureCanvasTkAgg(f, root)


#X axis variable for drop-down menu
var1 = tk.StringVar()
var1.set(AXISOPTIONS[0])

#Y axis variable for drop down menu
var2 = tk.StringVar()
var2.set(AXISOPTIONS[1])

#Do initial solving and prediction
paramValues = getParamValuesAsList()
solveModel(*paramValues)
predict(*paramValues[0:10])

#Make the initial plot
makePlot(getPlottingVariables(), canvas, axesSubplot)

#Add a toolbar for manipulating the matplotlib plot
toolbar = NavigationToolbar2Tk(canvas, root)
toolbar.update()


#Add the drop down menus for plotting
dropDown1 = tk.OptionMenu(root, var1, *AXISOPTIONS, command=lambda value:makePlot(getPlottingVariables(), canvas, axesSubplot))
dropDown1Label = tk.Label(root, text = "x axis:")

dropDown2 = tk.OptionMenu(root, var2, *AXISOPTIONS, command=lambda value:makePlot(getPlottingVariables(), canvas, axesSubplot))
dropDown2Label = tk.Label(root, text = "y axis:")


#Pack everything for display in the widget
bd1Label.pack(padx = 5, pady = 0, side = tk.LEFT)
bd1Entry.pack(padx = 5, pady = 0, side = tk.LEFT)

d2Label.pack(padx = (15,5), pady = 0, side = tk.LEFT)
d2Entry.pack(padx = 5, pady = 0, side = tk.LEFT)

d3Label.pack(padx = (15,5), pady = 0, side = tk.LEFT)
d3Entry.pack(padx = 5, pady = 0, side = tk.LEFT)

topFrame.pack(fill="x", expand="True")

a11Label.pack(padx = 5, pady = 0, side = tk.LEFT)
a11Entry.pack(padx = 5, pady = 0, side = tk.LEFT)

a12Label.pack(padx = (15,5), pady = 0, side = tk.LEFT)
a12Entry.pack(padx = 5, pady = 0, side = tk.LEFT)

a21Label.pack(padx = (15,5), pady = 0, side = tk.LEFT)
a21Entry.pack(padx = 5, pady = 0, side = tk.LEFT)

secondFrame.pack(fill="x", expand="True")

a22Label.pack(padx = 5, pady = 0, side = tk.LEFT)
a22Entry.pack(padx = 5, pady = 0, side = tk.LEFT)

a23Label.pack(padx = (15,5), pady = 0, side = tk.LEFT)
a23Entry.pack(padx = 5, pady = 0, side = tk.LEFT)

a32Label.pack(padx = (15,5), pady = 0, side = tk.LEFT)
a32Entry.pack(padx = 5, pady = 0, side = tk.LEFT)

thirdFrame.pack(fill="x", expand="True")

a33Label.pack(padx = 5, pady = 0, side = tk.LEFT)
a33Entry.pack(padx = 5, pady = 0, side = tk.LEFT)

ix1Label.pack(padx = (15,5), pady = 0, side = tk.LEFT)
ix1Entry.pack(padx = 5, pady = 0, side = tk.LEFT)

fourthFrame.pack(fill="x", expand="True")

ix2Label.pack(padx = 5, pady = 0, side = tk.LEFT)
ix2Entry.pack(padx = 5, pady = 0, side = tk.LEFT)

ix3Label.pack(padx = (15,5), pady = 0, side = tk.LEFT)
ix3Entry.pack(padx = 5, pady = 0, side = tk.LEFT)

rerunButton = tk.Button(fifthFrame, text="Rerun", command = lambda: rerunModel(canvas, axesSubplot, getPlottingVariables()))
rerunButton.pack(padx = 5, pady = 0, side = tk.RIGHT)

fifthFrame.pack(fill="x", expand="True")

sixthFrame = tk.Frame(root)

seventhFrame = tk.Frame(root)

recordCurrentValues()

canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=False)

dropDown1.pack(padx=5, pady=0, side=tk.LEFT)
dropDown1Label.pack(padx=5, pady=0, side=tk.LEFT)

dropDown2Label.pack(padx = (30,5), pady = 0, side = tk.LEFT)
dropDown2.pack(padx = 5, pady = 0, side = tk.LEFT)


#Main loop of the tkinter widget
root.mainloop()




