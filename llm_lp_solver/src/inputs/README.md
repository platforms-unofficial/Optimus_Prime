# Inputs 
examples are coming from https://people.brunel.ac.uk/~mastjjb/jeb/or/morelp.html
# Examples
## textbook_1
Let

x be the number of units of X produced in the current week
y be the number of units of Y produced in the current week
then the constraints are:

50x + 24y <= 40(60) machine A time
30x + 33y <= 35(60) machine B time
x >= 75 - 30
i.e. x >= 45 so production of X >= demand (75) - initial stock (30), which ensures we meet demand
y >= 95 - 90
i.e. y >= 5 so production of Y >= demand (95) - initial stock (90), which ensures we meet demand
The objective is: maximise (x+30-75) + (y+90-95) = (x+y-50)
i.e. to maximise the number of units left in stock at the end of the week

It is plain from the diagram below that the maximum occurs at the intersection of x=45 and 50x + 24y = 2400

Solving simultaneously, rather than by reading values off the graph, we have that x=45 and y=6.25 with the value of the objective function being 1.25



## textbook_2
For product 1 applying exponential smoothing with a smoothing constant of 0.7 we get:

M1 = Y1 = 23
M2 = 0.7Y2 + 0.3M1 = 0.7(27) + 0.3(23) = 25.80
M3 = 0.7Y3 + 0.3M2 = 0.7(34) + 0.3(25.80) = 31.54
M4 = 0.7Y4 + 0.3M3 = 0.7(40) + 0.3(31.54) = 37.46

The forecast for week five is just the average for week 4 = M4 = 37.46 = 31 (as we cannot have fractional demand).

For product 2 applying exponential smoothing with a smoothing constant of 0.7 we get:

M1 = Y1 = 11
M2 = 0.7Y2 + 0.3M1 = 0.7(13) + 0.3(11) = 12.40
M3 = 0.7Y3 + 0.3M2 = 0.7(15) + 0.3(12.40) = 14.22
M4 = 0.7Y4 + 0.3M3 = 0.7(14) + 0.3(14.22) = 14.07

The forecast for week five is just the average for week 4 = M4 = 14.07 = 14 (as we cannot have fractional demand).

We can now formulate the LP for week 5 using the two demand figures (37 for product 1 and 14 for product 2) derived above.

Let

x1 be the number of units of product 1 produced

x2 be the number of units of product 2 produced

where x1, x2>=0

The constraints are:

15x1 + 7x2 <= 20(60) machine X

25x1 + 45x2 <= 15(60) machine Y

x1 <= 37 demand for product 1

x2 <= 14 demand for product 2

The objective is to maximise profit, i.e.

maximise 10x1 + 4x2 - 3(37- x1) - 1(14-x2)

i.e. maximise 13x1 + 5x2 - 125

The graph is shown below, from the graph we have that the solution occurs on the horizontal axis (x2=0) at x1=36 at which point the maximum profit is 13(36) + 5(0) - 125 = £343