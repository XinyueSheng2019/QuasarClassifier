
# Damped Harmonic Oscillator AGN light curve simulation

from matplotlib import pyplot as plt
from math import sin, fabs, pi

def f(k,x):    #The restoring force due to the spring
    restoring_force = -k*x
    return restoring_force

def d(b,v):     #The drag force. b is the drag coefficent and v is the speed
    drag_force = -b*v
    return drag_force

def driving(a, f, t): #The driving force.  This is a function of time. f is the frequency of the driving force and a is its maximum value
    driving_force = a*sin(2*pi*f*t)
    return driving_force

def cycle(my_list, new): #Takes a three item list, moves the elements down one place and adds a new element into position [2]
    my_list[0] = my_list[1]
    my_list[1] = my_list[2]
    my_list[2] = new
    return my_list
      

def simulate_oscillations(driving_force_amplitude, driving_frequency, drag_coefficient, k, m, dt, option): #This does all the work.
    
    x = 3 # Initial dispalcement
    t,v = 0,2 #Initial time and speed
    
    x_list = [0,0,0]            #This list keeps track of the three previous x values. If the number in the middle is the largest then 
                                #we have found a local maximum (i.e. the amplitude).
    displacement = []
    time = []                   #
                                #These are lists that will hold the data used to produce a pretty graph at the end
    amplitude_list = []         #
    time1 = []
    
    
    flag = 0    
    
    while flag == 0:    #This loop keeps going until we are satisfied that the amplitude has reached a stable, maximum value
    
        r = f(k,x) + d(drag_coefficient,v) + driving(driving_force_amplitude, driving_frequency, t) #Calculates the resultant force on the mass
        a = r / m                   #Calculates the acceleration of the mass
        v = v + (a*dt)              #Updates the speed
        x = x + (v*dt)              #Updates the position
        x_list = cycle(x_list, x)   #Updates x_list with the latest x value
        
        if (x_list[1]) > (x_list[0]) and (x_list[1]) > (x_list[2]):  #Checks to see if x_list[1] is larger than x_list[0] and x_list[2]
            amplitude_list.append(x_list[1])                         #If it is, we add this to our list of amplitudes
            time1.append(t-dt)                                       #Records the time at which the mass reached the final amplitude

        l = len(amplitude_list)
        if l > 3: #Wait until we have 3 amplitues to compare
            if fabs(amplitude_list[l-1] - amplitude_list[l-2]) < 1e-5 and fabs(amplitude_list[l-2] - amplitude_list[l-3]) < 1e-5: #If the amplitude is changing by less than 0.0001 each cycle then it is constant
                #print("The amplitude of this system is ", amplitude_list[l-1], 'when the driving frequency is ', driving_frequency) #Outputs the final amplitude of the system. This line can be removed for long runs
                flag = 1 #Breaks out of the loop
            
        time.append(t)          #This data can be used to plot a pretty graph of displacement
        displacement.append(x)  #against time to check individual simulations.  not used in final version

        t = t + dt              #Advances the time

    if option == 'show_graph':   
        plt.plot(time1, amplitude_list)
        plt.plot(time, displacement)
        plt.suptitle('A forced, damped oscilator.')
        plt.xlabel('time')
        plt.ylabel('displacement')
        plt.grid('on')
        plt.show()

    return amplitude_list[l-1]

def run_the_experiment(a,b,c):
    step_size = (b-a)/c
    run = 0
    f_list = []
    results = []
    while run <=c:
        results.append(simulate_oscillations(3, a, 0.4, 20, 0.5, 0.001, 'show_graph'))  #(driving_force_amplitude, driving_frequency, drag_coefficient, k, m, dt, option)
        f_list.append(a)
        a = a + step_size
        run = run + 1

    plt.plot(f_list, results)
    plt.suptitle('Frequency response of damped oscillator')
    plt.xlabel('Driving Frequency')
    plt.ylabel('Amplitude')
    plt.grid('on')
    plt.draw
    plt.show()


run_the_experiment(0.1, 1.5, 1) #run_the_experiment(begin frequecy range, end of frequency range, number of points between a and b)

