# EDA_Projects
Two data science and electronics-related projects developed during college using Spyder Simulator and LTspice.

**Current Status:** Done ✅

## Project 2
This work aims to develop a Python script for the automatic determination of Thin Film Transistor(TFT) model parameters.

A second objective of this project considers comparing different approaches for the evaluation of the model parameters.

## Methodology

**Part 1:**
1. Obtain the TFT output and transfer characteristics and plot them.
2. Using the transfer characteristic, in saturation:
   a) Obtain the analytical expression for gm.
   b) Obtain the analytical expression for ID/gm.
   c) Numerically obtain the array containing gm.
   d) Obtain Vt, and m by fitting the numerically obtained data for Id/gm to the expression in b.
   e) Obtain K.
   f) Validate results.

**Part 2:**
1. Obtain the TFT transfer characteristic in the linear region and plot it.
2. Using the transfer characteristic obtained in 1. proceed with:
   a) Obtain the analytical expression for dg = dev^2ID/devVgs.
   b) Conclude on how to evaluate Vt.
   c) Numerically obtain the array containing dg.
   d) Obtain Vt, from dg.
3. Using the obtained value for Vt, determine the remaining parameters.
4. Compare results obtained in Part1 and Part2.

**Part 3:**
1. Using the transfer characteristic in the saturation regime, obtain by optimization the values for the model parameter, i.e , K, m, VT and Rs.
2. Validate the results obtained by comparing results obtained with the model against the initial transfer characteristic.
3.
    a) Obtain the expression for gm.
    b) Obtain the expression for F=ID/gm as a function of ID.
    c) Using optimisation, determine the values for Rs, K and m.
    d) Using curve_fit obtain value for VT (use the values obtained in c., as initial points for the optimisation).
    e) Validate the results obtained by comparing the output characteristic in saturation obtained with the model, against the initial transfer characteristic.

(Note:In this part the contact resistance at the source, Rs, should be considered)

## Project 3
This work aims to introduce the design of integrated inductors.

## Methodology

**Part 1:**
Implement in Python the functions that allow to determine the values of the components of the p-model of integrated inductors.

**Part 2:**
Considering that the integrated inductor has one of the terminals connected to ground, implement a function in Python that allows determining the complex impedance of the circuit: 
1. Determine the inductor inductance value for a frequency of 1GHz.
2. Determine the inductor quality factor for a frequency of 2.4GHz.
3. Plot the inductance graph as a function of frequency, for a frequency between 1e4 Hz and 1e11 Hz.
4. Plot the inductor quality factor as a function of frequency, for a frequency between 1e4 Hz and 1e11 Hz.
5. Determine the resonance frequency of the inductor.

**Part 3:**
In Ltspice edit the schematic of the circuit. Considering the values of the components obtained in Python, obtain the inductance and quality factor plots as a function of frequency, for a frequency between 1e2 and 1e4 Hz and 1e1 and 1e11 Hz. 2. Compare the plots obtained in Ltspice, with those generated in Python.

**Part 4:**
1. In Python, using optimization functions, determine the geometric parameter ( i.e Dout, n, and w) values for a 4.5nH octagonal inductor operating at a frequency of 2.4 GHz.
2. For the solution obtained in 1., indicate the quality factor of the inductor, in that frequency, as well as the value of the respective resonance frequency.
3. Repeat 1., considering that the frequency of resonance must be higher than 20 GHz.
4. Consider that, due to implementation constraints, the number of turns can only have a decimal part of 0 or 5. How can you modify results obtained in 2.so that this constraint is satisfied.

**Part 5:**
Develop the VerilogA model the inductor. For the inductor parameters obtained in Part 4, obtain the corresponding frequency response.
