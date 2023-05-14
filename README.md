# Intelligent Airfoil Design 

## The Problem

High aircraft fuel consumption is a driving force behind global carbon emissions. Emissions in the aviation industry are expected to increase, potentially comprising a quarter of the 1.5 °C carbon budget by 2050. Improving the efficiency of airfoil design is key to the creation of more fuel-efficient airplanes.

## The Current Software

Current component optimization is inaccessible due to its reliance on computational fluid dynamics, which is expensive, difficult to use, and computationally expensive, resulting in inefficient designs. However, rapid advancements in machine learning (ML) techniques have opened new avenues to make this process faster and more affordable. 

## The Solution

We constructed a series of generative machine learning models and evaluated their simulation runtimes and accuracy. Each model was trained to output the structure of an airfoil that most closely produces desired lift, drag and moment coefficients. Models utilizing ensemble methods, including Random Forest Regressors, Extra Trees Regressors, Bagging Regressors and K-Nearest Neighbors Bagging, as well as the multilayer perceptron, produce accurate and efficient predictions of airfoil geometries. Machine learning models present a significant improvement in runtime and ease of use over traditional computational fluid dynamic software. 

## Media

Check out our paper "[Machine Learning Approaches for the Component Optimization of Airfoils](https://github.com/AkiraY1/IntelligentAirfoilDesign/ML_Airfoil_Paper_Final.pdf)"

![Paper](https://github.com/AkiraY1/IntelligentAirfoilDesign/Media/AirfoilPaperPage.png?raw=true)

I thought it was pretty cool seeing XFOIL produce the polar files:

![XFOIL](https://github.com/AkiraY1/IntelligentAirfoilDesign/Media/XFOIL.png?raw=true)
