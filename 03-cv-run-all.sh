#!/bin/bash

echo ''
echo 'Decision Tree'
echo ''
python3 03-cv.py decision_tree

echo ''
echo 'Linear SVC'
echo ''
python3 03-cv.py linear_svc

echo ''
echo 'Random Forest'
echo ''
python3 03-cv.py random_forest

echo ''
echo 'KNN'
echo ''
python3 03-cv.py knn
#python3 03-cv.py gradient_boosting


