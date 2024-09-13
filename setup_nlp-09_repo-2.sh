#!/bin/bash

git pull origin master
echo -e "\nnlp-09 repo" >> README.md
git add .
git commit -m 'inital commit'
git push -u origin master