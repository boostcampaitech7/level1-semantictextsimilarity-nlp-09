#!/bin/bash

folder_list=("donghoon" "donghyuk" "heejun" "jeongeun" "minjun")
email_list=("woosa0114@gmail.com" "pangyongpy@gmail.com" "y10cuk@naver.com" "irin7348@naver.com" "mj84242160@gmail.com")
name_list=("mrsuit0114" "someDeveloperDH" "gwaksital" "wjddms4299" "minjunk2")
branch_list=("donghoon" "donghyuk" "heejun" "jeongeun" "minjun")

remote_url="git@github.com:gwaksital/boostcamp_project1-Semantic_Text_Similarity.git"

for i in "${!folder_list[@]}"; do
    folder="${folder_list[$i]}"
    email="${email_list[$i]}"
    name="${name_list[$i]}"
    branch="${branch_list[$i]}"

    cd "$folder" || { echo "Directory $folder not found"; continue; }
    git init
    git remote add origin "$remote_url"
    git config --local user.email "$email"
    git config --local user.name "$name"
    git fetch origin
    git checkout -b "$branch" "origin/$branch"
    echo -e "\ngit init with .sh file\n" >> generate_server.txt
    git add .
    git commit -m 'git init with .sh file'
    git push origin "$branch"
    cd ..

done