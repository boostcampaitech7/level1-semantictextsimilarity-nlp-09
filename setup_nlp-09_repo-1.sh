#!/bin/bash

email="y10cuk@naver.com"
name="gwaksital"
remote_url="git@github.com:boostcampaitech7/level1-semantictextsimilarity-nlp-09.git"

git init
git config --local user.email "$email"
git config --local user.name "$name"
git remote add origin "$remote_url"
ssh-keygen -t ed25519 -C "$email"
echo "Public key:"
cat ~/.ssh/id_ed25519.pub