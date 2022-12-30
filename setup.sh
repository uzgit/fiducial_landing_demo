#!/usr/bin/bash

echo "installing..."

alias_ll="alias ll=\"ls -lah\""
grep -E "$alias_ll" ~/.bashrc || echo "$alias_ll" >> ~/.bashrc

sudo apt-get install vim htop tree mlocate git

pip3 install numpy scipy opencv-python
