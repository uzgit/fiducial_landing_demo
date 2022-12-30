#!/usr/bin/bash

echo "installing..."

alias_ll="alias ll=\"ls -lah\""
grep -E "$alias_ll" ~/.bashrc || echo "$alias_ll" >> ~/.bashrc

sudo apt-get -y install vim htop tree mlocate git xdotool python3-opencv

pip3 install numpy

git clone --recursive https://github.com/pupil-labs/apriltags.git
cd apriltags
pip3 install -e .[testing]
