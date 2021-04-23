#!/bin/bash
# init script for Ubuntu based servers
sudo apt-get update -y
sudo apt-get install     apt-transport-https     ca-certificates     curl     gnupg-agent     software-properties-common -y
sudo apt-get update - y
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository    "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update -y
sudo apt-get install docker-ce docker-ce-cli containerd.io -y
sudo usermod -aG docker ubuntu
sudo apt-get install python3-pip parallel -y
sudo curl -L "https://github.com/docker/compose/releases/download/1.25.4/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose
parallel --number-of-cpus
parallel --number-of-cores

# required for the way Azure allocates temporary storage; allows to run as-is but link to larger disk located at /mnt
sudo mkdir /mnt/covid19
sudo chown ubuntu:ubuntu /mnt/covid19
ln -s /mnt/covid19 .

