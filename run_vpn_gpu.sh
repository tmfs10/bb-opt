docker pull haoyangz/ec2-launcher-pro
docker run -i -v /cluster/ec2:/cluster/ec2 -v /etc/passwd:/root/passwd:ro \
	-v /cluster/ec2/cred:/credfile:ro \
	-v $(pwd)/docker_cmd.txt:/commandfile \
	--rm haoyangz/ec2-launcher-pro python ec2run.py GPU VPN -u root -p 12.5 -i p3.8xlarge -a ami-6d720012 -n 5 -e tmfs10@gmail.com
