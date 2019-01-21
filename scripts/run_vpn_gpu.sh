docker pull haoyangz/ec2-launcher-pro
docker run -i -v /cluster/ec2:/cluster/ec2 -v /cluster/sj1:/cluster/sj1 -v /etc/passwd:/root/passwd:ro \
	-v /cluster/ec2/cred:/credfile:ro \
	-v /cluster/sj1/bb_opt/docker_cmd.txt:/commandfile \
	--rm haoyangz/ec2-launcher-pro python ec2run.py GPU VPN -u root -p 2 -i g2.2xlarge -a ami-6d720012 -e tmfs10@gmail.com
