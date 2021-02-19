# logdown.py
ssh TimtheTyrant@login05.osgconnect.net '
rm worker/Log.tar.gz;
cd worker;
./post_process.sh
'
scp TimtheTyrant@login05.osgconnect.net:worker/Log.tar.gz osg_output/Log.tar.gz
tar -xzf osg_output/Log.tar.gz
python3 consolidate-osg-output.py

# mv /home/timothytyree/Documents/GitHub/care_worker/python/osg_output/longest_traj_by_area_pbc.csv /home/timothytyree/Documents/GitHub/care_worker/python/osg_output/longest_traj_by_area_fk_pbc.csv
