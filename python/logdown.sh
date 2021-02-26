# logdown.py
ssh TimtheTyrant@login05.osgconnect.net '
rm af2d/Log.tar.gz;
cd af2d;
./post_process.sh
'
scp TimtheTyrant@login05.osgconnect.net:af2d/Log.tar.gz osg_output/Log.tar.gz
tar -xzf osg_output/Log.tar.gz

SAVEFN='osg_output/emsd_by_area_by_diffCoef_pbc.csv'
# SAVEFN='longest_traj_by_area_pbc.csv'
python3 consolidate-osg-output.py $SAVEFN
#move result to care
# mv /home/timothytyree/Documents/GitHub/care_worker/python/osg_output/$(SAVEFN).csv /home/timothytyree/Documents/GitHub/care_worker/python/osg_output/longest_traj_by_area_fk_pbc.csv
