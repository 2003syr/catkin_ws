cmake_minimum_required(VERSION 2.8.3)

project(mobile_arm_description)

find_package(catkin REQUIRED)

catkin_package()

find_package(roslaunch)

catkin_install_python(PROGRAMS
  scripts/arm_idle_position_hold.py
  scripts/enforce_link_gravity.py
	scripts/diagnose_idle_motion.py
	scripts/verify_arm_initial_pose.py
	scripts/verify_planar_base_response.py
	scripts/verify_controllers.py
	scripts/verify_velocity_response.py
	DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

foreach(dir config launch meshes models urdf)
	install(DIRECTORY ${dir}/
		DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)

if(CATKIN_ENABLE_TESTING)
	catkin_add_nosetests(test/test_arm_idle_position_hold.py)
	catkin_add_nosetests(test/test_description_contract.py)
endif()
