<?xml version="1.0" encoding="UTF-8" ?>
<mujoco model="reacher">
    <asset>
      <mesh name="forearm" file="@forearm_stl_path"/>
    </asset>

    <compiler angle="radian" inertiafromgeom="true"/>
	<default>
		<joint armature="1" damping="1" limited="true"/>
		<geom contype="0" friction="1 0.1 0.1" rgba="0.7 0.7 0 1"/>
	</default>
	<option gravity="0 0 -9.81" integrator="RK4" timestep="0.01"/>
	<worldbody>
        @[if not no_environment ]
          <!-- Arena -->
          <include file="flat_pusher_arena_inc.xml"/>
          <!-- Target -->
          <include file="flat_pusher_target_inc.xml"/>
        @[end if]
		<!-- Robot -->
        <include file="flat_pusher_robot_inc.xml"/>
	</worldbody>
	<actuator>
		<motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="1000.0" joint="joint0"/>
		<motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="1000.0" joint="joint1"/>
		<motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="1000.0" joint="joint2"/>
		<motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="1000.0" joint="joint3"/>
	</actuator>
</mujoco>
