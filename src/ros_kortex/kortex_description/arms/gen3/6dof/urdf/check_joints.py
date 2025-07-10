from urdf_parser_py.urdf import URDF

robot = URDF.from_xml_file("gen3_on_rail.urdf")
for joint in robot.joints:
    if joint.type != "fixed":
        print(f"{joint.name}: {joint.type}")

