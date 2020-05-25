import urx
from time import sleep


rob = urx.Robot("172.16.174.128")
rob.set_tcp((0, 0, 0.1, 0, 0, 0))
rob.set_payload(2, (0, 0, 0.1))
rob.set_simulation(True)

sleep(0.2)  #leave some time to robot to process the setup commands
#rob.movel((0,0,0.1,0,0,0), relative=True)
try:
    l = 0.05
    v = 0.05
    a = 0.3

    pose = rob.getl()
    print("robot tcp is at: ", pose)
    print("absolute move in base coordinate ")
    pose[2] += l
    rob.movel(pose, acc=a, vel=v)
    print("relative move in base coordinate ")
    rob.translate((0, 0, -l), acc=a, vel=v)
    print("relative move back and forth in tool coordinate")
    rob.translate_tool((0, 0, -l), acc=a, vel=v)
    rob.translate_tool((0, 0, l), acc=a, vel=v)
finally:
    rob.close()