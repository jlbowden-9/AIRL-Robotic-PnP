import pybullet as p
import pybullet_data as pd
import math
import time
import numpy as np

p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())

textureId = -1

useProgrammatic = 1
useTerrainFromPNG = 0
useDeepLocoCSV = 0
updateHeightfield = False

heightfieldSource = useProgrammatic
import random
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
heightPerturbationRange = 0.05
if heightfieldSource==useProgrammatic:
  numHeightfieldRows = 256
  numHeightfieldColumns = 256
  heightfieldData = [0]*numHeightfieldRows*numHeightfieldColumns 
  for j in range (int(numHeightfieldColumns/2)):
    for i in range (int(numHeightfieldRows/2) ):
      height = random.uniform(0,heightPerturbationRange)
      heightfieldData[2*i+2*j*numHeightfieldRows]=height
      heightfieldData[2*i+1+2*j*numHeightfieldRows]=height
      heightfieldData[2*i+(2*j+1)*numHeightfieldRows]=height
      heightfieldData[2*i+1+(2*j+1)*numHeightfieldRows]=height

  orn90z = p.getQuaternionFromEuler([0, 0, math.pi / 2])
  terrainShape = p.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD, meshScale=[.05,.05,1], collisionFrameOrientation = orn90z, heightfieldTextureScaling=(numHeightfieldRows-1)/2, heightfieldData=heightfieldData, numHeightfieldRows=numHeightfieldRows, numHeightfieldColumns=numHeightfieldColumns)
  terrain  = p.createMultiBody(0, terrainShape)
  p.resetBasePositionAndOrientation(terrain,[0,0,0], [0,0,0,1])


p.changeVisualShape(terrain, -1, rgbaColor=[1,1,1,1])


sphereRadius = 0.05
colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
colBoxId = p.createCollisionShape(p.GEOM_BOX,
                                  halfExtents=[sphereRadius, sphereRadius, sphereRadius])

mass = 1
visualShapeId = -1

link_Masses = [1]
linkCollisionShapeIndices = [colBoxId]
linkVisualShapeIndices = [-1]
linkPositions = [[0, 0, 0.11]]
linkOrientations = [[0, 0, 0, 1]]
linkInertialFramePositions = [[0, 0, 0]]
linkInertialFrameOrientations = [[0, 0, 0, 1]]
indices = [0]
jointTypes = [p.JOINT_REVOLUTE]
axis = [[0, 0, 1]]

for i in range(3):
  for j in range(3):
    for k in range(3):
      basePosition = [
          i * 5 * sphereRadius, j * 5 * sphereRadius, 1 + k * 5 * sphereRadius + 1
      ]
      baseOrientation = [0, 0, 0, 1]
      if (k & 2):
        sphereUid = p.createMultiBody(mass, colSphereId, visualShapeId, basePosition,
                                      baseOrientation)
      else:
        sphereUid = p.createMultiBody(mass,
                                      colBoxId,
                                      visualShapeId,
                                      basePosition,
                                      baseOrientation,
                                      linkMasses=link_Masses,
                                      linkCollisionShapeIndices=linkCollisionShapeIndices,
                                      linkVisualShapeIndices=linkVisualShapeIndices,
                                      linkPositions=linkPositions,
                                      linkOrientations=linkOrientations,
                                      linkInertialFramePositions=linkInertialFramePositions,
                                      linkInertialFrameOrientations=linkInertialFrameOrientations,
                                      linkParentIndices=indices,
                                      linkJointTypes=jointTypes,
                                      linkJointAxis=axis)

      
      p.changeDynamics(sphereUid,
                       -1,
                       spinningFriction=0.001,
                       rollingFriction=0.001,
                       linearDamping=0.0)
      for joint in range(p.getNumJoints(sphereUid)):
        p.setJointMotorControl2(sphereUid, joint, p.VELOCITY_CONTROL, targetVelocity=1, force=10)

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
p.setGravity(0, 0, -10)
p.setRealTimeSimulation(1)

p.getNumJoints(sphereUid)
for i in range(p.getNumJoints(sphereUid)):
  p.getJointInfo(sphereUid, i)


while (p.isConnected()):
  keys = p.getKeyboardEvents()
  
  if updateHeightfield and heightfieldSource==useProgrammatic:
    for j in range (int(numHeightfieldColumns/2)):
      for i in range (int(numHeightfieldRows/2) ):
        height = random.uniform(0,heightPerturbationRange)#+math.sin(time.time())
        heightfieldData[2*i+2*j*numHeightfieldRows]=height
        heightfieldData[2*i+1+2*j*numHeightfieldRows]=height
        heightfieldData[2*i+(2*j+1)*numHeightfieldRows]=height
        heightfieldData[2*i+1+(2*j+1)*numHeightfieldRows]=height
    #GEOM_CONCAVE_INTERNAL_EDGE may help avoid getting stuck at an internal (shared) edge of the triangle/heightfield.
    #GEOM_CONCAVE_INTERNAL_EDGE is a bit slower to build though.
    #flags = p.GEOM_CONCAVE_INTERNAL_EDGE
    flags = 0
    terrainShape2 = p.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD, flags = flags, meshScale=[.05,.05,1], heightfieldTextureScaling=(numHeightfieldRows-1)/2, heightfieldData=heightfieldData, numHeightfieldRows=numHeightfieldRows, numHeightfieldColumns=numHeightfieldColumns, replaceHeightfieldIndex = terrainShape)
    

  #print(keys)
  #getCameraImage note: software/TinyRenderer doesn't render/support heightfields!
  #p.getCameraImage(320,200, renderer=p.ER_BULLET_HARDWARE_OPENGL)
  time.sleep(0.01)
