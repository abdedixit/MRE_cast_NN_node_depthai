import numpy as np
import cv2
import depthai as dai
from pathlib import Path

p = dai.Pipeline()

left = p.create(dai.node.MonoCamera)
right = p.create(dai.node.MonoCamera)
manipLeft = p.create(dai.node.ImageManip)
manipRight = p.create(dai.node.ImageManip)
nn = p.create(dai.node.NeuralNetwork)
cast = p.create(dai.node.Cast)
castXout = p.create(dai.node.XLinkOut)

left.setCamera("left")
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

right.setCamera("right")
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

nnBlobPath = "concat_openvino_2022.1_6shave.blob"
nn.setBlobPath(nnBlobPath)
nn.setNumInferenceThreads(2)

castXout.setStreamName("cast")
cast.setOutputFrameType(dai.ImgFrame.Type.BGR888p)

# Linking
left.out.link(nn.inputs['img1'])
right.out.link(nn.inputs['img2'])
nn.out.link(cast.input)
cast.output.link(castXout.input)

# Pipeline is defined, now we can connect to the device
with dai.Device(p) as device:
    qCast = device.getOutputQueue(name="cast", maxSize=4, blocking=False)

    while True:
        inCast = qCast.get()
        assert isinstance(inCast, dai.ImgFrame)
        cv2.imshow("Concated frames", inCast.getCvFrame())

        if cv2.waitKey(1) == ord('q'):
            break
