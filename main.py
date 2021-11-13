import open3d as o3d
import cv2
import pyrealsense2 as rs
import numpy as np


if __name__ == '__main__':
    pipeline = rs.pipeline()
    config = rs.config()
    colorizer = rs.colorizer()
    align = rs.align(rs.stream.color)
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    while True:
        frame = pipeline.wait_for_frames()
        aligned_frames = align.process(frame)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        color_image = np.asanyarray(color_frame.get_data())
        colorized_depth = colorizer.colorize(depth_frame)
        depth_image = np.asanyarray(colorized_depth.get_data())
        cv2.imshow('Depth Stream', depth_image)
        c = cv2.waitKey(1)
        if c == 13:
            cv2.destroyAllWindows()
            break