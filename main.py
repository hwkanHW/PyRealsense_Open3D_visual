import open3d as o3d
import cv2
import pyrealsense2 as rs
import numpy as np
from open3d.open3d.geometry import create_rgbd_image_from_color_and_depth, create_point_cloud_from_rgbd_image


class Pipeline:
    def __init__(self, stream=rs.stream.depth):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(stream)
        self.colorizer = rs.colorizer()
        self.data_read_config()
        self.color_image = None
        self.depth_image = None
        self.colorized_depth_image = None
        self.frame = None

    def pipe_start(self):
        self.pipeline.start(self.config)

    def data_read_config(self):
        self.config.enable_stream(rs.stream.depth, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, rs.format.rgb8, 30)

    def update_frame(self):
        frame = self.pipeline.wait_for_frames()
        self.frame = self.align.process(frame)
        ##
        depth_frame = self.frame.get_depth_frame()
        color_frame = self.frame.get_color_frame()
        colorized_depth_frame = self.colorizer.process(depth_frame)
        ##
        self.color_image = np.asanyarray(color_frame.get_data())
        self.depth_image = np.asanyarray(depth_frame.get_data())
        self.colorized_depth_image = np.asanyarray(colorized_depth_frame.get_data())

    def get_color_image(self):
        return self.color_image

    def get_depth_image(self):
        return self.depth_image

    def get_colorized_depth_image(self):
        return self.colorized_depth_image

    def get_parameters(self):
        profile = self.frame.get_profile()
        return profile.as_video_stream_profile().get_intrinsics()


class Visualization:
    def __init__(self):
        self.visualizer = o3d.Visualizer()
        self.visualizer.create_window("3D windows")
        self.pointcloud = None

    def create_pointcloud(self, color_image, depth_image, intrinsics):
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
        img_color = o3d.geometry.Image(color_image)
        img_depth = o3d.geometry.Image(depth_image)
        rgbd = create_rgbd_image_from_color_and_depth(img_color, img_depth, depth_trunc=9.0, convert_rgb_to_intensity=False)
        self.pointcloud = create_point_cloud_from_rgbd_image(rgbd, pinhole_camera_intrinsic)

    def update_geometry(self):
        self.visualizer.add_geometry(self.pointcloud)
        self.visualizer.poll_events()
        self.visualizer.update_renderer()

    def creat_and_udate(self, color_image, depth_image, intrinsics):
        self.create_pointcloud(color_image, depth_image, intrinsics)
        self.update_geometry()
        self.pointcloud.clear()

    def closeWindows(self):
        self.visualizer.close()


if __name__ == '__main__':
    pipe = Pipeline()
    vis = Visualization()
    pipe.pipe_start()
    while True:
        pipe.update_frame()
        color = pipe.get_color_image()
        depth = pipe.get_depth_image()
        colorized_depth = pipe.get_colorized_depth_image()
        intrinsics = pipe.get_parameters()
        ##
        vis.creat_and_udate(color, depth, intrinsics)
        ##
        cv2.imshow("color stream", colorized_depth)
        c = cv2.waitKey(1)
        if c == 13:
            cv2.destroyAllWindows()
            vis.closeWindows()
            break
