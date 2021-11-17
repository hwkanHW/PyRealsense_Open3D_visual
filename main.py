import open3d as o3d
import cv2
import pyrealsense2 as rs
import numpy as np


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

    def get_intrinsics(self):
        profile = self.frame.get_profile()
        return profile.as_video_stream_profile().get_intrinsics()


class Visualization:
    def __init__(self):
        self.visualizer = o3d.visualization.Visualizer()
        self.visualizer.create_window("3D windows")
        self.pointcloud = None
        self.init = False

    def create_pointcloud(self, color_image, depth_image, intrinsics):
        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
        img_color = o3d.geometry.Image(color_image)
        img_depth = o3d.geometry.Image(depth_image)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img_color,
                                                                  img_depth,
                                                                  depth_scale=500,
                                                                  depth_trunc=9.0,
                                                                  convert_rgb_to_intensity=False)
        self.pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)

    def update_geometry(self):
        if not self.init:
            self.visualizer.add_geometry(self.pointcloud)
            self.init = False
        else:
            self.visualizer.update_geometry(self.pointcloud)
        self.visualizer.poll_events()
        self.visualizer.update_renderer()

    def create_and_update(self, color, depth, intrinsics):
        self.create_pointcloud(color, depth, intrinsics)
        self.update_geometry()
        self.pointcloud.clear()

    def draw_geometry(self):
        o3d.visualization.draw_geometries([self.pointcloud])

    def close_window(self):
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
        intrinsics = pipe.get_intrinsics()
        ##
        vis.create_and_update(color, depth, intrinsics)
        ##
        cv2.imshow("color stream", colorized_depth)
        c = cv2.waitKey(1)
        if c == 13:
            cv2.destroyAllWindows()
            vis.close_window()
            break
