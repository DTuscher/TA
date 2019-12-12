from loguru import logger
import json
import numpy as np
import sys
sys.path.append("/usr/local/lib")
try:
    import pyrealsense2 as rs
except ImportError:
    logger.warning("Unable to import pyrealsense2.")


class RealsenseSensor():
    """Class for interacting with a RealSense D400-series sensor.

    pyrealsense2 should be installed from source with the following
    commands:
    
    FIRST TRY PIP INSTALL!!!! If this does not work for you, then go ahead:

    >>> git clone https://github.com/IntelRealSense/librealsense
    >>> cd librealsense
    >>> mkdir build
    >>> cd build
    >>> cmake .. \
        -DBUILD_EXAMPLES=true \
        -DBUILD_WITH_OPENMP=false \
        -DHWM_OVER_XU=false \
        -DBUILD_PYTHON_BINDINGS=true \
        -DPYTHON_EXECUTABLE:FILEPATH=/path/to/your/python/library/ \
        -G Unix\ Makefiles
    >>> make -j4
    >>> sudo make install
    >>> export PYTHONPATH=$PYTHONPATH:/usr/local/lib
    """

    def __init__(self, configFile):
        self.config = json.load(open(configFile))
        self.advanced_cfg = json.load(open(self.config["advanced"]))
        self.advanced_cfg = str(self.advanced_cfg).replace("'", '\"')
        self._running = None,
        self.id = self.config["id"]
        self._pipe = rs.pipeline()
        self._rs_cfg = rs.config()
        self._align = rs.align(rs.stream.color)
        self._intrinsics = {}
        ctx = rs.context()
        self._spatial_filter = rs.spatial_filter()
        self._hole_filter = rs.hole_filling_filter()
        self.depth_cam = True
        self._running = False
        # prints out cams and S/N, which should be defined as id in config
        for d in ctx.devices:
            logger.info("Cam {d} connected.", d=d)
        
        # post-processing filters
        #self._colorizer = rs.colorizer()
        #self._spatial_filter = rs.spatial_filter()
        #self._hole_filling = rs.hole_filling_filter()

    def __del__(self):
        if self._running:
            self.stop()

    def _setup_pipe(self):
        self._rs_cfg.enable_device(self.id)
        self._rs_cfg.enable_stream(
            rs.stream.color,
            640,
            480,
            rs.format.rgb8,
            30
        )
        self._rs_cfg.enable_stream(
            rs.stream.depth,
            640,
            480,
            rs.format.z16,
            30
        )

    def _get_intrinsics(self):
        stream = self._profile.get_stream(rs.stream.color)
        obj = stream.as_video_stream_profile().get_intrinsics()
        self._intrinsics = obj

    def start(self):
        self._setup_pipe()
        self._profile = self._pipe.start(self._rs_cfg)
        self._get_intrinsics()
        self._set_depth_scale()
        for _ in range(5):
            self._pipe.wait_for_frames()
        self._running = True
        self._dev = self._profile.get_device()
        #self.advanced_mode = rs.rs400_advanced_mode(self._dev)
        #self.advanced_mode.load_json(self.advanced_cfg)
        logger.info("Realsense stream started.")
        #logger.info("Advanced mode is {}".format(self.advanced_mode.is_enabled()))

    def stop(self):
        if not self._running:
            return False
        self._pipe.stop()
        self._running = False
        return True

    def _to_numpy(self, frame, dtype):
        arr = np.asanyarray(frame.get_data(), dtype=dtype)
        return arr

    def _read_image(self, spatial=False, hole_filling=False):
        
        frames = self._pipe.wait_for_frames()
        frames = self._align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if spatial:
            depth_frame = self._spatial_filter.process(depth_frame)
        if hole_filling:
            depth_frame = self._hole_filter.process(depth_frame)
        
        depth_image = self._to_numpy(depth_frame, np.float32)

        depth_image *= self._depth_scale
        color_im = self._to_numpy(color_frame, np.uint8)
        return color_im, depth_image
    
    
    def _set_depth_scale(self):
        sensor = self._profile.get_device().first_depth_sensor()
        self._depth_scale = sensor.get_depth_scale()
    
    def frames(self, spatial=False, hole_filling=False):
        color_im, depth_im = self._read_image(spatial, hole_filling)
        return color_im.copy(), depth_im.copy()

