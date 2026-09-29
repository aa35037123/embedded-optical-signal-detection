"""BGR frame sources for network transport; no detection runs on the sender."""
import cv2


def frames(args):
    """Yield resized BGR frames, closing camera/video resources on every exit."""
    camera = video = None
    try:
        if args.image:
            frame = cv2.imread(str(args.image))
            if frame is None:
                raise ValueError(f'Cannot read image: {args.image}')
            frame = cv2.resize(frame, (args.width, args.height))
            while True:
                yield frame
        elif args.video:
            video = cv2.VideoCapture(str(args.video))
            if not video.isOpened():
                raise ValueError(f'Cannot open video: {args.video}')
            count = 0
            while True:
                ok, frame = video.read()
                if not ok:
                    if count == 0:
                        raise ValueError('Video returned no frames')
                    break
                count += 1
                yield cv2.resize(frame, (args.width, args.height))
        else:
            from src.capture.camera import CameraAcquisition
            camera = CameraAcquisition(args.width, args.height, args.fps)
            while True:
                yield camera.read()
    finally:
        if camera is not None:
            camera.stop()
        if video is not None:
            video.release()
