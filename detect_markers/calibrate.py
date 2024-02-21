import cv2
import tqdm


def calibrate(
    src: str,
    dst: str,
    dictionary_id: int,
    squares_x: int,
    squares_y: int,
    square_length: float,
    marker_length: float,
    detector_params_file: str | None = None,
    show_img: bool = False,
):
    video_capture = cv2.VideoCapture(src)

    if not video_capture.isOpened():
        return

    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize ChArUco board
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard(
        (squares_y, squares_x), square_length, marker_length, dictionary
    )

    # Initialize ChArUco detector
    charuco_params = cv2.aruco.CharucoParameters()
    detector_params = cv2.aruco.DetectorParameters()
    if detector_params_file:
        detector_params_fs = cv2.FileStorage(
            detector_params_file, cv2.FILE_STORAGE_READ
        )
        if not detector_params_fs.isOpened():
            return
        detector_params.readDetectorParameters(detector_params_fs.root())
    charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)

    all_object_points = []
    all_image_points = []

    progress_ui = tqdm.tqdm(total=n_frames)

    while True:
        ret, frame = video_capture.read()

        progress_ui.update()

        if not ret:
            # End of video
            break

        charuco_corners, charuco_ids, marker_corners, marker_ids = (
            charuco_detector.detectBoard(frame)
        )

        if charuco_corners is None or charuco_corners.size < 4:
            # Not enough corners detected
            continue

        if show_img:
            image_drawn = cv2.aruco.drawDetectedCornersCharuco(
                frame, charuco_corners, charuco_ids
            )
            cv2.imshow("Frame", image_drawn)
            cv2.waitKey(1)

        object_points, image_points = board.matchImagePoints(
            charuco_corners, charuco_ids
        )

        if object_points.size == 0 or image_points.size == 0:
            continue

        all_object_points.append(object_points)
        all_image_points.append(image_points)

    video_capture.release()
    progress_ui.close()

    n_detected_frames = len(all_object_points)

    if n_detected_frames < 4:
        print("Not enough valid frames for calibration")
        return

    reprojection_error, camera_matrix, distortion_coefficients, _, _ = (
        cv2.calibrateCamera(
            all_object_points, all_image_points, (frame_width, frame_height), None, None
        )
    )

    camera_params_fs = cv2.FileStorage(dst, cv2.FILE_STORAGE_WRITE)
    camera_params_fs.write("image_width", frame_width)
    camera_params_fs.write("image_height", frame_height)
    camera_params_fs.write("camera_matrix", camera_matrix)
    camera_params_fs.write("distortion_coefficients", distortion_coefficients)
    camera_params_fs.write("reprojection_error", reprojection_error)

    print(f"{n_detected_frames} frames used for calibration")
    print(f"Reprojection Error: {reprojection_error}")
