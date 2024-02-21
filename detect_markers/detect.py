import cv2
import numpy as np
import pandas as pd
import tqdm


def detect(
    src: str,
    dst: str,
    camera_params_file: str,
    marker_ids: list[int],
    dictionary_id: int,
    marker_length: float,
    detector_params_file: str | None = None,
    show_img: bool = False,
):
    video_capture = cv2.VideoCapture(src)

    if not video_capture.isOpened():
        return

    n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize ArUco detector
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    detector_params = cv2.aruco.DetectorParameters()
    if detector_params_file:
        detector_params_fs = cv2.FileStorage(
            detector_params_file, cv2.FILE_STORAGE_READ
        )
        if not detector_params_fs.isOpened():
            return
        detector_params.readDetectorParameters(detector_params_fs.root())
    aruco_detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    # Get camera params
    camera_params_fs = cv2.FileStorage(camera_params_file, cv2.FILE_STORAGE_READ)
    if not camera_params_fs.isOpened():
        return
    camera_matrix = camera_params_fs.getNode("camera_matrix").mat()
    distortion_coefficients = camera_params_fs.getNode("distortion_coefficients").mat()

    # Corners of marker
    object_points = np.array(
        [
            [-marker_length / 2, marker_length / 2, 0.0],
            [marker_length / 2, marker_length / 2, 0.0],
            [marker_length / 2, -marker_length / 2, 0.0],
            [-marker_length / 2, -marker_length / 2, 0.0],
        ]
    )

    dfs = []
    for id in marker_ids:
        columns = [f"{id}_x", f"{id}_y", f"{id}_z", f"{id}_rx", f"{id}_ry", f"{id}_rz"]
        dfs.append(pd.DataFrame(np.empty([n_frames, 6]), columns=columns))

    times = np.empty(n_frames)

    progress_ui = tqdm.tqdm(total=n_frames)

    while True:
        progress_ui.update()

        ret, frame = video_capture.read()
        pos = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        # End of video
        if not ret:
            # Drop nonexist frames
            for df in dfs:
                df.drop(df.index[pos + 1 :], inplace=True)

            break

        times[pos] = video_capture.get(cv2.CAP_PROP_POS_MSEC) * 1e-3

        corners, ids, _ = aruco_detector.detectMarkers(frame)

        if show_img:
            image_drawn = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.imshow("Frame", image_drawn)
            cv2.waitKey(1)

        for id in marker_ids:
            df = dfs[id]

            # Marker not detected
            if ids is None or id not in ids:
                df.iloc[pos, :] = None
                continue

            corner = corners[np.where(ids == id)[0][0]]

            ret, rvec, tvec = cv2.solvePnP(
                object_points, corner, camera_matrix, distortion_coefficients
            )

            if not ret:
                df.iloc[pos, :] = None
                continue

            df.iloc[pos, :] = np.concatenate([tvec, rvec]).flatten()

    video_capture.release()
    progress_ui.close()

    df_concat = pd.concat(dfs, axis=1)
    df_concat.set_index(times[: len(df_concat)], inplace=True)
    df_concat.to_csv(dst, index=True, index_label="t")
