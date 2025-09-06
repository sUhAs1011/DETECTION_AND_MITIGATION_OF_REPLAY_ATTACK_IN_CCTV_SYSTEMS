import cv2
import numpy as np
import os


optical_flow_folder = r"C:\Users\Admin\Downloads\pre-process\optical_flow"

video_folder = r"C:\Users\Admin\Downloads\pre-process\original_videos"


optical_flow_files = [f for f in os.listdir(optical_flow_folder) if f.endswith('.npz')]

for flow_file in optical_flow_files:
    video_name = flow_file.replace("_flow.npz", ".mp4")
    video_path = os.path.join(video_folder, video_name)


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_name}")
        continue

    try:
        data = np.load(os.path.join(optical_flow_folder, flow_file))
        flow_data = data['motion_vectors']
    except Exception as e:
        print(f"Error loading {flow_file}: {e}")
        cap.release()
        continue


    fps = int(cap.get(cv2.CAP_PROP_FPS))
    flow_height, flow_width = flow_data.shape[1], flow_data.shape[2]


    os.makedirs("visualized_optical_flow", exist_ok=True)
    output_video_path = os.path.join("visualized_optical_flow", f"{video_name}_optical_flow.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (flow_width, flow_height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= len(flow_data):
            break

   
        frame = cv2.resize(frame, (flow_width, flow_height))

        flow = flow_data[frame_idx]


        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255 

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2  
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  


        optical_flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        blended = cv2.addWeighted(frame, 0.7, optical_flow_vis, 0.3, 0)


        out.write(blended)
        cv2.imshow("Optical Flow Visualization", blended)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    out.release()
    print(f"✅ Saved Optical Flow visualization: {output_video_path}")

cv2.destroyAllWindows()

