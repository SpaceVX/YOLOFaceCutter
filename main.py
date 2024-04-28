import os
import cv2
import numpy as np
import onnxruntime as ort
import gradio as gr
import torch
from PIL import Image
import logging
from collections import defaultdict
from statistics import mean
import face_recognition
from moviepy.editor import concatenate_videoclips, ImageSequenceClip



# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ort_device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Используется устройство: {device}")

# Загрузка и инициализация модели yoloface_8n.onnx
session = ort.InferenceSession("model/yoloface_8n.onnx")

def preprocess_frame(frame, target_size=(640, 640)):
    frame_resized = cv2.resize(frame, target_size)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_rgb = frame_rgb.astype(np.float32)
    frame_rgb /= 255.0
    frame_rgb = np.transpose(frame_rgb, (2, 0, 1))
    frame_rgb = np.expand_dims(frame_rgb, 0)
    return frame_rgb

def compute_color_histogram(image, bins=8):
    """Compute color histogram of an image."""
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def is_similar(image1, image2, duplicate_rate_threshold):
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    hist1 = compute_color_histogram(image1_rgb)
    hist2 = compute_color_histogram(image2_rgb)

    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return correlation > duplicate_rate_threshold

def process_frame(frame_count, frame, padding, existing_faces, score_face_threshold, duplicate_rate_threshold, faces_directory, score_detect_threshold):
    logger.info(f"Processing frame #{frame_count}")

    frame_preprocessed = preprocess_frame(frame)
    frame_preprocessed = torch.tensor(frame_preprocessed, device=device)

    #logger.info("Executing model.")
    inputs = {session.get_inputs()[0].name: frame_preprocessed.cpu().numpy()}
    outputs = session.run(None, inputs)

    detections = outputs[0]
    faces_detected = 0
    face_confidences_frame = []
    face_images_frame = []

    for detection in detections:
        confidences = detection[:, -1]
        for confidence, bbox in zip(confidences, detection[:, :4]):
            if confidence > score_face_threshold:
                bbox = [int(round(coord)) for coord in bbox]
                left, top, right, bottom = bbox_to_roi(bbox, padding, frame.shape)
                roi = bbox_to_roi(bbox, padding, frame.shape)
                face_frame = frame[roi[1]:roi[3], roi[0]:roi[2]]

                # Zoom in
                zoom_factor = 0.5
                face_height, face_width, _ = face_frame.shape
                face_center_x = bbox[0] + (bbox[2] / 2)
                face_center_y = bbox[1] + (bbox[3] / 2)

                zoomed_width = int((right - left) * zoom_factor)
                zoomed_height = int((bottom - top) * zoom_factor)

                left_x = max(0, left - zoomed_width // 2)
                right_x = min(frame.shape[1] - 1, right + zoomed_width // 2)
                top_y = max(0, top - zoomed_height // 2)
                bottom_y = min(frame.shape[0] - 1, bottom + zoomed_height // 2)

                zoomed_face_frame = frame[top_y:bottom_y, left_x:right_x]
                zoomed_face_frame_rgb = cv2.cvtColor(zoomed_face_frame, cv2.COLOR_BGR2RGB)

                # Check for similar face images, if we don't have any similar ones, add it
                is_new_face = True
                for existing_face in existing_faces:
                    if is_similar(zoomed_face_frame_rgb, existing_face, duplicate_rate_threshold):
                        is_new_face = False
                        #logger.info(f"Duplicate face detected on frame #{frame_count}. Skipping.")
                        break

                if is_new_face:
                    existing_faces.append(zoomed_face_frame_rgb)
                    face_confidences_frame.append(confidence)

                    # Save the zoomed face
                    faces_detected += 1
                    logger.info(f"Face detected with confidence {confidence}. Bbox coordinates: {bbox}")
                    face_image = Image.fromarray(zoomed_face_frame_rgb)
                    face_image_path = os.path.join(faces_directory, f"face_{frame_count}_{faces_detected}.jpg")
                    face_image.save(face_image_path)
                    face_images_frame.append(face_image)
                    logger.info(f"Saved face #{faces_detected} on frame #{frame_count}")

    logger.info(f"Number of faces detected on frame #{frame_count}: {faces_detected}")
    return existing_faces, face_confidences_frame, face_images_frame


def bbox_to_roi(bbox, padding, frame_shape):
    bbox_center = [(bbox[2] + bbox[0]) // 2, (bbox[3] + bbox[1]) // 2]

    roi = [
        max(0, bbox_center[0] - padding),
        max(0, bbox_center[1] - padding),
        min(frame_shape[1], bbox_center[0] + padding),
        min(frame_shape[0], bbox_center[1] + padding),
    ]
    return roi

def track_faces_in_frames(uploaded_video, selected_faces, score_detect_threshold, fps_value):
    # Загрузка изображений лиц и получение их кодировок

    selected_face_images = [face_recognition.load_image_file(face.name) for face in selected_faces]
    selected_face_encodings = [face_recognition.face_encodings(face)[0] for face in selected_face_images if face_recognition.face_encodings(face)]
    logger.info(f"Number of selected faces: {len(selected_face_encodings)}")

    # Open video file
    cap = cv2.VideoCapture(uploaded_video.name)
    logger.info(f"Opening video: {uploaded_video.name}")

    # List of frames with selected face
    fps = fps_value if fps_value else 24
    frames_with_selected_face = []

    # Get video information
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video information: Total frames: {total_frames}, Width: {frame_width}, Height: {frame_height}")

    # Process each frame
    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            logger.info("Reached end of video")
            break

        # Detect faces in the frame
        frame_locations = face_recognition.face_locations(frame)
        if not frame_locations:
            logger.debug(f"No faces detected in frame {frame_num}")
            continue

        # Get face encodings from the frame
        frame_encodings = face_recognition.face_encodings(frame, frame_locations)
        
        # Compare encodings to selected faces
        for frame_encoding in frame_encodings:
            face_distances = face_recognition.face_distance(selected_face_encodings, frame_encoding)
            if len(face_distances) > 0 and min(face_distances) < score_detect_threshold:
                logger.info(f"Face detected in frame {frame_num} with score: {min(face_distances)}")
                frames_with_selected_face.append(frame)
                break  # Break after first match

    # Close video
    cap.release()
    logger.info("Video processing completed")

    # Create video from frames
    if frames_with_selected_face:
        video_output_path = create_video(frames_with_selected_face, fps=fps)
        out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (frame_width, frame_height))
        for frame in frames_with_selected_face:
            out.write(frame)
        out.release()
        logger.info(f"Video created: {video_output_path}")
        message = f"Найдено кадров с лицами: {len(frames_with_selected_face)}"
    else:
        video_output_path = None
        logger.info("No faces found in the video")
        message = "Лица не обнаружены"

    return video_output_path, message

def process_video(uploaded_video, score_face_threshold, duplicate_rate_threshold, score_detect_threshold):
    logger.info("Начало обработки видео.")
    logger.info("Идет процесс сохранения кадров.")
    
    cap = cv2.VideoCapture(uploaded_video.name)
    padding = int(max(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2)

    frames_directory = "temp/frames"  
    os.makedirs(frames_directory, exist_ok=True)  

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Сохраняем все кадры видео
        frame_path = os.path.join(frames_directory, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
        logger.info(f"Кадр {frame_count} сохранён.")
    
    logger.info("Начат процесс поиска лиц")
    cap = cv2.VideoCapture(uploaded_video.name)
    padding = int(max(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2)

    faces_directory = "temp"  # Directory to store face frames
    os.makedirs(faces_directory, exist_ok=True)  # Create the directory if it doesn't exist

    frame_count = 0
    existing_faces = []
    face_confidences = []
    face_images = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        existing_faces, face_confidences_frame, face_images_frame = process_frame(frame_count, frame, padding, existing_faces, score_face_threshold, duplicate_rate_threshold, faces_directory, score_detect_threshold)
        face_confidences.extend(face_confidences_frame)
        face_images.extend(face_images_frame)
        avg_face_score = mean(face_confidences) if face_confidences else 0
        avg_face_score_text = f"AVG Face Score Detect: {avg_face_score:.2f}"
        yield face_images, avg_face_score_text

    cap.release()
    logger.info("Видео обработано. И готово к дальнейшим обработкам.")

def create_video(frames, output_path="output.mp4", fps=24):  
    logger.info("Начало создания нового видео")
    # Убедимся, что все кадры имеют одинаковые размеры
    frame_size = frames[0].shape[:2]
    frames = [frame for frame in frames if frame.shape[:2] == frame_size]
    
    # Создадим видеоклип только если есть кадры
    if frames:
        # Сохраняем кадры во временные файлы JPEG
        temp_files = []
        for i, frame in enumerate(frames):
            temp_file_path = f"frame_{i}.jpeg"
            Image.fromarray(frame).save(temp_file_path)
            temp_files.append(temp_file_path)
        
        # Создаем видеоклип из временных файлов
        clips = [ImageSequenceClip(temp_files, fps=fps)]
        
        # Соединяем клипы и сохраняем видео
        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(output_path, fps=24, codec="libx264", logger=None)  # Используем кодек libx264
        logger.info("Новое видео успешно создано")
        
        # Удаляем временные файлы
        for temp_file in temp_files:
            os.remove(temp_file)
        
        return output_path
    else:
        logger.warning("Нет кадров для создания видео.")
        return None


def gradio_ui():
    blocks = gr.Blocks(theme=gr.themes.Soft(primary_hue="teal", secondary_hue="pink"), title="YOLOFaceCutter 🌊")
    with blocks as demo:
        gr.Markdown("# YOLOFaceCutter 🌊")
        main_info = gr.Markdown("""
                YOLOFaceCutter- это мощный инструмент, который позволяет с лёгкостью редактировать видео, фокусируясь на конкретных лицах.\n
              """)
        with gr.Row():       
            file_input = gr.File(label="Загрузить видео")
            face_images = gr.Gallery(label="Найденные лица", show_label=True)
            face_select = gr.Files(label="Перенесите нужные лица")
            submit_btn = gr.Button("Создать видео")
            video_output = gr.Video(label="Ваше новое видео")

        with gr.Accordion(label="Настройки"):
            score_face_slider = gr.Slider(minimum=0.1, maximum=1000.0, value=0.3, step=0.1, label="Score Face Detect")
            duplicate_rate_slider = gr.Slider(minimum=0.1, maximum=100.0, value=0.9, step=0.1, label="Duplicate Rate")
            avg_face_score_text = gr.Textbox(label="AVG Face Score Detect:")
            
            # Подробные описания настроек
            settings_info = gr.Markdown("""
                **Score Face:** Определяет  Score Face, необходимый для обнаружения лица. Ореинтеровка - AVG.\n
                **Duplicate Rate:** Максимальная частота дублирования кадров с одним лицом.Если много повторных лиц, пробуйте эксперементировать с настройками. \n
                **AVG Face Score Detect:** Средний Score Face обнаруженных лиц. \n
                
              """)
            
        with gr.Accordion(label="Настройки Видео"):                  
            score_detect_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.6, step=0.1, label="Face Score Detect Video:")
            score_detect_text = gr.Textbox(label="АVG Face Score Detect Video:") 
            fps_slider = gr.Slider(minimum=10.0, maximum=60.0, value=24.0, step=1.0, label="FPS")

            # Подробные описания настроек 
            settings_info = gr.Markdown("""
                **Face Score Detect Video:** Определяет  Score Face, необходимый для обнаружения лица в видео. Ореинтеровка - AVG.\n
                **AVG Face Score Detect Video:** Средний Score Face обнаруженных лиц при создании видео. \n
                
              """)

            file_input.change(
                fn=process_video,
                inputs=[file_input, score_face_slider, duplicate_rate_slider, score_detect_slider], 
                outputs=[face_images, avg_face_score_text]
            )

            submit_btn.click(
                fn=track_faces_in_frames,
                inputs=[file_input, face_select, score_detect_slider, fps_slider],
                outputs=[video_output, score_detect_text]
            )
            
        with gr.Accordion(label=""):
            big_block = gr.HTML("""
              <img src="https://i.postimg.cc/qMprpnPT/34.png" style='height: 50%;'>
            """)   

    demo.queue()
    demo.launch(favicon_path="images/icon.ico", inbrowser=True)
    
if __name__ == "__main__":
    gradio_ui()
