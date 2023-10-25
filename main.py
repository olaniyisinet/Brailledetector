# import pywhatkit as pywhatkit
# import speech_recognition as sr
# import pyttsx3
import tempfile
import numpy as np
import streamlit as st
from gtts import gTTS
from io import BytesIO
import cv2
import tensorflow as tf

try:

    from tensorflow.keras.models import load_model

    sound_file = BytesIO()

    st.title('Braille Detector')
    st.markdown("<h3 style='text-align: center; '></h3>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; '>This program checks an image to determine if there is braille within an image</p>",
        unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; '></h3>", unsafe_allow_html=True)



    def predict_upload():
        resize = tf.image.resize(cv2_imgg, (120, 120))
        print('got here4')

        yhat = new_model.predict(np.expand_dims(resize / 120, 0))
        print(yhat)
        ans = round(yhat[0][0][0] * 100, 2)
        print('probability value:', yhat[0])

        sample_coords = yhat[1][0]
        frame = cv2_imgg

        if yhat[0] < confidence:
            text = f'The uploaded image does not have braille ❌.\n There is a {ans}% chance that this picture has ' \
                   f'braille '
            st.error(text)
            print(text)

            # speech to text
            tts = gTTS(
                f'The uploaded image does not have braille. There is a {ans}% chance that this picture has braille.',
                lang='en')
            tts.write_to_fp(sound_file)
            st.audio(sound_file)
            # talk('image does not have braille')



        elif yhat[0] > confidence:
            st.snow()
            text = f'Predicted class is braille ✔. \n There is a {ans}% chance that this picture has braille'
            print(text)
            st.success(text)
            tts = gTTS(f'Predicted class is braille. There is a {ans}% chance that this picture has braille', lang='en')
            tts.write_to_fp(sound_file)
            st.audio(sound_file)
            st.write('To interpret the braille, ABEG head-over to my senior man')
            st.markdown('<a href = "https://angelina-reader.ru/">or Visit Angelinas website </a>',
                        unsafe_allow_html=True)
            # Controls the main rectangle
            cv2.rectangle(frame,
                          tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
                          tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)),
                          (255, 0, 0), 2)
            # Controls the label rectangle
            cv2.rectangle(frame,
                          tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                       [0, -30])),
                          tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                       [80, 0])),
                          (255, 0, 0), -1)

            # Controls the text rendered
            cv2.putText(frame, f'braille', tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                                        [0, -5])),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'confidence: {ans} %',
                        tuple(np.add(np.multiply(sample_coords[:2], [450, 350]).astype(int),
                                     [0, -5])),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)

            # print out image to web page
            st.image(frame, caption='Annotated Image')


    # caching the data as the function will run repeatedly
    @st.cache_data
    def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]
        if width is None and height is None:
            return image
        if width is None:
            r = width / float(w)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        resized = cv2.resize(image, dim, interpolation=inter)
        return resized


    # func to save BytesIO on a drive
    def write_bytesio_to_file(filename, bytesio):
        """
        Write the contents of the given BytesIO to a file.
        Creates the file or overwrites the file if it does
        not exist yet.
        """
        with open(filename, "wb") as outfile:
            # Copy the BytesIO stream to the output file
            outfile.write(bytesio.getbuffer())

    #function to predict a video upload
    def predict_vid_upload():
        # frame = cv2_imgg
        fps = 0
        i = 0
        stframe = st.empty()

        #analyse each frame and apply something on the frames
        while video.isOpened():
            i += 1
            ret, frame = video.read()
            if not ret:
                continue
            print('got here4')
            frame = frame[50:500, 50:500, :]

            # converting from bgr to rgb
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # resizing the image to image size 120 (trained my model on this size[it was faster to train on smaller pixels])
            resized = tf.image.resize(rgb, (120, 120))
            #predicting per frame
            yhat = new_model.predict(np.expand_dims(resized / 255, 0))
            ans = round(yhat[0][0][0] * 100, 2)
            print('probability value:', yhat[0])

            # getting feature coordinates bbox from prediction
            sample_coords = yhat[1][0]

            if yhat[0] > confidence:
                # Controls the main rectangle
                cv2.rectangle(frame,
                              tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
                              tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)),
                              (255, 0, 0), 2)

                # Controls the label rectangle
                cv2.rectangle(frame,
                              tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                           [0, -30])),
                              tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                           [80, 0])),
                              (255, 0, 0), -1)

                # Controls the text rendered
                cv2.putText(frame, f'braille', tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int),
                                                            [0, -5])),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f'confidence: {ans} %',
                            tuple(np.add(np.multiply(sample_coords[:2], [450, 350]).astype(int),
                                         [0, -5])),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)

                print(i)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # stframe.image(gray)

                if i == 113:
                    break
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break

            #creating a standard dimension for each frame
            frame1 = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
            frame1 = image_resize(image=frame, width=640)
            stframe.image(frame1, channels='BGR', use_column_width=True)


        out_mp4.release()
        video.release()

        col1, col2 = st.columns(2)
        col1.header("Original Video")
        # col1.video(temp_file_to_save)
        col2.header("Output from OpenCV (MPEG-4)")

    #     -------------------------------------------sidebar-----------------------------------------------------------------
    with st.sidebar:
        st.markdown('---')
        file_type = st.sidebar.selectbox('Choose a method to upload file',
                                         ['Upload an image', 'Take a photo', 'Upload a video'])

        confidence = st.sidebar.slider('Braille confidence', min_value=0.5, max_value=1.0, value=0.6)

        st.markdown('---')
        st.image("https://upload.wikimedia.org/wikipedia/commons/4/4c/Braille_closeup.jpg",
                 caption='An example of a braille')
        st.markdown('---')
        st.markdown('Keele Group coursework, 2023. Do not Copy!')
        # feedback = st.sidebar.write()


    # -------------------------------------------------Body


    #
    # col1, col2, col3 = st.columns(3)
    #
    # with col1:
    #     st.header("A cat")
    #     st.image("https://static.streamlit.io/examples/cat.jpg")
    #
    # with col2:
    #     st.header("A dog")
    #     st.image("https://static.streamlit.io/examples/dog.jpg")
    #
    # with col3:
    #     st.header("An owl")
    #     st.image("https://static.streamlit.io/examples/owl.jpg")

    # -----------------------------------------------Drop Down Menu---------------------------------------------------------------------------------
    if file_type == 'Take a photo':
        st.write('To take a photo, click on capture below \n note camera functionality is limited at the moment')
        img_file_buffer = st.camera_input("Take a picture")
        st.write('nice face!!!')
        if img_file_buffer is not None:
            st.image(img_file_buffer)
            print('got here23')
            # To read image file buffer with OpenCV:
            bytes_data = img_file_buffer.getvalue()
            cv2_imgg = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            print('got here2')
            new_model = load_model('brailledetect2.h5',
                                   compile=False)
            trigger = st.button('Predict', on_click=predict_upload)




    elif file_type == 'Upload an image':
        st.write("Kindly upload an image")
        img_file_buffer = st.file_uploader('Choose a file. Note! Allowed formats are :red[".JPG, JPEG, .PNG"]',
                                           type=["jpg", "jpeg", "png"])

        # uploadingFile()
        print('a')
        if img_file_buffer is not None:
            # To read file as bytes:
            print('got here0')
            bytes_data = img_file_buffer.getvalue()
            cv2_imgg = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            print('got here2')
            new_model = load_model('brailledetect2.h5',
                                   compile=False)
            print('got here2')
            trigger = st.button('Predict', on_click=predict_upload)
        else:
            print('no image')

    elif file_type == 'Upload a video':
        st.write("Kindly upload a video")
        vid = st.file_uploader('Choose a file. Note! Allowed formats are :red["video/mp4"]', ['mp4', 'mov', 'avi'])
        print('heyyhy')
        tfile = tempfile.NamedTemporaryFile(delete=False)

        print('juoooo')
        #
        temp_file_to_save = './temp_file_1.mp4'
        temp_file_result = './temp_file_2.mp4'
        # uploadingFile()
        print('a')
        if vid is not None:
            # save uploaded video to disc
            write_bytesio_to_file(temp_file_to_save, vid)

            tfile.write(vid.read())
            video = cv2.VideoCapture(tfile.name)
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_input = int(video.get(cv2.CAP_PROP_FPS))

            # record
            st.write(width, height, fps_input)

            fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
            out_mp4 = cv2.VideoWriter(temp_file_result, fourcc_mp4, fps_input, (width, height), isColor=False)



            print('got here2')
            new_model = load_model('brailledetect2.h5',
                                   compile=False)
            print('got here2')
            trigger = st.button('Predict', on_click=predict_vid_upload)




    tab1, tab2, tab3 = st.tabs(["Instructions", "Braille Alphabets", "Braille on sign"])

    with tab1:
        st.header("Instructions for Best result")
        st.image("https://angelina-reader.ru/static/images/help/correct_light.png?q=1652031178", width=500)
        st.markdown('''**How to take a photo in a proper way**\n
To get a satisfactory result, you need to follow a few simple rules:\n
the light should fall from the (upper) side of the sheet farthest from you and give a contrasting image of points with light and shadow.\n
The sheet should be photographed from above.\n
The image scale is such that the A4 sheet occupies almost the entire frame.\n
Resolution of at least 1000 points vertically and horizontally (photos from most smartphones and cameras are suitable).''')
        st.write(':red[Follow image below for best the result]')

        col1, col2, col3 = st.columns(3, gap='large')
        #
        with col1:
            st.header("Correct")
            st.image("https://angelina-reader.ru/static/images/help/page_example.jpg?q=1652031178", width=200)

        with col2:
            st.header("Wrong")
            st.image("https://angelina-reader.ru/static/images/help/wrong_image.jpg?q=1652031178", width=200)



    with tab2:
        st.header("Braille Alphabets")
        st.image("https://www.boldbusiness.com/wp-content/uploads/2017/10/Braille-Alphabet.jpg", width=200)
        st.markdown('**What is a Braille?**')
        st.markdown('''Braille :blue[(/breɪl/ BRAYL)] is a tactile writing system used by people who are visually impaired, 
        including people who are blind, deafblind or who have low vision.\n\n It can be read either on embossed paper or 
        by using refreshable braille displays that connect to computers and smartphone devices. Braille can be 
        written using a slate and stylus, a braille writer, an electronic braille notetaker or with the use of a 
        computer connected to a braille embosser.''')


    with tab3:
        st.header("Braille on sign")
        st.image("https://horizonsignco.com/wp-content/uploads/2021/01/adab.jpg", width=200)






except:
    print('errorrrrr21')



 # record
            # codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            #
            # # video = cv2.VideoWriter(output_video_path, fourcc, fps, img_size)
            #
            # # codec = cv2.VideoWriter_fourcc('M', 'J' 'P', 'G')
            # out = cv2.Video('output1.mp4', codec, fps_input, (width,height))

            # v2

            # drawing_spec = mp_drawing.DrawingSpec
            # To read file as bytes:
            # bytes_data = vid.read()
            # print('got here0')

            # bytes_data = img_file_buffer.getvalue()
            # cv2_imgg = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            # cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)