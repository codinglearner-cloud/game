import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import random
import time
from PIL import Image, ImageSequence
import mediapipe as mp
from pymongo import MongoClient
import base64

# main10.py 그대로 가져옴옴
#효과음 자동재생함수 base html 가져옴옴
# === MongoDB 연결 ===
client = MongoClient(
    "mongodb+srv://jsheek93:j103203j@cluster0.7pdc1.mongodb.net/"
    "?retryWrites=true&w=majority&appName=Cluster0"
)
db = client['game']
users_col = db['game']

# === 게임 설정 ===
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FLY_GIF_PATH = 'fly_transparent.gif'  # 투명 배경 처리된 파일
SPEED = 5
GAME_DURATION = 60  # 초

# === Blood 이미지 설정 및 크기 조정 ===
blood_bgr = cv2.imread('catched.jpeg', cv2.IMREAD_COLOR)
BLOOD_SCALE = 0.5  # 이미지 축소 비율
blood_bgr = cv2.resize(blood_bgr, None, fx=BLOOD_SCALE, fy=BLOOD_SCALE, interpolation=cv2.INTER_AREA)
bh, bw = blood_bgr.shape[:2]
gray = cv2.cvtColor(blood_bgr, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
mask = mask.astype(float) / 255.0


def play_sound_autoplay(file_path, placeholder):
    placeholder.empty()

    # """file_path에 있는 mp3를 base64로 인코딩해
    #    <audio autoplay> 태그를 placeholder에 삽입합니다."""
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    html = f"""
    <audio autoplay hidden>
      <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    placeholder.html(html)  
# , height=0
catch_snd_pl = st.empty()
bite_snd_pl = st.empty()

BLOOD_DURATION = 0.2  # 초
last_catch_time = None
catch_pos = None  # 피 효과를 표시할 위치

st.title("🦟 벌레레 잡기 게임")

if 'username' not in st.session_state:
    st.session_state.username = None
# 로그인 / 회원가입
if st.session_state.username is None:
    auth_mode = st.sidebar.radio("접속 모드 선택", ['로그인', '회원가입'])
    if 'username' not in st.session_state:
        st.session_state.username = None

    if auth_mode == '회원가입':
        st.sidebar.subheader("회원가입")
        new_user = st.sidebar.text_input("사용자 이름", key="signup_user")
        new_pass = st.sidebar.text_input("비밀번호", type='password', key="signup_pass")
        new_pass_confirm = st.sidebar.text_input("비밀번호 확인", type='password', key="signup_pass_confirm")

        if not new_user or not new_pass:
            st.sidebar.warning("개인정보를 입력해주세요")

        else:
            if st.sidebar.button("회원가입", key="do_signup"):
                if users_col.find_one({'username': new_user}):
                    st.sidebar.error("이미 존재하는 사용자입니다.")

                
                elif new_pass != new_pass_confirm:
                    st.sidebar.warning("비밀번호를 다시 확인해 주세요")

                else:
                    users_col.insert_one({'username': new_user, 'password': new_pass, 'score': None})
                    st.sidebar.success("회원가입 성공! 로그인 해주세요.")
                    
    elif auth_mode == '로그인':
        st.sidebar.subheader("로그인")
        user = st.sidebar.text_input("아이디", key="login_user")
        password = st.sidebar.text_input("비밀번호", type='password', key="login_pass")
        if st.sidebar.button("로그인", key="do_login"):
            
            if not user or not password:
                st.sidebar.warning("아이디와 비밀번호를 입력해 주세요")

            else:

                found = users_col.find_one({'username': user, 'password': password})
                if found:
                    st.session_state.username = user
                    
                    st.rerun()
                else:
                    st.sidebar.error("로그인 실패: 잘못된 정보입니다.")
else:
    st.sidebar.success(f"{st.session_state.username}님 환영합니다!")
    st.sidebar.write(f"👤 로그인: {st.session_state.username}")





# 랭킹 보기
if st.session_state.username:
    if st.sidebar.button("🏆 랭킹 보기", key="show_ranking"):
        st.subheader("🏆 최고 점수 랭킹")
        ranking = users_col.find({'score': {'$ne': None}}).sort('score', -1).limit(10)
        for i, u in enumerate(ranking, start=1):
            st.write(f"{i}등 - {u['username']} : {u['score']}점")
    if st.sidebar.button("로그아웃", key="log_out"):
        st.session_state.clear()
       
        st.rerun()
# 게임 시작
if st.session_state.username:
    



    if st.button('게임 시작', key="start_game"):
        stop_button = st.button("게임 종료", key="end_game")


        # 초기값
        score = 0; caught = False; face_penalty = False
        last_catch_time = None; catch_pos = None
        start_time = time.time(); game_running = True
        

        # GIF & Mediapipe 준비
        pil_gif = Image.open(FLY_GIF_PATH)
        frames = [cv2.cvtColor(np.array(f.convert('RGBA')), cv2.COLOR_RGBA2BGRA)
                  for f in ImageSequence.Iterator(pil_gif)]
        frame_count = len(frames)
        mpHands = mp.solutions.hands
        hands = mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        mpDraw = mp.solutions.drawing_utils
        face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5)
        cap = cv2.VideoCapture(0)

        # 파리 초기 위치
        fly_x = random.randint(0, WINDOW_WIDTH - frames[0].shape[1])
        fly_y = random.randint(0, WINDOW_HEIGHT - frames[0].shape[0])
        dx = random.choice([-1, 1]); dy = random.choice([-1, 1])
        frame_index = 0

        # 화면 플레이스홀더
        video_pl = st.empty(); score_pl = st.empty(); timer_pl = st.empty()

        while game_running and not stop_button:
            elapsed = time.time() - start_time
            if elapsed > GAME_DURATION: break

            success, img = cap.read()
            if not success:
                st.error('카메라를 열 수 없습니다.'); break

            img = cv2.flip(img, 1)
            img = cv2.resize(img, (WINDOW_WIDTH, WINDOW_HEIGHT))
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 손 인식
            res_h = hands.process(imgRGB); hand_pts = []
            if res_h.multi_hand_landmarks:
                for hl in res_h.multi_hand_landmarks:
                    for idx, lm in enumerate(hl.landmark):
                        palm_idxs = [0, 1, 5, 9, 13, 17]
                        h, w, _ = img.shape
                        for idx in palm_idxs:
                            lm = hl.landmark[idx]
                            px, py = int(lm.x * w), int(lm.y * h)
                            hand_pts.append((px, py))
                    # 손바닥 윤곽선(녹색) 그리기
                    mpDraw.draw_landmarks(
                        img, hl, mpHands.HAND_CONNECTIONS,
                        mpDraw.DrawingSpec(color=(0,255,0), thickness=2),
                        mpDraw.DrawingSpec(color=(0,128,0), thickness=1)
                    )

            # 파리 위치 업데이트
            fly_frame = frames[frame_index % frame_count]; fh, fw = fly_frame.shape[:2]
            frame_index += 1
            fly_x += dx*SPEED; fly_y += dy*SPEED
            if fly_x < 0 or fly_x+fw > WINDOW_WIDTH: dx*=-1; fly_x = max(0, min(WINDOW_WIDTH-fw, fly_x))
            if fly_y < 0 or fly_y+fh > WINDOW_HEIGHT: dy*=-1; fly_y = max(0, min(WINDOW_HEIGHT-fh, fly_y))

            # blood 표시 여부 판단
            show_blood = last_catch_time and (time.time() - last_catch_time < BLOOD_DURATION)

            # overlay 베이스
            overlay = img.copy()
            # 파리 그리기 (혈흔 표시 중이 아닐 때만)
            if not show_blood:
                alpha_f = fly_frame[:,:,3]/255.0
                for c in range(3):
                    overlay[fly_y:fly_y+fh, fly_x:fly_x+fw, c] = (
                        alpha_f*fly_frame[:,:,c] + (1-alpha_f)*overlay[fly_y:fly_y+fh, fly_x:fly_x+fw, c]
                    )

            # 잡기 판정
            if not caught:
                for cx, cy in hand_pts:
                    if fly_x<=cx<=fly_x+fw and fly_y<=cy<=fly_y+fh:
                        score+=1; caught=True; last_catch_time=time.time(); catch_pos=(cx,cy); play_sound_autoplay('catch.mp3', catch_snd_pl); break
                    
                    # 
            else:
                if all(not(fly_x<=cx<=fly_x+fw and fly_y<=cy<=fly_y+fh) for cx, cy in hand_pts): caught=False

            # 얼굴 감지 감점
            res_f = face_detector.process(imgRGB); faces=[]
            if res_f.detections:
                h, w, _=img.shape
                for det in res_f.detections:
                    bb=det.location_data.relative_bounding_box
                    x1,y1 = int(bb.xmin*w), int(bb.ymin*h)
                    x2,y2 = x1+int(bb.width*w), y1+int(bb.height*h)
                    faces.append((x1,y1,x2,y2)); cv2.rectangle(overlay,(x1,y1),(x2,y2),(255,0,0),2)
            if not face_penalty:
                for x1,y1,x2,y2 in faces:
                    if fly_x<x2 and x1<fly_x+fw and fly_y<y2 and y1<fly_y+fh:
                        score-=1; face_penalty=True;play_sound_autoplay('bite.mp3', bite_snd_pl); break
            else:
                if all(not(fly_x<x2 and x1<fly_x+fw and fly_y<y2 and y1<fly_y+fh) for x1,y1,x2,y2 in faces): face_penalty=False

            # 화면에 보여줄 최종 프레임
            frame_to_show = overlay.copy()
            # blood overlay
            if show_blood and catch_pos:
                x0 = max(0, min(WINDOW_WIDTH-bw, catch_pos[0]-bw//2))
                y0 = max(0, min(WINDOW_HEIGHT-bh, catch_pos[1]-bh//2))
                for c in range(3):
                    frame_to_show[y0:y0+bh, x0:x0+bw, c] = (
                        mask*blood_bgr[:,:,c] + (1-mask)*frame_to_show[y0:y0+bh, x0:x0+bw, c]
                    )

            # 출력
            video_pl.image(cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB), channels='RGB')
            score_pl.text(f'점수: {score}')
            timer_pl.text(f'남은 시간: {int(GAME_DURATION-elapsed)}초')
            time.sleep(0.03)

        cap.release(); st.success("게임 종료!")
        if stop_button:
            st.rerun()
    

        else:
            st.write(f"{st.session_state.username}님의 점수는 {score}점입니다.")
            user_doc = users_col.find_one({'username': st.session_state.username})
            if user_doc['score'] is None or score>user_doc['score']:
                users_col.update_one({'username': st.session_state.username},{'$set':{'score':score}})
                st.success("🎉 최고 기록 갱신!")
            else:
                st.info(f"이전 최고 기록({user_doc['score']}점)을 넘지 못했습니다.")

else:
    st.warning("게임을 시작하려면 먼저 로그인하세요!")
