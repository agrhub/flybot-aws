# FlyBot vision Lambda: OpenCV 5 (headless) + YOLO26n / YOLOv7-tiny ONNX + the agent, python 3.12 on arm64.
#
#   docker buildx build --platform linux/arm64 -t flybot-vision -f lambda/vision/Dockerfile lambda/vision
#   docker run --rm -p 9000:8080 flybot-vision                         # Lambda RIE
#   curl -s -XPOST localhost:9000/2015-03-31/functions/function/invocations -d @data/samples/request.json
#
# No torch, no onnxruntime: cv2.dnn runs the ONNX. Versions are pinned in requirements-lambda.txt (same as
# uv.lock). The YOLOv7-tiny ONNX (24 MB) is baked in from the public model bucket (SHA-256 checked by
# models/fetch_models.py) so the image works offline; the YOLO26 ONNX is baked in when models/yolo26/ exists in
# the build context (infra/build-image.sh stages it), else fetched at cold start from
# YOLO26_URL into /tmp (sentinel/vision/detector.py). Image target: <= 600 MB (measured by infra/build-image.sh).
FROM public.ecr.aws/lambda/python:3.12-arm64

# OPENCV_FORCE_DNN_ENGINE takes a number (1 classic, 2 new, 3 per call), never "new": the code passes ENGINE_NEW itself
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_DIR=/opt/models

COPY requirements-lambda.txt /tmp/requirements.txt
# No `find` in the AL2023 minimal Lambda base image; --no-compile already avoids .pyc files.
RUN pip install --no-cache-dir --no-compile -r /tmp/requirements.txt && rm -rf /tmp/requirements.txt /root/.cache \
    && rm -rf /var/lang/lib/python3.12/site-packages/cv2/qt /var/lang/lib/python3.12/site-packages/numpy/tests \
              /var/lang/lib/python3.12/site-packages/scipy/*/tests

# Detector weights. The build context (staged by infra/build-image.sh) carries models/fetch_models.py plus any
# fetched models/yolov7-tiny/*.onnx and models/yolo26/*.onnx(.json); whatever is missing for the baseline detector
# is downloaded here from the public model bucket (SHA-256 checked).
COPY models/ /opt/models/
RUN python /opt/models/fetch_models.py --only detector --root /opt && rm -f /opt/models/fetch_models.py \
    && ls -laR /opt/models && python -c "import boto3, cv2; print('boto3', boto3.__version__, 'cv2', cv2.__version__)"

COPY sentinel ${LAMBDA_TASK_ROOT}/sentinel
CMD ["sentinel.handler.handler"]
