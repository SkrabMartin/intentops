FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
WORKDIR /opt/intentops
COPY pyproject.toml README.md LICENSE ./
COPY app ./app
COPY dashboard ./dashboard
COPY ml ./ml
RUN pip install --no-cache-dir . \
    && useradd --create-home --uid 10001 intentops
COPY .streamlit ./.streamlit
COPY THIRD_PARTY_NOTICES.md ./
USER intentops
EXPOSE 8000 8501
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
