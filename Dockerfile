# FULL=true 时安装可选重库（pycausalimpact / tigramite / econml），
# 对应工具走真 BSTS / PCMCI / LinearDML 路径；默认保持苗条镜像。
ARG FULL=false

FROM python:3.11-slim AS runtime-base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

# econml 无 linux/aarch64 wheel，需 gcc 编译 Cython 扩展；
# 用 python:3.11 完整镜像（自带编译工具链）打 wheel，再拷回 slim，
# 避免在 slim 里 apt 安装 build-essential
FROM python:3.11 AS wheels
COPY requirements-full.txt ./
# econml 预生成的 Cython .c 基于 numpy 1.x API，build isolation 会拉
# numpy 2.x 头文件导致编译失败；固定 numpy<2 + 关闭隔离（与运行时
# requirements.txt 的 numpy>=1.26,<2 一致）
RUN pip install "numpy>=1.26,<2" cython setuptools wheel \
    && pip wheel --no-cache-dir --no-build-isolation -w /wheels -r requirements-full.txt

FROM runtime-base AS final-false

FROM runtime-base AS final-true
COPY --from=wheels /wheels /wheels
COPY requirements-full.txt ./
RUN pip install --no-index --find-links=/wheels -r requirements-full.txt \
    && rm -rf /wheels

FROM final-${FULL}
COPY . .

RUN useradd --uid 10001 --no-create-home --home-dir /app appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 50057

CMD ["python3", "server.py", "--host", "0.0.0.0", "--port", "50057"]
