# syntax=docker/dockerfile:1.4
FROM amazonlinux:2023

# Install pre-reqs
RUN yum update -y && yum install shadow-utils cmake g++ gcc python3 python3-pip python3-setuptools -y
RUN pip3 --version

# Create the user
ARG USERNAME=ec2-user
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME
RUN useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

WORKDIR /app

# # Install AWS CLI
# RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
# RUN unzip awscliv2.zip
# RUN ./aws/install
# RUN aws --version

# Copy files
COPY requirements.txt /app
COPY home.py /app

COPY config.yaml /app
COPY image.png /app

RUN pip3 install -r requirements.txt
RUN pip3 install pandas numpy plotly scikit-learn spacy
RUN python3 -m spacy download en_core_web_sm

RUN chown $USERNAME:$USERNAME /app

USER $USERNAME

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "home.py", "--server.port=8501", "--server.address=0.0.0.0"]
