FROM nvcr.io/nvidia/pytorch:21.05-py3
# Install linux packages
RUN apt update && apt install -y zip htop screen
# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
# Copy contents
COPY . /usr/src/app
RUN pip install -r requirements.txt
