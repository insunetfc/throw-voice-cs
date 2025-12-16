
# 0. Introduction

This document explains how to install FreeSWITCH on Ubuntu 22.04 using a full
source build. Because Ubuntu 22.04 is not officially supported by FreeSWITCH’s
apt repository, the system must be prepared with both apt‑based and
source‑compiled dependencies. This guide provides context, rationale, and
step‑by‑step instructions to ensure a reproducible installation.


# 1. System Preparation

Before building FreeSWITCH, ensure the OS is updated and that required build
tools are present. These tools support autotools, cmake‑based projects, and
codec libraries.

## 1.1 Update System
```
sudo apt update && sudo apt upgrade -y
```

## 1.2 Install Core Build Tools
```
sudo apt install -y git wget curl unzip build-essential g++ automake autoconf libtool cmake pkg-config python3-dev yasm nasm
```

These provide compilation support for dependencies later built from source.


# 2. Standard Dependencies via apt

FreeSWITCH relies on many multimedia and networking libraries. Ubuntu provides
usable versions for several of these, which we install using apt:

```
sudo apt install -y libjpeg-dev libncurses5-dev libssl-dev libpcre3-dev libspeex-dev libspeexdsp-dev libsqlite3-dev libcurl4-openssl-dev libldns-dev libedit-dev libtiff-dev liblua5.2-dev libopus-dev libsndfile1-dev libavformat-dev libavresample-dev libavcodec-dev libavutil-dev libswscale-dev libswresample-dev libasound2-dev libunwind-dev libevent-dev
```

These form the baseline multimedia stack FreeSWITCH expects.


# 3. Source-Built Dependencies

Some Ubuntu packages (notably sofia-sip, spandsp, and low‑level codec libraries)
are too old or incompatible. We must compile patched or modern versions.

## 3.1 libks
```
git clone https://github.com/signalwire/libks.git
cd libks
cmake .
make -j$(nproc)
sudo make install
sudo ldconfig
```

## 3.2 sofia-sip (patched version required)
```
git clone https://github.com/freeswitch/sofia-sip.git
cd sofia-sip
./bootstrap.sh
./configure
make -j$(nproc)
sudo make install
sudo ldconfig
```

## 3.3 spandsp
```
git clone https://github.com/freeswitch/spandsp.git
cd spandsp
./bootstrap.sh
./configure
make -j$(nproc)
sudo make install
sudo ldconfig
```

## 3.4 libtiff
```
wget https://download.osgeo.org/libtiff/tiff-4.5.1.tar.gz
tar -xvf tiff-4.5.1.tar.gz
cd tiff-4.5.1
./configure
make -j$(nproc)
sudo make install
sudo ldconfig
```

## 3.5 libyuv
```
git clone https://chromium.googlesource.com/libyuv/libyuv
cd libyuv
cmake .
make -j$(nproc)
sudo make install
sudo ldconfig
```

## 3.6 libvpx
```
git clone https://chromium.googlesource.com/webm/libvpx
cd libvpx
./configure
make -j$(nproc)
sudo make install
sudo ldconfig
```

These ensure consistent codec and SIP stack behavior.


# 4. Building FreeSWITCH

## 4.1 Clone Repository
```
git clone https://github.com/signalwire/freeswitch.git
cd freeswitch
git checkout v1.10.9
```

## 4.2 Bootstrap & Configure
```
./bootstrap.sh -j
./configure --enable-core-pgsql-support
```

## 4.3 Build
```
make -j$(nproc)
```

## 4.4 Install
```
sudo make install
sudo make cd-sounds-install cd-moh-install
```

The build process compiles core telephony modules, codecs, and XML processing
engines.


# 5. Systemd Service Setup

Create a service for automatic startup:

```
sudo cp debian/freeswitch-systemd.freeswitch.service /etc/systemd/system/freeswitch.service
sudo systemctl daemon-reload
sudo systemctl enable freeswitch
sudo systemctl start freeswitch
```

This allows FreeSWITCH to run persistently and restart on reboot.


# 6. Verification Procedures

## Check Service Status
```
systemctl status freeswitch
```

## Enter CLI
```
fs_cli
```

## Play Test Audio
```
fs_cli -x "playback local_stream://moh"
```

## View Logs
```
tail -f /usr/local/freeswitch/log/freeswitch.log
```

These checks confirm proper installation and media subsystem operation.


# 7. Directory Layout

FreeSWITCH installs into `/usr/local/freeswitch/`:

- **bin/** – binaries (`freeswitch`, `fs_cli`)
- **conf/** – all XML configuration (SIP profiles, dialplans)
- **log/** – runtime logs
- **sounds/** – system prompts and music on hold
- **/var/lib/freeswitch/** – databases, runtime state

Knowing this layout helps with debugging and customization.


# 8. Custom Audio Handling

Convert audio for FreeSWITCH compatibility:

```
ffmpeg -i input.mp3 -ac 1 -ar 8000 output.wav
```

Store files here:

```
/usr/local/freeswitch/sounds/custom/
```

Reference in dialplans:

```
<action application="playback" data="custom/output.wav"/>
```


# 9. Troubleshooting

## Build Fails
Re-run configure and inspect missing libraries:
```
./configure
```

## SIP Fails to Register
Ensure patched sofia-sip is installed.

## No Audio
Check codec negotiation and NAT configuration.

## Missing MOH
```
sudo make cd-moh-install
```

These steps resolve common issues during installation and operation.
