# weights link: https://drive.google.com/file/d/1OAGAXCEQ4L1v14ejWiJCMWYpT5kin4or/view?usp=sharing
DOWNLOAD_DIR = "./checkpoint"
gshell init # gshell required to download files from google drive - https://pypi.org/project/gshell/
gshell download --with-id 1OAGAXCEQ4L1v14ejWiJCMWYpT5kin4or
unzip track2.zip -d ${DOWNLOAD_DIR}
