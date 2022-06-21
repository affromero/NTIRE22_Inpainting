# weights link: https://drive.google.com/file/d/1gS6gMSqcXlN_hDQJT5dXT0QOYp6AjvNZ/view?usp=sharing
DOWNLOAD_DIR = "./checkpoint"
gshell init # gshell required to download files from google drive - https://pypi.org/project/gshell/
gshell download --with-id 1gS6gMSqcXlN_hDQJT5dXT0QOYp6AjvNZ
unzip track1_models.zip -d ${DOWNLOAD_DIR}
