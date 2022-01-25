from pyrogram import Client
import sys
from pathlib import Path
import os

WORKINGDIR = os.path.abspath("../aiworkingdir")

api_id= 1614783                                             # account 09173898626
api_hash = '65b3a547d9d2f0b6145a8ad3896e6313'               # account 09173898626

client = Client(session_name='myclient', api_id=api_id, api_hash=api_hash)


temp = f'{WORKINGDIR}/report_final/report.zip'


client.start()
client.send_document(sys.argv[1], f"{temp}")
