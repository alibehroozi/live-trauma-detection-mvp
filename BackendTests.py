import requests

BASE_URL = 'http://185.220.224.66:8005'


response = requests.post(BASE_URL + '/upload-dicom', data={'dicomzip'})
assert response, response.reason

response = requests.post(BASE_URL + '/send-telegram', data={"shiersan98"})
assert response, response.reason

response = requests.get(BASE_URL + '/download-report')
assert response, response.reason