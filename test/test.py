import requests

resp = requests.post("http://localhost:5000/predict",
                     files={'file': open('mnist_three.png', 'rb')})

print(resp.text)
