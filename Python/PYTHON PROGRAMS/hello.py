import pyqrcode
s = "vijay koppadi"
url = pyqrcode.create(s)
url.svg("myqr.svg", scale=8)