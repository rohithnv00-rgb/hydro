from flask import Flask,render_template, request

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/log',methods=['post','get'])
def log():
    msg=request.form['un']
    return render_template('dashboard.html',msg=msg)

app.run(debug=True)
