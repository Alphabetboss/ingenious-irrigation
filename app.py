from flask import Flask, render_template, request, jsonify
from schedule_manager import load_schedule, save_schedule

app = Flask(__name__)

@app.route("/")
def dashboard():
    return render_template("dashboard.html", schedule=load_schedule())

@app.route("/schedule", methods=["POST"])
def update_schedule():
    data = request.json
    save_schedule(data)
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)