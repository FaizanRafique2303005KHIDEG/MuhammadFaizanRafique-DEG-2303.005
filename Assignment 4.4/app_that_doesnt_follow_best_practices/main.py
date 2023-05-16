import json
import logging
import os
from flask import Flask, render_template, request
from logging import StreamHandler
from config import Config

app = Flask(__name__)
#Load the configuration from the config class
app.config.from_object(Config)

#Load the logs to container output from streamhandler
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
stream_handler = StreamHandler()
root_logger.addHandler(stream_handler)


TODO_FILE_NAME = "/app/save_data/todo.json"
if os.path.exists(TODO_FILE_NAME):
    with open(TODO_FILE_NAME) as f:
        TODO_ITEMS = json.load(f)
else:
    TODO_ITEMS = []

if app.debug:
    app.logger.setLevel(logging.DEBUG)
    app.logger.debug("Debug mode is enabled.")
else:
    app.logger.debug("Debug mode is disabled.")


def save_todo_items(content):
    with open(TODO_FILE_NAME, "r+") as f:
        data = json.load(f)
        data.append(content)
        f.seek(0)
        json.dump(data, f)
        f.truncate()



@app.route("/", methods=["GET", "POST"])
def main():
    if request.method == "POST":
        content = request.form["content"]
        save_todo_items(content)

    return render_template("index.html", todo_items=TODO_ITEMS)



if __name__ == "__main__":
    app.run(host="0.0.0.0")
